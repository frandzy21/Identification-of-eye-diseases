import tensorflow as tf
import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.utils import resample

#1.Налаштування
print("Розпочинаємо 'Розумне навчання' (ResNet50 + Hyperparameters)...")

TRAIN_DIR = 'Training_Set/Training'
TRAIN_CSV = 'Training_Set/RFMiD_Training_Labels.csv'
VALID_DIR = 'Evaluation_Set/Validation'
VALID_CSV = 'Evaluation_Set/RFMiD_Validation_Labels.csv'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
IMAGE_EXTENSION = '.png'
EPOCHS = 20

#2.Підготовка даних (з Oversampling)
print("Завантажуємо та балансуємо дані...")
df_train = pd.read_csv(TRAIN_CSV)
df_valid = pd.read_csv(VALID_CSV)

def create_binary_label(row):
    return 'healthy' if row['Disease_Risk'] == 0 else 'pathology'

df_train['binary_label'] = df_train.apply(create_binary_label, axis=1)
df_valid['binary_label'] = df_valid.apply(create_binary_label, axis=1)
df_train['filename'] = df_train['ID'].astype(str) + IMAGE_EXTENSION
df_valid['filename'] = df_valid['ID'].astype(str) + IMAGE_EXTENSION

df_majority = df_train[df_train.binary_label == 'pathology']
df_minority = df_train[df_train.binary_label == 'healthy']

df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
df_train_balanced = pd.concat([df_majority, df_minority_upsampled])
df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Дані збалансовано: {df_train_balanced['binary_label'].value_counts()}")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train_balanced, directory=TRAIN_DIR, x_col='filename', y_col='binary_label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

validation_generator = valid_datagen.flow_from_dataframe(
    dataframe=df_valid, directory=VALID_DIR, x_col='filename', y_col='binary_label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

#4.Модель з Fine-Tuning
print("Налаштовуємо архітектуру ResNet50...")

base_model = ResNet50(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#5.Callbacks (Розумне навчання)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.000001)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

checkpoint = ModelCheckpoint('retinal_model_resnet_tuned.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

#6.Навчання
print("Компілюємо та запускаємо...")
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[lr_reduction, early_stopping, checkpoint]
)

print("Навчання завершено. Найкраща модель збережена у 'retinal_model_resnet_tuned.h5'")