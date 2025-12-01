import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input # <--- ВАЖЛИВО: Спец. функція для ResNet
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Налаштування ---
print("Розпочинаємо оцінку моделі ResNet50...")

# ВКАЖІТЬ ПРАВИЛЬНЕ ІМ'Я ФАЙЛУ!
MODEL_PATH = 'retinal_model_resnet_tuned.h5'

# Шляхи до тестових даних
TEST_DIR = 'Test_Set/Test'
TEST_CSV = 'Test_Set/RFMiD_Testing_Labels.csv'

IMG_SIZE = (224, 224)
IMAGE_EXTENSION = '.png'
BATCH_SIZE = 32

# --- 2. Завантаження моделі ---
if not os.path.exists(MODEL_PATH):
    print(f"ПОМИЛКА: Не знайдено файл моделі '{MODEL_PATH}'. Спочатку запустіть train.py!")
    exit()

print(f"Завантажуємо модель {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Модель успішно завантажена.")
except Exception as e:
    print(f"Помилка завантаження: {e}")
    exit()

# --- 3. Підготовка тестових даних ---
print("Завантажуємо тестовий CSV...")
df_test = pd.read_csv(TEST_CSV)

def create_binary_label(row):
    return 'healthy' if row['Disease_Risk'] == 0 else 'pathology'

df_test['binary_label'] = df_test.apply(create_binary_label, axis=1)
df_test['filename'] = df_test['ID'].astype(str) + IMAGE_EXTENSION

print(f"Тестові дані: \n{df_test['binary_label'].value_counts()}")

# --- ВАЖЛИВА ЗМІНА ---
# Для ResNet50 ми НЕ використовуємо rescale=1./255
# Ми використовуємо спеціальну функцію preprocess_input
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input # <--- ЦЕ МАЄ СПІВПАДАТИ З TRAIN.PY
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=TEST_DIR,
    x_col='filename',
    y_col='binary_label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Не перемішувати для коректної оцінки!
)

# --- 4. Отримання прогнозів ---
print(f"Робимо прогнози на {test_generator.n} зображеннях...")

y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred_binary = (y_pred_probs > 0.5).astype(int)

# --- 5. Звіт ---
print("\n--- 📊 ЗВІТ ПРО ОЦІНКУ МОДЕЛІ (ResNet50) ---")
class_labels = list(test_generator.class_indices.keys())
print(f"Класи: {test_generator.class_indices}")

cm = confusion_matrix(y_true, y_pred_binary)
print("\n--- Матриця помилок ---")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nРозшифровка:")
print(f"  True Negative (TN - Здорові визнані здоровими): {tn}")
print(f"  False Positive (FP - Здорові визнані хворими): {fp}")
print(f"  False Negative (FN - Хворі визнані здоровими): {fn} <--- (Пропущені хвороби)")
print(f"  True Positive (TP - Хворі визнані хворими): {tp}")

print("\n--- Детальний звіт (Precision, Recall) ---")
print(classification_report(y_true, y_pred_binary, target_names=class_labels))

print("\nОцінку завершено.")