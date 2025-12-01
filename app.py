import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input  # <--- ВАЖЛИВО для ResNet
import os

#1.Завантаження навченої моделі (ResNet50)
MODEL_PATH = 'retinal_model_resnet_tuned.h5'

print(f"Завантажуємо збережену модель: {MODEL_PATH}...")

if not os.path.exists(MODEL_PATH):
    print(
        f"ПОМИЛКА: Не вдалося знайти модель. Переконайтеся, що файл '{MODEL_PATH}' знаходиться в тій самій папці, що й app.py")
    exit()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Модель ResNet50 успішно завантажена.")
except Exception as e:
    print(f"ПОМИЛКА: Не вдалося завантажити модель.")
    print(f"Деталі помилки: {e}")
    exit()

class_names = ['Здорове око', 'Око з патологією']


#2.Функція для передбачення
def predict_image(input_img):
    """
    Ця функція приймає зображення від Gradio, обробляє його
    та повертає прогноз моделі.
    """
    try:
        img = input_img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)[0][0]

        if prediction < 0.5:
            confidence = 1 - prediction
            return {class_names[0]: float(confidence), class_names[1]: float(prediction)}
        else:
            confidence = prediction
            return {class_names[0]: float(1 - prediction), class_names[1]: float(confidence)}

    except Exception as e:
        print(f"Помилка під час передбачення: {e}")
        return {"Помилка": str(e)}


#3.Інтерфейс Gradio

# Заголовки та опис
app_title = " Система діагностики сітківки ока"

app_description = """
Ласкаво просимо! Ця програма є прототипом для курсової роботи. 
Вона використовує потужну згорткову нейронну мережу для аналізу зображень сітківки ока.
**Мета програми:** визначити, чи є на зображенні загальні ознаки патології (ризик захворювання), чи око є здоровим.
"""

app_instructions = """
### Інструкція з використання
1.  **Завантажте зображення:** Натисніть на поле "Завантажте фото сітківки" або просто перетягніть файл у цю область.
2.  **Формат:** Програма найкраще працює зі знімками, схожими на ті, що були у тренувальному наборі (формату `.png`, `.jpg` або `.jpeg`).
3.  **Отримайте результат:** Модель проаналізує фото (це може зайняти кілька секунд) і видасть результат у відсотках.

---
### Важливе зауваження
Ця програма є **навчальним проектом** і **не є сертифікованим медичним інструментом**.

* Результат **'Око з патологією'** не означає, що у вас 100% є хвороба, а лише те, що модель побачила ознаки, *схожі* на хворі очі з датасету.
* Результат **'Здорове око'** також не гарантує відсутності хвороби (модель має відомий рівень помилок).

**Завжди консультуйтеся з кваліфікованим лікарем** для будь-якої медичної діагностики.
"""

print("Запускаємо інтерфейс Gradio...")

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Завантажте фото сітківки"),
    outputs=gr.Label(num_top_classes=2, label="Результат діагностики"),

    title=app_title,
    description=app_description,
    article=app_instructions,

    examples=[
        'Test_Set/Test/1.png',
        'Test_Set/Test/2.png'
    ]
)

if __name__ == "__main__":
    iface.launch(share=True)