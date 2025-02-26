import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# โหลดโมเดลที่ฝึกไว้
model = tf.keras.models.load_model("fruit_classifier_vgg16.h5")

# รายชื่อคลาสของผลไม้
labels = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']

# ฟังก์ชันเปิดไฟล์รูปภาพ
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        img = Image.open(file_path).resize((224, 224))  # ปรับขนาดภาพให้ตรงกับโมเดล
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        image_label.file_path = file_path  # เก็บ path ไว้ใช้ตอนทำนาย

# ฟังก์ชันทำนายผลไม้จากภาพ
def recognize_image():
    if hasattr(image_label, "file_path"):  # ตรวจสอบว่ามีภาพที่เลือกหรือไม่
        img_path = image_label.file_path
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class] * 100  # เปลี่ยนเป็นเปอร์เซ็นต์

        result_label.config(text=f"ผลลัพธ์: {labels[predicted_class]} ({confidence:.2f}%)", fg="blue")

# สร้าง GUI ด้วย Tkinter
root = tk.Tk()
root.title("Fruit Classifier")

# ตั้งค่า background และขนาดหน้าต่าง
root.configure(bg="lightblue")
root.geometry("400x500")

# ปุ่มเลือกภาพ
btn_browse = tk.Button(root, text="Browse Image", command=browse_image, font=("Arial", 12), bg="#4CAF50", fg="white", padx=20, pady=10)
btn_browse.pack(pady=20)

# แสดงภาพที่อัปโหลด
image_label = tk.Label(root, bg="white", width=224, height=224)
image_label.pack(pady=10)

# ปุ่มทำนายภาพ
btn_recognize = tk.Button(root, text="Recognize", command=recognize_image, font=("Arial", 12), bg="#008CBA", fg="white", padx=20, pady=10)
btn_recognize.pack(pady=20)

# แสดงผลลัพธ์
result_label = tk.Label(root, text="อัปโหลดภาพแล้วกด Recognize", font=("Arial", 14), bg="lightblue")
result_label.pack(pady=10)

# เริ่ม GUI
root.mainloop()
