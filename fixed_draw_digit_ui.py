
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import messagebox  # Fixed import

# Load the trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Set up the canvas
width, height = 200, 200
white = (255, 255, 255)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.canvas = tk.Canvas(self, width=width, height=height, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = Image.new("RGB", (width, height), white)
        self.draw = ImageDraw.Draw(self.image1)

        tk.Button(self, text="Predict", command=self.predict_digit).pack()
        tk.Button(self, text="Clear", command=self.clear_canvas).pack()

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

def predict_digit(self):
    try:
        print("[INFO] Starting prediction...")

        img = self.image1.convert('L')
        img = ImageOps.invert(img)
        img = img.resize((28, 28), Image.ANTIALIAS)

        print("[INFO] Image preprocessed")

        img = np.array(img).astype('float32') / 255.0
        img = img.reshape(1, 28, 28, 1)

        print("[INFO] Image reshaped:", img.shape)

        prediction = model.predict(img)
        digit = np.argmax(prediction)
        prob = np.max(prediction)

        print(f"[INFO] Prediction result: {digit} (Confidence: {prob:.2f})")
        messagebox.showinfo("Prediction", f"Digit: {digit}\nConfidence: {prob:.2f}")

    except Exception as e:
        print("[ERROR] Something went wrong:", e)


    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, width, height], fill=white)

if __name__ == '__main__':
    app = App()
    app.mainloop()
