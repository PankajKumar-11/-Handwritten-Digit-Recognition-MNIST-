# Handwritten Digit Recognition (MNIST)

This project is a simple interactive application that allows you to draw digits from 0 to 9 on a canvas, and uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to predict the drawn digit.

## 🖼️ Samples
Here are some sample drawings recognized by the model:

![Four](four.png)
![Three](three.png)

## 🛠️ Technologies Used
- **Python 3**
- **TensorFlow / Keras** (for the CNN model)
- **Tkinter** (for the interactive drawing GUI)
- **Pillow (PIL)** (for image processing)
- **NumPy** & **Matplotlib**

## 🚀 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PankajKumar-11/-Handwritten-Digit-Recognition-MNIST-.git
   cd -Handwritten-Digit-Recognition-MNIST-
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   Start the interactive GUI by running the graphical script:
   ```bash
   python fixed_draw_digit_ui.py
   ```

## 🎮 How to Use
1. A small window will appear with a white canvas.
2. Use your mouse to click and drag to draw a single digit (0 through 9).
3. Click the **Predict** button to let the neural network analyze your drawing.
4. An alert/message box will pop up, or the window UI will update to show you the predicted digit and its confidence probability!
5. Click **Clear** to wipe the canvas to try again.

## 📁 Repository Structure
- `digit_recognition.ipynb`: Jupyter Notebook containing the code and process used to train the CNN model on the MNIST dataset.
- `mnist_cnn_model.h5` / `mnist_cnn_model.keras`: The saved pre-trained model files.
- `fixed_draw_digit_ui.py` / `draw_digit_ui.py`: The Tkinter user interface scripts that load the model and let you draw and predict.
- `requirements.txt`: Python package dependencies.
