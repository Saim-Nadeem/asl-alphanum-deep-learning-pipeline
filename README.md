# 🤟 ASL Hand Gesture Recognition

License: MIT  
Built with: Python · PyTorch · OpenCV · Tkinter GUI

An end-to-end deep learning project for real-time American Sign Language (ASL) hand gesture recognition. Includes image preprocessing, CNN training, evaluation metrics, and a drag-and-drop GUI for interactive predictions.

---

🔍 FEATURES

- 🎯 High Accuracy CNN classifier for 39 ASL gestures (A–Z, 0–10, space, nothing, del)
- 🛠️ Image Preprocessing: Background removal, resizing, normalization
- 📊 Model Training & Evaluation: Accuracy plots, confusion matrix
- 🖼️ Interactive GUI: Drag & drop single images or entire folders
- 💾 Model Persistence: Save and reload trained PyTorch models

---

📦 PROJECT STRUCTURE

├── Model_Make.ipynb           # Notebook for training the CNN model  
├── GUI.ipynb                  # Notebook for launching drag-and-drop GUI  
├── preprocess.py              # Script to clean, resize, and structure dataset  
├── requirements.txt           # List of dependencies  
├── asl_cnn_39class_cpu.pth    # Saved trained model (PyTorch)  
├── Preprocessed_data/         # Folder with preprocessed images per class  
├── test/                      # Folder with one test image per class  
└── README.md                  # Project documentation  

---

🧠 MODEL OVERVIEW

- Input Shape: 32x32 RGB  
- Architecture: Custom CNN with 3 conv blocks + dropout + batch norm  
- Classes: 39 total  
- Final Test Accuracy: 98.57%

> Model trained on CPU with over 42,000 images across all classes.

---

📁 DATASET

This project combines multiple datasets:

ASL Alphabet – Hand signs A–Z  
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

ASL Numbers – Synthetic hand signs 0–10  
https://www.kaggle.com/datasets/lexset/synthetic-asl-numbers

---

🧪 TECH STACK

Model Training: PyTorch, NumPy, OpenCV  
GUI: Tkinter, TkinterDnD2, Pillow  
Data Processing: Pandas, Matplotlib, Seaborn  
Evaluation: Scikit-learn, Confusion Matrix

---

🚀 GETTING STARTED

1️⃣ Clone the Repository

git clone https://github.com/your-username/asl-gesture-recognition.git  
cd asl-gesture-recognition

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Preprocess the Dataset

python preprocess.py

This script:
- Removes background
- Splits train/test
- Generates `test/` folder with sample images

4️⃣ Train the CNN

jupyter notebook Model_Make.ipynb  
(Train the model and save it as asl_cnn_39class_cpu.pth)

5️⃣ Launch the GUI

jupyter notebook GUI.ipynb  
Then drag and drop:
- A single image to get prediction and confidence
- A folder to get batch predictions

---

🧾 REQUIREMENTS

# Core libraries
numpy==1.26.4  
pandas==2.2.2  
opencv-python==4.9.0.80  
matplotlib==3.8.4  
seaborn==0.13.2  
scikit-learn==1.5.0  

# PyTorch (CPU version)
torch==2.3.0  
torchvision==0.18.0  

# GUI and drag-and-drop
tkinterdnd2==0.3.0  
Pillow==10.3.0  

Note: `tkinter` is part of the Python standard library but must be installed separately (e.g., `sudo apt install python3-tk` on Linux).

---

📊 RESULTS SNAPSHOT

Final Accuracy: 98.57%  
Epochs Trained: 10  
Total Images: 42,000+ across 39 classes

---

🖼️ GUI PREVIEW

(Add a screenshot of your GUI here, if desired)

---

📄 LICENSE

MIT License

---

🙌 ACKNOWLEDGMENTS

- ASL Alphabet Dataset – Kaggle  
- ASL Numbers Dataset – Kaggle  
- TkinterDnD2 by pmgagne
