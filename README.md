# AI-PREDICTION-ON-SHOPLIFTING
# 🛒 Shoplifting Detection System

A real-time shoplifting detection system built using machine learning and computer vision. The project includes training scripts, a detection GUI, and an automated Windows launcher for easy execution.

---

## 🚀 Features
- Real-time detection using a trained `model.h5`
- Simple GUI for monitoring behavior
- Auto-launch script (`run.bat`) using Python 3.10
- Automatic model presence check
- Easy model retraining

---

## 📂 Project Structure
├── detect.py # Main detection/GUI script
├── dataset.py # Model training script
├── model.h5 # Trained model (required)
├── run.bat # Windows launcher

yaml
Copy code

---

## ▶️ How to Run

### **Using the Launcher (Windows)**
Double-click:
run.bat

markdown
Copy code
The script will:
1. Check if `model.h5` exists  
2. If missing → prompt you to train the model  
3. If present → launch the detection GUI

### **Manual Run**
```bash
py -3.10 detect.py
🧠 Training the Model
If model.h5 is not available, train it with:

bash
Copy code
py -3.10 dataset.py
📋 Requirements
Python 3.10

TensorFlow / OpenCV

Other dependencies as required by your scripts

(I can generate a requirements.txt—just ask!)

📝 About run.bat
The launcher script:

Sets the working directory

Verifies model.h5

Provides clear errors and training instructions

Starts the detection interface automatically

📄 License
Add your preferred license here (MIT, GPL, etc.).

🤝 Contributions
Pull requests and improvements are welcome!

yaml
Copy code

---

If you'd like, I can also generate:

✅ `requirements.txt`  
✅ A better banner for the README  
✅ A version with images  
Just tell me!
