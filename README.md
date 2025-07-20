# 🧤 Sign Language Detection using Deep Learning

> A real-time sign language recognition system built using computer vision and deep learning, enabling seamless human-computer interaction for the hearing and speech impaired.

---

## 📽️ Demo
![WhatsApp Image 2025-07-20 at 00 43 12_0e08fd39](https://github.com/user-attachments/assets/7c438432-fd6b-44a5-b9ad-91454b0ed7ca)

---

## 🧠 Project Overview

This project leverages **Convolutional Neural Networks (CNNs)** and **real-time video processing** to detect and classify sign language gestures from live camera input. Trained on a curated dataset of hand signs, the model translates visual patterns into corresponding alphabetic or semantic labels.

> Designed with accessibility, speed, and accuracy in mind — enabling inclusive communication powered by AI.

---

## 🚀 Tech Domains

- **AI & ML:** PyTorch, TensorFlow, Transformers, Hugging Face, LangChain, LlamaIndex  
- **Languages:** Python, TypeScript, Java, C++, SQL  
- **Data Science:** Pandas, NumPy, Scikit-learn, XGBoost, Power BI  
- **Cloud & DevOps:** AWS, Docker, Kubernetes, MLflow, Airflow, GitHub Actions  

---

## ⚙️ Tech Stack

| Domain               | Tools / Frameworks                                       |
|----------------------|----------------------------------------------------------|
| Deep Learning        | PyTorch, TensorFlow                                      |
| CV & Preprocessing   | OpenCV, MediaPipe, PIL                                   |
| Data Handling        | Pandas, NumPy                                            |
| Visualization        | Matplotlib, Seaborn, Streamlit (for optional UI)         |
| Deployment (Optional)| Flask, Docker, AWS S3 / EC2                              |

---

## 🏗️ Model Architecture

- **Backbone:** Custom CNN / ResNet-18-based architecture
- **Input:** 64x64 grayscale or RGB hand-sign images
- **Output:** Multiclass prediction (26 alphabetic ASL signs or custom gesture set)
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam / SGD

---

## 📁 Dataset

- **Dataset Used:** [American Sign Language (ASL) Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Size:** ~87,000 labeled hand gesture images
- **Preprocessing:** Rescaling, normalization, data augmentation (flip, zoom, shift)

---

## 🧪 Setup & Installation

```bash
# Clone the repo
git clone https://github.com/vijaykanagaraj/sign-language-detection.git
cd sign-language-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Launch training or app
python train.py        # to train the model
python detect.py       # to run detection

python detect.py --camera 0 --model ./models/sign_cnn.pth

Make sure your webcam is connected.

Use clearly visible hand gestures.

Press Q to quit the camera window.

📈 Results
Training Accuracy: ~98.7%

Validation Accuracy: ~96.2%

Inference Speed: ~25 FPS (CPU), ~60 FPS (GPU)

Misclassification Rate: < 4% with clean input

🤖 Future Enhancements
🔤 Extend support to dynamic signs (e.g., motion-based gestures)

🌐 Add multilingual gesture mapping

📱 Deploy as a mobile app using TensorFlow Lite

🧩 Integrate with speech synthesis modules

🙌 Contribution
Contributions, issues, and feature requests are welcome!
