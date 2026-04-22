# 🥭 Agent-Based Multimodal Mango Quality Assessment and Recommendation System

An intelligent AI-powered system for **mango quality assessment, ripeness prediction, grade classification, price recommendation, and buyer decision support** using **computer vision, machine learning, multimodal learning, and agent-based workflow automation**.

---

## 🚀 Live Demo

[Click Here to Use the App](https://mango-quality-ai.streamlit.app)

---

## 🚀 Project Overview

This project uses **front-view image + back-view image + weight data** of Harumanis mangoes to automatically predict:

✅ Ripeness (Ripe / Unripe)  
✅ Quality Grade (Premium / Good / Average)  
✅ Confidence Score  
✅ Recommended Market Price (INR + RM)  
✅ Shelf Life Estimate  
✅ Similar Mango Suggestions  
✅ Buyer Insight Recommendation  

The system is built with an **Agentic AI Pipeline**, where multiple intelligent agents validate inputs, predict quality, and generate recommendations.

---

## 🧠 Key Features

### 🔍 Input Validation Agent

Checks uploaded mango images for:

- Blur detection  
- Brightness validation  
- Duplicate image detection  
- Same mango verification (front + back)

### 🤖 Prediction Agent

Uses trained ML models for:

- Ripeness Classification  
- Grade Prediction  
- Confidence Estimation

### 💰 Recommendation Agent

Provides:

- Price in INR  
- Live RM Conversion  
- Shelf Life Estimate  
- Similar Mangoes  
- Buyer Negotiation Insight

### 🌐 Dual Mode System

#### 1️⃣ Requirement Mode

User enters budget and preference.

#### 2️⃣ Upload Mode

User uploads mango images + weight for AI analysis.

---

## 🏗️ Tech Stack

### Programming

- Python

### Libraries

- Streamlit  
- OpenCV  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- scikit-image  
- joblib

### ML Models Used

- Gradient Boosting Classifier  
- Logistic Regression  
- SVM  
- Random Forest  
- Naive Bayes  
- Fusion Models

---

## 📊 Dataset

Custom Harumanis Mango Dataset containing:

- Front View Images  
- Back View Images  
- Weight Data  
- Ripeness Labels  
- Grade Labels

---

## ⚙️ Multimodal Learning Approach

The system combines:

- Visual Features from front image  
- Visual Features from back image  
- Mass / Weight Information

This improves prediction accuracy over single-input systems.

---

## 📈 Results

Achieved strong classification performance using multimodal fusion:

- High Ripeness Accuracy  
- Strong Grade Classification  
- Reliable Recommendation Output

Confusion matrices and evaluation results included in repository.

---

## 🖥️ User Interface

Built using **Streamlit Web App** with clean interactive workflow.

Features include:

- Image Upload UI  
- Live Predictions  
- Recommendation Dashboard  
- Validation Alerts

---

## 📂 Project Structure

```bash
Agent-Based-Multimodal-Mango-Quality-System/
│── app.py
│── model.py
│── requirements.txt
│── Harumanis_mango_weight_grade.xlsx
│── README.md
│── models/
│── notebooks/
│── images/
│── screenshots/
│── sample_images/
│   ├── normal_images/
│   └── edge_cases/
```

## 📸 Screenshots

### Home Page
![Home](screenshots/home.png)


### Requirement Mode
![Requirement](screenshots/requirement.png)


### Upload Mode
![Upload](screenshots/upload.png)


### Validation Agent
![Validation](screenshots/validation.png)


### Results
![Results](screenshots/results.png)

## 🧪 Sample Test Images

Use the images inside `sample_images/` to quickly test the deployed app.

### 📁 Normal Images
Located in:

sample_images/normal_images/

Includes front and back image pairs:

- sample1_front.jpg + sample1_back.jpg
- sample2_front.jpg + sample2_back.jpg
- sample3_front.jpg + sample3_back.jpg

### ⚠️ Edge Case Images
Located in:

sample_images/edge_cases/

Includes robustness testing samples:

- blur.jpeg
- bright1.jpeg
- nonbright.jpeg
- rotated_sample.jpeg

These help test performance under blur, brightness variation, and rotation.

## ⚙️ Installation & Run Locally

git clone https://github.com/Aditya125dev/Agent-Based-Multimodal-Mango-Quality-System.git
cd Agent-Based-Multimodal-Mango-Quality-System
pip install -r requirements.txt
streamlit run app.py

## 👨‍💻 Author

Aditya Sengar
B.Tech Artificial Intelligence & Machine Learning
Symbiosis Institute of Technology, Pune


## ⭐ Support

If you like this project, consider starring the repository.

