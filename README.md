# 🧠 Sentiment Analysis using Machine Learning and BERT

## 📌 Overview

This project focuses on building a sentiment analysis system that classifies text into **positive** and **negative** sentiments.
It compares traditional machine learning models with a transformer-based model (**BERT**) and provides an interactive **Streamlit web application** for real-time predictions.

---

## 🎯 Problem Statement

With the growing volume of online text (tweets, reviews, comments), automated sentiment analysis is essential for:

* Customer feedback analysis
* Brand monitoring
* Social media insights

This project evaluates multiple models to identify the most effective approach.

---

## 📊 Dataset

* **Sentiment140 Dataset**
* Source: https://www.kaggle.com/datasets/kazanova/sentiment140

### Details:

* 1.6 million tweets
* Labels:

  * `0` → Negative
  * `4` → Positive

---

## ⚙️ Approach

### 1. Data Preprocessing

* Removed URLs, mentions, special characters
* Lowercasing
* Tokenization
* Stopword removal (for ML models)

---

### 2. Models Used

#### 🔹 Machine Learning

* Logistic Regression
* Support Vector Machine (SVM)
* TF-IDF features

#### 🔹 Deep Learning

* BERT (Transformer-based model)
* Fine-tuned for sentiment classification

---

## 📈 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~82%     |
| SVM                 | ~84%     |
| BERT                | ~90–92%  |

### 🔍 Insights:

* BERT captures contextual meaning better
* ML models are faster but less accurate
* Transformers perform well on real-world text

---

## 🌐 Streamlit Web Application

An interactive UI is built using Streamlit to test models in real time.

### ✨ Features:

* Input custom text and get sentiment prediction
* Select different models:

  * Logistic Regression
  * SVM
  * BERT
* Compare predictions across models
* Instant output (Positive / Negative)

---

## 🧪 How to Use the App

### Run the Streamlit app:

```bash
streamlit run app.py
```

### In the UI:

1. Enter any sentence (e.g., product review or tweet)
2. Choose a model from the dropdown
3. Click **Predict**
4. View the predicted sentiment

---

## 🔁 Model Switching

The application allows dynamic model selection, enabling users to:

* Compare how different models interpret the same text
* Observe differences in prediction accuracy
* Understand strengths of each approach

Example:

* Input: *"The movie was surprisingly good"*

  * Logistic Regression → Positive
  * SVM → Positive
  * BERT → Positive (higher confidence)

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* HuggingFace Transformers
* TensorFlow / PyTorch
* Streamlit
* Pandas, NumPy

---

## ▶️ How to Run the Project

### 1. Clone repository

```bash
git clone https://github.com/yourusername/sentiment-analysis-bert.git
cd sentiment-analysis-bert
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run model (optional)

```bash
python main.py
```

### 4. Run web app

```bash
streamlit run app.py
```

---

## 📸 Sample Output

```
Input: "I absolutely love this product!"
Prediction: Positive 😊

Input: "Worst service ever."
Prediction: Negative 😡
```

---

## 🚀 Key Highlights

* End-to-end NLP pipeline
* Comparison of ML vs Transformer models
* Interactive web application
* Real-time sentiment prediction

---

## 🔮 Future Improvements

* Add neutral sentiment class
* Deploy on cloud (Streamlit Cloud / AWS)
* Improve inference speed
* Add confidence scores visualization

---

## 👨‍💻 Author

Your Name
Master’s in Artificial Intelligence
National College of Ireland

---
