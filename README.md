# 🧠 Sentiment Analysis using Machine Learning and BERT

## 📌 Overview

This project focuses on building a robust sentiment analysis system that classifies text into **positive** and **negative** sentiments.
It compares traditional machine learning models with a deep learning-based transformer model (**BERT**) to evaluate performance differences.

The goal is to demonstrate how modern NLP techniques outperform classical approaches in understanding contextual meaning in text data.

---

## 🎯 Problem Statement

With the increasing volume of user-generated content (reviews, tweets, comments), it becomes essential to automatically analyze sentiment for:

* Business insights
* Customer feedback analysis
* Social media monitoring

This project aims to develop and compare multiple models for accurate sentiment classification.

---

## 📊 Dataset

The dataset used in this project is:

* **Sentiment140 Dataset**
* Source: https://www.kaggle.com/datasets/kazanova/sentiment140

### Dataset Details:

* 1.6 million tweets
* Labels:

  * `0` → Negative
  * `4` → Positive
* No neutral class

---

## ⚙️ Approach

### 1. Data Preprocessing

* Removed URLs, mentions, and special characters
* Lowercasing text
* Tokenization
* Stopword removal (for ML models)

---

### 2. Models Used

#### 🔹 Traditional Machine Learning Models

* Logistic Regression
* Support Vector Machine (SVM)

Features:

* TF-IDF Vectorization

---

#### 🔹 Deep Learning Model

* BERT (Bidirectional Encoder Representations from Transformers)

Features:

* Context-aware embeddings
* Fine-tuned on the dataset

---

## 📈 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~82%     |
| SVM                 | ~84%     |
| BERT                | ~90–92%  |

### 🔍 Observations:

* BERT significantly outperforms traditional models
* Classical models struggle with sarcasm and context
* Transformer models capture semantic meaning more effectively

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* TensorFlow / PyTorch
* HuggingFace Transformers
* Pandas, NumPy, Matplotlib

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/sentiment-analysis-bert.git
cd sentiment-analysis-bert
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

---

## 📸 Sample Output

Example:

```
Input: "I absolutely love this product!"
Output: Positive 😊

Input: "This is the worst experience ever."
Output: Negative 😡
```

---

## 🚀 Key Highlights

* Comparison between ML and Deep Learning models
* Real-world dataset with large-scale training
* Demonstrates effectiveness of transformer models
* Practical NLP pipeline implementation

---

## 🔮 Future Improvements

* Add neutral sentiment class
* Deploy as a web application (Streamlit/Flask)
* Real-time Twitter sentiment analysis
* Model optimization for faster inference

---

## 👨‍💻 Author

Your Name
Master’s in Artificial Intelligence
National College of Ireland

---
