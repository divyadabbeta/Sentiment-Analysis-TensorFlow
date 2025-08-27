# 🎬 Sentiment Analysis on IMDb Movie Reviews  

## 📌 Project Overview  
This project implements a **sentiment analysis model** using the **IMDb dataset of 50,000 movie reviews**.  
The goal is to classify each review as either **positive** or **negative**.  
The project leverages **deep learning (LSTM with TensorFlow)** to understand the underlying sentiment of text data.  

---

## ⚙️ Requirements  
Ensure the following Python libraries are installed before running the project:  
- TensorFlow  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-Learn  

Install them using:  

pip install tensorflow numpy pandas matplotlib scikit-learn

## 📂 Dataset

The dataset used is the IMDb Movie Reviews Dataset from Kaggle.

Size: 50,000 reviews

Columns:

review → the text of the review

sentiment → label (positive or negative)

## 🧹 Data Preprocessing

Cleaning the text

Removed HTML tags and special characters

Converted text to lowercase

Tokenization & Padding

Converted words into numeric sequences

Applied padding to ensure equal sequence length (200 words)

Label Encoding

Converted positive → 1 and negative → 0

Train-Test Split

80% training data, 20% testing data

## 🧠 Model Architecture

The model is built using TensorFlow Keras Sequential API:

Embedding Layer → Transforms word indices into dense vectors (16 dimensions)

LSTM Layer (64 units) → Captures sequential patterns in reviews

LSTM Layer (32 units) → Extracts deeper sequence dependencies

Dense Layer (24 units, ReLU activation) → Fully connected layer for feature learning

Output Layer (Sigmoid activation) → Predicts probability of review being positive

Loss Function: Binary Cross-Entropy
Optimizer: Adam
Metric: Accuracy

## 📊 Training & Evaluation

The model was trained for 10 epochs with a validation split of 20%.

Performance visualization was done using Matplotlib to track training vs. validation accuracy.

Final evaluation on the test set provided a reliable measure of generalization.

## ✅ Results

The model achieved high accuracy in classifying sentiments.

Predictions on new reviews showed strong alignment with expected outcomes.

Example:

"I absolutely loved this movie! The plot was thrilling and the characters were so well developed." → Positive

"The film was a disaster. Poor acting and a predictable storyline." → Negative

## 🚀 Applications

Analyzing customer feedback and product reviews

Enhancing recommendation systems

Social media sentiment monitoring

## 📌 Future Improvements

Use pre-trained word embeddings like GloVe or Word2Vec

Apply regularization techniques to prevent overfitting

Deploy the model as a web application for real-time sentiment prediction
