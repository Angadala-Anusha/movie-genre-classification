import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
# Assuming your dataset has columns: 'plot_summary' and 'genre'
try:
    df = pd.read_csv('task1.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The dataset 'your_dataset.csv' was not found. Please check the file path.")
    exit()

# Basic data inspection
print(df.head())

# Ensure that there are no missing values in 'plot_summary' or 'genre' columns
if df['plot_summary'].isnull().sum() > 0 or df['genre'].isnull().sum() > 0:
    print("Warning: Missing values detected in the dataset.")
    df = df.dropna(subset=['plot_summary', 'genre'])

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Apply preprocessing to the 'plot_summary' column
df['processed_plot'] = df['plot_summary'].apply(preprocess_text)

# Splitting data into training and testing sets
X = df['processed_plot']  # Features: Plot summaries
y = df['genre']  # Target variable: Genre

# Convert the target labels (genre) to categorical if they are not already
y = pd.factorize(y)[0]  # Convert genres to numerical labels

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train models
# You can choose between Naive Bayes, Logistic Regression, or Support Vector Machine

# Naive Bayes
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_tfidf, y_train)

# Logistic Regression
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_tfidf, y_train)

# Support Vector Machine
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Predictions on the test set
y_pred_nb = naive_bayes_model.predict(X_test_tfidf)
y_pred_lr = logistic_regression_model.predict(X_test_tfidf)
y_pred_svm = svm_model.predict(X_test_tfidf)

# Evaluation: Accuracy and Classification Report
print("\nNaive Bayes Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

print("\nLogistic Regression Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("\nSVM Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Example: Predicting genre for a new plot summary
new_plot = ["A young girl discovers she has magical powers and must save her kingdom from an evil sorcerer."]
new_plot_preprocessed = [preprocess_text(plot) for plot in new_plot]
new_plot_tfidf = tfidf_vectorizer.transform(new_plot_preprocessed)

# Predict genre using the trained models
print("\nPredicted Genre using Naive Bayes:", naive_bayes_model.predict(new_plot_tfidf))
print("Predicted Genre using Logistic Regression:", logistic_regression_model.predict(new_plot_tfidf))
print("Predicted Genre using SVM:", svm_model.predict(new_plot_tfidf))
