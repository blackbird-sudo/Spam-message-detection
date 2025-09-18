# Spam-message-detection
This project is a spam detection system that uses ML to classify text messages as spam. It processes text data with TF-IDF vectorization and trains a Support Vector Machine classifier through grid search optimization. The system evaluates model performance, makes predictions on new messages, and can load the trained model for deployment.

import csv

print(f"csv.QUOTE_ALL = {csv.QUOTE_ALL}")

import pandas as pd

import numpy as np

import sklearn.svm as svm

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

import seaborn as sns

# Load data 
def load_data(file_path, target_column=None, text_column=None):

    df = pd.read_csv(file_path, encoding='latin-1', quoting=csv.QUOTE_ALL, 
                    escapechar='\\', on_bad_lines='warn')

    print("Available columns:", df.columns.tolist())
    
    
    if text_column is None:
        text_candidates = []
        for col in df.columns:
            if (df[col].dtype == 'object' and 
                col != target_column and 
                df[col].str.len().mean() > 10):
                text_candidates.append(col)
        
        if text_candidates:
            text_column = text_candidates[0]
            print(f"Selected text column: {text_column}")
        else:
            raise ValueError("No suitable text column found")
    
    
    if target_column is None:
        common_targets = ['label', 'category', 'class', 'sentiment', 'spam', 'target', 'type']
        for col in common_targets:
            if col in df.columns and col != text_column:
                target_column = col
                print(f"Selected target column: {target_column}")
                break
        
        if target_column is None:
            potential_targets = [col for col in df.columns if col != text_column]
            if potential_targets:
                target_column = potential_targets[0]
                print(f"Selected target column: {target_column}")
            else:
                raise ValueError("No suitable target column found")
    
    
    empty_text_count = df[text_column].isna().sum() + (df[text_column] == "").sum()
    print(f"Empty or missing text entries: {empty_text_count}")
    
    
    x_text = df[text_column].fillna('')
    y = df[target_column]
    
    
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        target_names = le.classes_
        print(f"Encoded target labels: {dict(enumerate(target_names))}")
    else:
        target_names = [f"class_{i}" for i in np.unique(y)]
        print(f"Unique target values: {np.unique(y)}")
    
    return x_text, y, target_names, df, text_column, target_column

def train_spam_detection(x_text, y, target_names):

    x_train, x_test, y_train, y_test = train_test_split(x_text, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ('svm', svm.SVC(
            kernel='linear',
            random_state=42,
            class_weight='balanced',
            probability=True,
        ))
    ])
    
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf'],
        'tfidf__max_features': [3000, 5000, 7000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        verbose=2,
        scoring='f1'  
    )
    
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_model, x_test, y_test, grid_search

def evaluate_model(model, x_test, y_test, target_names):

    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return accuracy, cm, y_pred, y_pred_prob

def predict_new_message(model, messages, target_names):

    predictions = model.predict(messages)
    probabilities = model.predict_proba(messages)
    
    for i, (msg, pred, prob) in enumerate(zip(messages, predictions, probabilities)):
        print(f"\nMessage {i+1}:")
        print(f"Text: {msg[:100]}..." if len(msg) > 100 else f"Text: {msg}")
        print(f"Prediction: {target_names[pred]}")
        print(f"Probability: {max(prob):.4f}")
        print(f"Confidence: {'Spam' if target_names[pred]=='spam' else 'Ham'} (Score: {max(prob):.4f})")
    
    return predictions, probabilities

def save_model(model, file_name='spam_detection.pkl'):

    import joblib
    joblib.dump(model, file_name)
    print(f"Model saved as: {file_name}")

def load_saved_model(file_name='spam_detection.pkl'):

    import joblib
    return joblib.load(file_name)

if __name__ == "__main__":

    try:
        file_path = "C:/Users/user/Downloads/spam.csv"
        x_text, y, target_names, df, text_column, target_column = load_data(file_path)
        
        best_model, x_test, y_test, grid_search = train_spam_detection(x_text, y, target_names)
        accuracy, cm, y_pred, y_pred_prob = evaluate_model(best_model, x_test, y_test, target_names)
        
        test_messages = [
            "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
            "Hey, are we still meeting for lunch tomorrow at 12?",
            "URGENT: Your bank account needs verification. Please confirm your details immediately.",
            "Meeting reminder: Project review at 3 PM in conference room B."
        ]
        
        predictions, probabilities = predict_new_message(best_model, test_messages, target_names)
        save_model(best_model)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your file path and data format")

      
