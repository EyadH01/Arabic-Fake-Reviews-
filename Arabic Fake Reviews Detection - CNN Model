
=======================================
A deep learning model for detecting fake reviews in Arabic text using 
Convolutional Neural Networks (CNN) with text preprocessing and feature extraction.
"""

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Conv1D, Activation, Dropout, MaxPooling1D, Flatten, Dense
import keras
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# =========================
# Arabic Text Preprocessing
# =========================
stop_words = set(stopwords.words('arabic'))
stemmer = SnowballStemmer("arabic")

def preprocess(text):
    """
    Preprocess Arabic text by normalizing letters, removing diacritics,
    tokenizing, removing stopwords, and applying stemming.
    
    Args:
        text (str): Raw Arabic text
        
    Returns:
        str: Preprocessed text
    """
    # Normalize Arabic letters
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[ى]", "ي", text)
    text = re.sub(r"[ؤئ]", "ء", text)
    text = re.sub(r"[ة]", "ه", text)
    
    # Remove diacritics
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    
    # Remove repeated characters
    text = re.sub(r"(.)\1+", r"\1", text)

    # Tokenize, remove stopwords, and apply stemming
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    
    return ' '.join(tokens)

# =========================
# Data Loading and Preprocessing
# =========================
def load_and_preprocess_data(file_path):
    """
    Load dataset and preprocess text data.
    
    Args:
        file_path (str): Path to Excel file containing 'text' and 'label' columns
        
    Returns:
        DataFrame: Preprocessed dataframe
    """
    df = pd.read_excel(file_path)
    df['text'] = df['text'].astype(str).apply(preprocess)
    
    # Map labels to binary values: 'fake' = 0, 'real' = 1
    df['label'] = df['label'].map({'fake': 0, 'real': 1})
    
    return df

# =========================
# Model Building
# =========================
def build_cnn_model(input_shape, num_classes=2):
    """
    Build a CNN model for text classification.
    
    Args:
        input_shape (tuple): Shape of input data
        num_classes (int): Number of output classes
        
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential([
        Conv1D(128, 5, padding='same', input_shape=input_shape),
        Activation('relu'),
        Dropout(0.2),
        MaxPooling1D(pool_size=8),
        Flatten(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.RMSprop(1e-5, decay=1e-6),
        metrics=['accuracy']
    )
    
    return model

# =========================
# Main Training Pipeline
# =========================
def main(data_path, test_size=0.2, max_features=5000, batch_size=32, epochs=10):
    """
    Main training pipeline for the fake review detection model.
    
    Args:
        data_path (str): Path to dataset
        test_size (float): Proportion of test data
        max_features (int): Maximum number of features for vectorization
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
    """
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=test_size, random_state=42
    )
    
    # Vectorization
    print("Vectorizing text data...")
    vectorizer = CountVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    
    # Reshape for CNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # One-hot encode labels
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    
    # Build and train model
    print("Building CNN model...")
    model = build_cnn_model(input_shape=(max_features, 1))
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model and vectorizer
    print("\nSaving model...")
    model.save("arabic_fake_review_cnn.h5")
    
    import pickle
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Training completed successfully!")
    
    return model, vectorizer, history

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    # Update this path to your dataset
    DATA_PATH = 'Balanced_Data.xlsx'
    
    model, vectorizer, history = main(
        data_path=DATA_PATH,
        test_size=0.2,
        max_features=5000,
        batch_size=32,
        epochs=10
    )
