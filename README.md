# Arabic Fake Reviews Detection using CNN

**Author:** Eyad Harb  
**Language:** Python  

---

## Overview
A deep learning project for detecting fake reviews in Arabic text using **Convolutional Neural Networks (CNN)**. The system includes:

- Arabic text preprocessing: normalization, tokenization, stopword removal, stemming.
- Feature extraction using Bag-of-Words.
- Training a CNN for binary classification (`fake` or `real`).
- Saving trained model and vectorizer for later inference.

---

## Dataset
- The dataset should be an Excel file (`.xlsx`) with **`text`** and **`label`** columns.
- Labels must be either `fake` or `real`.
- Example dataset: [Arabic Fake Reviews Detection Dataset on Kaggle](https://www.kaggle.com/datasets/shathaalturke/afrd-arabic-fake-reviews-detection)

---

## Requirements
- Python 3.8+
- Pandas
- NLTK
- Scikit-learn
- Keras + TensorFlow
- Openpyxl (for reading Excel files)

Install dependencies:
```bash
pip install pandas nltk scikit-learn keras tensorflow openpyxl
