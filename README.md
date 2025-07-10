# SentimentAnalysis

# 📊 Sentiment Analysis on Text Comments using Machine Learning

This project demonstrates how to perform sentiment analysis on user comments using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. It includes data cleaning, visualization, model training, and evaluation using popular Python libraries.

## 📁 Dataset

- **File:** `sentiment_data.csv`
- **Columns:**
  - `Unnamed: 0` – Index column (dropped during preprocessing)
  - `Comment` – Raw text comment
  - `Sentiment` – Target label (e.g., Positive, Negative)


## 🛠️ Libraries Used

- `pandas`, `numpy` – Data manipulation
- `matplotlib`, `seaborn`, `wordcloud` – Visualization
- `nltk` – Text preprocessing (lemmatization)
- `sklearn` – Machine learning pipeline and evaluation
- `re`, `string` – Custom text cleaning utilities

## 🔍 Data Preprocessing

### Cleaning Steps:
- Convert text to lowercase
- Remove URLs, mentions, emojis, and special characters
- Lemmatize words
- Strip whitespaces and normalize text

### Key Functions:
    # Combines several preprocessing steps:
     - Lowercasing
     - URL, mention, emoji, special char removal
     - Lemmatization

| Model                    | Accuracy (Approx.) |
| ------------------------ | ------------------ |
| Logistic Regression      | 76%                |
| Random Forest Classifier | 80%                |
| Multinomial Naive Bayes  | 60%                |


### Common Pipeline Components:
FunctionTransformer to apply clean_text
TfidfVectorizer for text feature extraction
ngram_range=(1, 2) → Unigrams and bigrams
min_df=2, max_df=0.95 → Filter rare and overly common terms


## How to Run
Clone the repository or copy the code into a .py file

Place the sentiment_data.csv file in the same directory

Install dependencies:

bash
Copy
Edit
_pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud_
Download NLTK resources:

python
Copy
Edit
import nltk
nltk.download('wordnet')
Run the script:

bash
Copy
Edit
python sentiment_analysis.py
## 💡 Future Improvements
Incorporate deep learning models (e.g., LSTM, BERT)

Add support for multi-language sentiment classification

Build an interactive web app using Streamlit or Flask

## 📄 License
This project is for educational purposes only.
