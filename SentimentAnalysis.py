import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import re
import string
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

#Initial view of dataset
df = pd.read_csv(r'D:\Python\SentimentAnalysis\sentiment_data.csv')
print(df.head(5))

# Get summary of the DataFrame, including all columns
print(df.describe(include='all'))

# Drop 1st column as it is not required
df = df.drop('Unnamed: 0',axis = 1)

#Feature Engineering

def lemmatize_text(text):
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())

def remove_missing_values(df):
    return df.dropna()

def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_emojis(text):
    emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_emojis(text):
    emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_special_chars(text):
    allowed_chars = set(string.ascii_letters + "áéíóúãõàâêôç ")
    return ''.join(c for c in text if c in allowed_chars)

def clean_text(text):
    if not isinstance(text, str):
        return '' 
    
    text = text.lower().strip() 
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_emojis(text)
    text = remove_special_chars(text)
    text = re.sub(r'\s+', ' ', text) 
    text = lemmatize_text(text)
    return text
df['clean_Comment'] = df['Comment'].apply(clean_text)

X = df['clean_Comment'] 
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

all_text = ' '.join(df['clean_Comment'])
wordcloud = WordCloud(width=800, height=400, 
                      background_color='white',
                      colormap='YlOrRd').generate(all_text)

# Plota
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Using logistic regression

text_cleaner = FunctionTransformer(lambda x: x.apply(clean_text))  # transforma a Series

pipeline_LR = Pipeline([
    ('cleaner', text_cleaner),
    ('tfidf', TfidfVectorizer(
    stop_words='english',
    max_features=20000,           # Try larger vocab size
    ngram_range=(1, 2),           # Include unigrams + bigrams
    min_df=2,                     # Remove very rare terms
    max_df=0.95                   # Remove overly common terms
)
),
    ('lr', LogisticRegression(max_iter=1000))
])
# Accuracy of 76%

pipeline_RF = Pipeline([
    ('cleaner', text_cleaner),
    ('tfidf', TfidfVectorizer(
    stop_words='english',
    max_features=20000,           # Try larger vocab size
    ngram_range=(1, 2),           # Include unigrams + bigrams
    min_df=2,                     # Remove very rare terms
    max_df=0.95                   # Remove overly common terms
)
),
   ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Accuracy of 80%

pipeline_NB = Pipeline([
    ('cleaner', text_cleaner),
    ('tfidf', TfidfVectorizer(
    stop_words='english',
    max_features=20000,           # Try larger vocab size
    ngram_range=(1, 2),           # Include unigrams + bigrams
    min_df=2,                     # Remove very rare terms
    max_df=0.95                   # Remove overly common terms
)
),
   ('nb', MultinomialNB())
])

# Accuracy of 60%


pipeline_LR.fit(X_train, y_train)
pipeline_RF.fit(X_train, y_train)
pipeline_NB.fit(X_train, y_train)

#Validation for Logistic Regression
y_pred = pipeline_LR.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Validation for Random Forest
y_pred = pipeline_RF.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Validation for MultinomialNB
y_pred = pipeline_NB.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))





