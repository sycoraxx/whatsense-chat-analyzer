# WhatSense - Approximating Sentiments and Analyzing Chats 
1. Displays visualizations showing various texting trends and periodicity.
2. Analyzes the words used by their usage frequency.
3. Approximates Sentiments using a Sentiment Analysis Model.

### Sentiment Analysis Model

- Trained on a 1.6 Million Tweets [dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).
- Training data: 90%, Validation data: 10%
- Data Cleaning involving several steps: Lemmatization, Emoji Textification, Stopwords Removal, etc.
- Feature Extraction using Tf-Idf Vectorizer with 50,000 features (unigram and bi-gram)
- Model - Logistic Regression
- Accuracy: 81% on Test Data.

**deployed link [here](https://sycoraxx-whatsense-chat-analyzer-app-xljxvn.streamlit.app/)**
