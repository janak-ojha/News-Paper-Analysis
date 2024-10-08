import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.tokenize import TreebankWordTokenizer

# import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
# def clean_article(article):
#     tokenizer = TreebankWordTokenizer()
#     tokens = tokenizer.tokenize(article)
#     tokens = [token.lower() for token in tokens]
#     tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens if token.isalnum()]
#     stop_words = set(stopwords.words('english'))
#     cleaned_tokens = [token for token in tokens if token not in stop_words]
#     return ' '.join(cleaned_tokens)


def clean_article(article):
    tokens = word_tokenize(article)
    tokens = [token.lower() for token in tokens]
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens if token.isalnum()]
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)

