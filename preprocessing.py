import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

def clean_article(article):
    # Tokenize the article into words
    tokens = word_tokenize(article)
    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    # Remove non-alphanumeric characters
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens if token.isalnum()]
    # Get the set of English stopwords
    stop_words = set(stopwords.words('english'))
    # Remove stopwords from the tokens
    cleaned_tokens = [token for token in tokens if token not in stop_words]
    # Return the cleaned article as a single string
    return ' '.join(cleaned_tokens)
