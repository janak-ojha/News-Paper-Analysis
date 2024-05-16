from textblob import TextBlob

def get_mood(article):
    analysis = TextBlob(article)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'
