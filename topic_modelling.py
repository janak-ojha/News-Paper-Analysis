from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def perform_topic_modeling(cleaned_articles, num_topics=5):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(cleaned_articles)
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_topics = nmf_model.fit_transform(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    return nmf_model, nmf_topics, tfidf_feature_names

def get_top_words(model, feature_names, n_top_words=10):
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return top_words

def assign_themes(nmf_topics, top_words):
    topic_labels = {topic: " ".join(words[:3]) for topic, words in top_words.items()}
    article_themes = []
    for topic_distribution in nmf_topics:
        dominant_topic = topic_distribution.argmax()
        theme = topic_labels[dominant_topic]
        article_themes.append(theme)
    return article_themes
