import streamlit as st
import pandas as pd
from preprocessing import clean_article
from sentiment_analysis import get_mood
from topic_modelling import perform_topic_modeling, get_top_words, assign_themes
import nltk

# Download the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')


def main():
    st.title("News Analysis Project: Uncover What Matters")

    # Upload file
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Data Loaded Successfully")

        # Clean articles
        df['Cleaned_Article'] = df['Article'].apply(clean_article)
        st.write("Original Articles vs Cleaned Articles:")

        # Display original and cleaned articles side by side
        col1, col2 = st.columns(2)
        with col1:
            st.header("Original Articles")
            for i, article in enumerate(df['Article'], start=1):
                st.subheader(f"Article {i}:")
                st.text_area(f"Original Article {i}", article, height=200)

        with col2:
            st.header("Cleaned Articles")
            for i, cleaned_article in enumerate(df['Cleaned_Article'], start=1):
                st.subheader(f"Article {i}:")
                st.text_area(f"Cleaned Article {i}", cleaned_article, height=200)

        # Mood check
        df['Mood'] = df['Cleaned_Article'].apply(get_mood)
        st.write("Mood Ratings:")
        st.write(df[['Article', 'Mood']])

        # Topic modeling (automatically select 10 topics)
        num_topics = 10
        nmf_model, nmf_topics, tfidf_feature_names = perform_topic_modeling(df['Cleaned_Article'], num_topics)
        top_words = get_top_words(nmf_model, tfidf_feature_names)
        df['Theme'] = assign_themes(nmf_topics, top_words)

        # Display results
        st.write("Themes:")
        st.write(df[['Article', 'Theme']])

if __name__ == "__main__":
    main()
