import streamlit as st
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to convert TextBlob sentiment to dataframe
def convert_to_df(sentiment):
    sentiment_dict = {'Polarity': sentiment.polarity, 'Subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['Metric', 'Value'])
    return sentiment_df

# Function to analyze token sentiment using VADER
def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list, neg_list, neu_list = [], [], []
    for word in docx.split():
        score = analyzer.polarity_scores(word)['compound']
        if score > 0.1:
            pos_list.append((word, score))
        elif score < -0.1:
            neg_list.append((word, score))
        else:
            neu_list.append(word)
    result = {'Positives': pos_list, 'Negatives': neg_list, 'Neutrals': neu_list}
    return result

def main():
    st.set_page_config(page_title="Sentiment Analysis App", layout="wide", page_icon=":bar_chart:")

    st.markdown("""
        <style>
        .reportview-container {
            background: #f0f2f6;
            padding: 2rem;
        }
        .sidebar .sidebar-content {
            background: #f9f9f9;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            text-align: center;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Sentiment Analysis NLP App")
    st.sidebar.subheader("Navigation")
    menu = ["📄 About", "🏠 Home"]
    choice = st.sidebar.selectbox("Menu", menu, index=0)

    if choice == "🏠 Home":
        st.subheader("Home - Sentiment Analysis")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here", height=200)
            submit_button = st.form_submit_button(label='Analyze')

        if submit_button:
            col1, col2 = st.columns(2)

            with col1:
                st.info("TextBlob Sentiment Analysis")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                if sentiment.polarity > 0:
                    st.markdown("### Sentiment: Positive :smiley:")
                elif sentiment.polarity < 0:
                    st.markdown("### Sentiment: Negative :angry:")
                else:
                    st.markdown("### Sentiment: Neutral 😐")

                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                chart = alt.Chart(result_df).mark_bar().encode(
                    x='Metric',
                    y='Value',
                    color=alt.Color('Metric', scale=alt.Scale(range=['#1f77b4', '#ff7f0e']))
                ).properties(title="Sentiment Analysis Visualization")
                st.altair_chart(chart, use_container_width=True)

            with col2:
                st.info("VADER Token Sentiment Analysis")
                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)

                # Token sentiment visualizations
                pos_df = pd.DataFrame(token_sentiments['Positives'], columns=['Token', 'Score'])
                neg_df = pd.DataFrame(token_sentiments['Negatives'], columns=['Token', 'Score'])

                pos_chart = alt.Chart(pos_df).mark_bar().encode(
                    x='Token',
                    y='Score',
                    color=alt.value('green')
                ).properties(title="Positive Token Sentiments")

                neg_chart = alt.Chart(neg_df).mark_bar().encode(
                    x='Token',
                    y='Score',
                    color=alt.value('red')
                ).properties(title="Negative Token Sentiments")

                st.altair_chart(pos_chart, use_container_width=True)
                st.altair_chart(neg_chart, use_container_width=True)

    else:
        st.subheader("About")
        st.image("C:\\Users\\HAI\\Desktop\\sentiment-analysis\\sentiment_analysis_app\\image.png", use_column_width=True)
        st.write("""
            ## About the Sentiment Analysis App

            This Sentiment Analysis NLP App allows you to analyze the sentiment of any text using two powerful libraries: 
            **TextBlob** and **VADER**.

            ### Features:
            - **TextBlob Sentiment Analysis**: Provides overall sentiment polarity and subjectivity.
            - **VADER Token Sentiment Analysis**: Analyzes individual token sentiments for more granular insights.

            ### Technologies Used:
            - **Streamlit**: For building the web application.
            - **TextBlob**: For natural language processing and sentiment analysis.
            - **VADER**: For sentiment analysis on social media text.
            - **Altair**: For creating interactive visualizations.

            ### How to Use:
            1. Navigate to the Home page.
            2. Enter the text you want to analyze in the text area.
            3. Click the 'Analyze' button to see the sentiment analysis results.

            This application is designed to help you understand the sentiment of your text data easily and effectively.
        """)
        st.markdown("""
            <style>
            .about-text {
                font-size: 18px;
                line-height: 1.6;
            }
            </style>
            """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
