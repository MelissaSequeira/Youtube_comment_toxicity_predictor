import streamlit as st
from googleapiclient.discovery import build
import pickle
import re, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pandas as pd

# Load model and vectorizer
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(rf"[{string.punctuation}]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("YouTube Comment Toxicity Analyzer")
video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Analyze"):
    if "v=" not in video_url:
        st.error("Invalid YouTube URL")
    else:
        # Extract video ID
        video_id = video_url.split("v=")[1].split("&")[0]
        
        # Fetch comments
        api_key = "AIzaSyAN0D6XcMCyJOLCHIAM9Yz7OkzX1-H3xMA"
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100, textFormat="plainText"
        )
        response = request.execute()
        comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']]
        
        # Preprocess comments
        processed_comments = [preprocess(c) for c in comments]
        X_new = vectorizer.transform(processed_comments)
        predictions = model.predict(X_new)
        
        # Create dataframe
        df_result = pd.DataFrame({"comment": comments, "toxic": predictions})
        st.write("Toxic vs Non-toxic Counts:")
        st.write(df_result['toxic'].value_counts())
        
        st.write("Comments:")
        st.write(df_result["comment"])

        # Top toxic comments
        st.write("Top Toxic Comments:")
        st.write(df_result[df_result['toxic']==1]['comment'].head(10))
        
        # Pie chart
        counts = df_result['toxic'].value_counts()
        labels = ["Non-toxic" if i==0 else "Toxic" for i in counts.index]
        fig, ax = plt.subplots()
        ax.pie(counts, labels=labels, autopct='%1.1f%%')
        ax.set_title("Toxicity Distribution")
        st.pyplot(fig)

