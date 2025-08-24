
# YouTube Comment Toxicity Analyzer

## Overview
The **YouTube Comment Toxicity Analyzer** is a Python-based application that uses **Machine Learning** to analyze the toxicity of comments on any YouTube video. Users can enter a video URL, and the app will fetch the top comments, classify them as **toxic** or **non-toxic**, and visualize the results.

The app is built using **Streamlit** for an interactive interface and uses a **logistic regression model trained on the Jigsaw toxic comment dataset**.

---

## Features
- Input any YouTube video URL for analysis.
- Fetches up to 100 top-level comments using the **YouTube Data API**.
- Preprocesses comments (removes punctuation, stopwords, and lemmatizes text).
- Classifies comments as **toxic** or **non-toxic** using a trained machine learning model.
- Displays results:
  - **Toxic vs Non-toxic counts**
  - **Top toxic comments**
  - **Toxicity distribution pie chart**
- Handles English comments effectively.

**Note:** The model currently works best for English comments. Comments written in **Hinglish or containing sarcasm** may not be accurately classified.

---

## Technology Stack
- Python 3.8+  
- [Streamlit](https://streamlit.io/)  
- [scikit-learn](https://scikit-learn.org/stable/)  
- [Google API Client](https://developers.google.com/api-client-library/python/)  
- NLTK  
- Matplotlib / Pandas  

---

## Installation

1. Clone the repository:
```bash
git clone <your-repo-link>
cd <your-project-folder>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Obtain a **YouTube Data API key**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)  
   - Enable **YouTube Data API v3**  
   - Create an **API key**  

4. Replace `YOUR_API_KEY` in `app.py` with your API key.

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

1. Enter a **YouTube video URL**.  
2. Click **Analyze**.  
3. View the **results**, including:  
   - Toxic vs Non-toxic counts  
   - Top toxic comments  
   - Toxicity pie chart  

---

## Project Structure

```
YT_Toxicity_Analyzer/
│
├── app.py                # Streamlit app
├── logistic_model.pkl    # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Python dependencies
├── README.md             # Project description
└── utils.py              # (Optional) helper functions
```

---

## How It Works
1. **Input video URL** → extract video ID.  
2. **Fetch comments** from YouTube API.  
3. **Preprocess text** (clean, lowercase, remove stopwords, lemmatize).  
4. **Transform comments** using the TF-IDF vectorizer.  
5. **Predict toxicity** using the saved model.  
6. **Display results** in the app.

---

## Future Improvements
- Support for **Hinglish or multilingual comments** using translation or multilingual models.  
- Detect **sarcasm and context-based toxicity** using **BERT or Transformers**.  
- Allow **batch analysis of multiple videos or channels**.  
- Include **emoji and punctuation features** for better sarcasm detection.  

---

## License
This project is open-source and available under the **MIT License**.
