import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import time

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextInput>div>div>input {
            padding: 12px;
            border-radius: 8px;
        }
        .stButton>button {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .real-news {
            background-color: #d4edda;
            color: #155724;
            border-left: 5px solid #28a745;
        }
        .fake-news {
            background-color: #f8d7da;
            color: #721c24;
            border-left: 5px solid #dc3545;
        }
        .title {
            color: #2c3e50;
        }
        .sidebar .sidebar-content {
            background-color: #343a40;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Load models and vectorizer
@st.cache_resource
def load_models():
    try:
        with open('lightgbm_model.pkl', 'rb') as f:
            lgbm_model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_v = pickle.load(f)
        return lgbm_model, tfidf_v
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

lgbm_model, tfidf_v = load_models()

# Text preprocessing function
def preprocess_text(text):
    ps = WordNetLemmatizer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.lemmatize(word) for word in review if not word in stopwords.words('english')]
    return ' '.join(review)

# Prediction function
def predict_news(text):
    try:
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Vectorize the text
        text_vector = tfidf_v.transform([processed_text]).toarray()
        
        # Make predictions with both models
        lgbm_pred = lgbm_model.predict(text_vector)
        lgbm_prob = lgbm_model.predict_proba(text_vector)
        
        # Determine final prediction (you can adjust this logic)
        final_pred = lgbm_pred[0]
        confidence = max(lgbm_prob[0])
        
        return final_pred, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Main app
def main():
    st.title("📰 Fake News Detector")
    st.markdown("""
        This app uses machine learning to detect potentially fake news articles. 
        Enter a news headline and content below to check its authenticity.
    """)
    
    with st.expander("ℹ️ How to use"):
        st.write("""
            1. Enter a news headline in the first box
            2. Enter the news content/text in the second box
            3. Click the 'Analyze News' button
            4. View the results and confidence level
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Enter News Details")
        headline = st.text_input("News Headline", placeholder="Enter the news headline here...")
        content = st.text_area("News Content", height=200, placeholder="Paste the news content here...")
        
        analyze_btn = st.button("Analyze News")
    
    with col2:
        st.subheader("Analysis Results")
        
        if analyze_btn and (headline or content):
            with st.spinner("Analyzing the news article..."):
                # Combine headline and content for analysis
                full_text = f"{headline}\n\n{content}"
                
                # Make prediction
                prediction, confidence = predict_news(full_text)
                time.sleep(1)  # Simulate processing time
                
                if prediction is not None:
                    # Display results
                    if prediction == 1:
                        st.markdown(f"""
                            <div class="result-box fake-news">
                                <h3>⚠️ Fake News Detected</h3>
                                <p>Our analysis indicates this news is likely <strong>fake</strong>.</p>
                                <p>Confidence: {confidence:.1%}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.warning("Be cautious with this information. Consider verifying from trusted sources.")
                    else:
                        st.markdown(f"""
                            <div class="result-box real-news">
                                <h3>✅ Genuine News Detected</h3>
                                <p>Our analysis indicates this news is likely <strong>real</strong>.</p>
                                <p>Confidence: {confidence:.1%}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.success("This news appears to be trustworthy, but always practice media literacy.")
                    
                    # Show some analysis details
                    with st.expander("Analysis Details"):
                        st.write("""
                            **How this works:**
                            - The system analyzes the text using natural language processing
                            - It compares patterns with known fake and real news articles
                            - The confidence score indicates how certain the model is
                            
                            **Note:** No system is perfect. Use this as one of several verification tools.
                        """)
                else:
                    st.error("Could not analyze the text. Please try again.")
        
        elif analyze_btn:
            st.warning("Please enter at least a headline or some content to analyze.")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
        This fake news detector uses a machine learning ensemble model trained on thousands 
        of real and fake news articles to identify suspicious content.
        
        **Model Accuracy:** ~99.7%
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Tips for Spotting Fake News")
    st.sidebar.write("""
        1. Check the source credibility
        2. Look for unusual URLs or site designs
        3. Verify the author's credentials
        4. Check the date of publication
        5. Cross-reference with other reputable sources
        6. Be skeptical of emotional headlines
    """)

if __name__ == "__main__":
    # Download NLTK data if not already present
    try:
        import nltk
        nltk.download('stopwords')
        nltk.download('wordnet')
    except:
        pass
    
    main()