import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import base64

# Set page title and description
st.set_page_config(page_title="Fake Review Detection System", layout="wide")
st.title("Fake Review Detection System")
st.write("Upload a CSV file with reviews to classify them as real or fake")

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        # Remove digits
        text = re.sub('\d+', '', text)
        # Remove extra spaces
        text = re.sub('\s+', ' ', text).strip()
        return text
    return ""

# Function to create a download link for dataframes
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data.head())
        
        # Check if required columns exist
        required_cols = ['text']  # Minimum required column
        
        if all(col in data.columns for col in required_cols):
            # Preprocess the text
            st.write("Preprocessing text...")
            data['processed_text'] = data['text'].apply(preprocess_text)
            
            # If there's a label column, we can train and evaluate the model
            if 'label' in data.columns:
                st.write("Training model with provided labels...")
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    data['processed_text'], 
                    data['label'], 
                    test_size=0.2, 
                    random_state=42
                )
                
                # Feature extraction
                vectorizer = TfidfVectorizer(max_features=5000)
                X_train_tfidf = vectorizer.fit_transform(X_train)
                X_test_tfidf = vectorizer.transform(X_test)
                
                # Train the model
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train_tfidf, y_train)
                
                # Evaluate the model
                predictions = model.predict(X_test_tfidf)
                accuracy = accuracy_score(y_test, predictions)
                report = classification_report(y_test, predictions)
                
                st.write(f"Model Accuracy: {accuracy:.2f}")
                st.text("Classification Report:")
                st.text(report)
                
                # Process all reviews
                st.write("Classifying all reviews...")
                X_all_tfidf = vectorizer.transform(data['processed_text'])
                data['prediction'] = model.predict(X_all_tfidf)
                
            else:
                st.write("No label column found. Using pre-trained model...")
                
                # Create a simple dataset for training
                sample_data = {
                    'text': [
                        "This product is amazing and exceeded my expectations!",
                        "I love this product, it works perfectly!",
                        "Great value for money, highly recommend!",
                        "Best purchase I've ever made, truly life-changing.",
                        "Absolutely fantastic product, 5 stars!",
                        "Amazing product! Best ever! Life changing! Must buy now!!!!",
                        "Incredible!!! Perfect!!! Awesome!!! Buy immediately!!!",
                        "This is the greatest product ever made in history!!!",
                        "Unbelievable quality!!! Changed my life overnight!!!",
                        "Spectacular results instantly!!! 100% satisfaction guaranteed!!!"
                    ],
                    'label': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 for real, 0 for fake
                }
                
                train_df = pd.DataFrame(sample_data)
                train_df['processed_text'] = train_df['text'].apply(preprocess_text)
                
                # Feature extraction
                vectorizer = TfidfVectorizer(max_features=5000)
                X_train_tfidf = vectorizer.fit_transform(train_df['processed_text'])
                
                # Train the model
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train_tfidf, train_df['label'])
                
                # Process all reviews
                st.write("Classifying reviews...")
                X_all_tfidf = vectorizer.transform(data['processed_text'])
                data['prediction'] = model.predict(X_all_tfidf)
            
            # Display results
            data['review_type'] = data['prediction'].map({1: 'Real', 0: 'Fake'})
            
            # Create separate dataframes for real and fake reviews
            real_reviews = data[data['prediction'] == 1]
            fake_reviews = data[data['prediction'] == 0]
            
            st.write(f"Found {len(real_reviews)} real reviews and {len(fake_reviews)} fake reviews.")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Real Reviews:")
                st.dataframe(real_reviews[['text', 'review_type']])
                st.markdown(get_download_link(real_reviews, 'real_reviews.csv', 'Download Real Reviews CSV'), unsafe_allow_html=True)
            
            with col2:
                st.write("Fake Reviews:")
                st.dataframe(fake_reviews[['text', 'review_type']])
                st.markdown(get_download_link(fake_reviews, 'fake_reviews.csv', 'Download Fake Reviews CSV'), unsafe_allow_html=True)
            
        else:
            st.error(f"CSV file must contain the following columns: {', '.join(required_cols)}")
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file with review data.")
    st.write("The CSV should have a 'text' column containing the review text.")
    st.write("If you include a 'label' column (1 for real, 0 for fake), the model will train on your data.")
    st.write("Otherwise, a pre-trained model will be used to classify the reviews.")

# Instructions for running the script
st.markdown("""
## How to Run This Project
1. Save this code as `fake_review_detection.py`
2. Install required libraries: `pip install streamlit pandas numpy scikit-learn`
3. Run the app: `streamlit run fake_review_detection.py`
4. Upload a CSV file with reviews to classify them
""")
