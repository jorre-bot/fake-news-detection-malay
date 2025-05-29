import streamlit as st
import joblib
import sqlite3
import pandas as pd
from datetime import datetime
import os

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('best_fake_news_model.pkl')

# Initialize the database
def init_db():
    conn = sqlite3.connect('fake_news.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         news_text TEXT NOT NULL,
         prediction TEXT NOT NULL,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    ''')
    conn.commit()
    conn.close()

# Save detection to database
def save_detection(news_text, prediction):
    conn = sqlite3.connect('fake_news.db')
    c = conn.cursor()
    c.execute('INSERT INTO detections (news_text, prediction) VALUES (?, ?)',
              (news_text, prediction))
    conn.commit()
    conn.close()

# Get detection history
def get_history():
    conn = sqlite3.connect('fake_news.db')
    history = pd.read_sql_query('SELECT * FROM detections ORDER BY timestamp DESC', conn)
    conn.close()
    return history

# Set up the Streamlit page
st.set_page_config(
    page_title="Malay Fake News Detection",
    page_icon="🔍",
    layout="wide"
)

# Initialize database
init_db()

# Main UI
st.title("Malay Fake News Detection")
st.write("This tool uses machine learning to analyze and detect potential fake news in Malay language texts.")

# Example text in sidebar
with st.sidebar:
    st.header("Example Format")
    st.info("""
    KOTA BHARU: Polis sedang giat mengesan tiga individu yang disyaki terlibat dalam satu kes rompakan bersenjata di sebuah kedai emas di pusat bandar pagi tadi. Ketua Polis Daerah Kota Bharu berkata, usaha menjejaki suspek sedang dipergiat dan orang ramai diminta menyalurkan maklumat sekiranya melihat atau mengenali mana-mana suspek yang terlibat. Siasatan lanjut masih dijalankan bagi mengenal pasti motif kejadian.
    """)
    st.caption("Note: News should be in Malay language and follow a similar formal news format.")

# Input area
news_text = st.text_area(
    "Enter news text to analyze:",
    height=200,
    placeholder="Paste your news text here (minimum 50 characters)..."
)

# Analyze button
if st.button("Analyze", type="primary"):
    if len(news_text.strip()) < 50:
        st.error("Please enter at least 50 characters of news text.")
    else:
        with st.spinner("Analyzing..."):
            # Load model and make prediction
            model = load_model()
            prediction = model.predict([news_text])[0]
            result = "FAKE" if prediction == 0 else "REAL"
            
            # Save detection
            save_detection(news_text, result)
            
            # Show result
            if result == "FAKE":
                st.error(f"Prediction: {result}")
            else:
                st.success(f"Prediction: {result}")
            
            st.write("Confidence: High")

# Show history in an expander
with st.expander("View Detection History"):
    history = get_history()
    if not history.empty:
        st.dataframe(
            history[['timestamp', 'news_text', 'prediction']],
            column_config={
                "timestamp": "Time",
                "news_text": "News Text",
                "prediction": "Prediction"
            },
            hide_index=True
        )
    else:
        st.info("No detection history yet.") 