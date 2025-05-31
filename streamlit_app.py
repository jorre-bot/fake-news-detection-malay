import streamlit as st
import joblib
import sqlite3
import pandas as pd
from datetime import datetime
import os
import hashlib
import re
import numpy as np
import secrets
import base64

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = {}
if 'last_activity' not in st.session_state:
    st.session_state.last_activity = datetime.now()

# Security Constants
MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_TIME = 300  # 5 minutes in seconds
SESSION_TIMEOUT = 3600  # 1 hour in seconds

def hash_password_with_salt(password):
    """Hash password with a secure salt"""
    salt = secrets.token_hex(16)
    # Use SHA-256 with a secure salt
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode(),
        salt.encode(),
        100000  # Number of iterations
    )
    # Combine salt and hash
    return f"{salt}${base64.b64encode(hashed).decode()}"

def verify_password(password, stored_password):
    """Verify a password against its stored hash"""
    try:
        salt, stored_hash = stored_password.split('$')
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        )
        return base64.b64encode(hashed).decode() == stored_hash
    except:
        return False

def sanitize_input(text):
    """Sanitize user input to prevent XSS and injection attacks"""
    if text is None:
        return ""
    # Remove HTML tags and special characters
    text = re.sub(r'[<>&\'";\\/]', '', text)
    # Limit length
    return text[:1000]  # Limit to 1000 characters

def check_rate_limit(username):
    """Check if user has exceeded login attempts"""
    now = datetime.now()
    if username in st.session_state.login_attempts:
        attempts, last_attempt = st.session_state.login_attempts[username]
        # Reset attempts if lockout time has passed
        if (now - last_attempt).total_seconds() > LOCKOUT_TIME:
            st.session_state.login_attempts[username] = (0, now)
            return True
        # Check if max attempts exceeded
        if attempts >= MAX_LOGIN_ATTEMPTS:
            return False
    return True

def update_login_attempts(username, success):
    """Update login attempts counter"""
    now = datetime.now()
    if success:
        # Reset on successful login
        st.session_state.login_attempts[username] = (0, now)
    else:
        # Increment attempts on failure
        attempts = st.session_state.login_attempts.get(username, (0, now))[0] + 1
        st.session_state.login_attempts[username] = (attempts, now)

def check_session_timeout():
    """Check if the session has timed out"""
    if st.session_state.authenticated:
        now = datetime.now()
        if (now - st.session_state.last_activity).total_seconds() > SESSION_TIMEOUT:
            st.session_state.authenticated = False
            st.session_state.username = None
            return True
    st.session_state.last_activity = datetime.now()
    return False

# Database functions
def init_db():
    conn = sqlite3.connect('fake_news.db', check_same_thread=False)
    c = conn.cursor()
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         username TEXT UNIQUE NOT NULL,
         email TEXT UNIQUE NOT NULL,
         password TEXT NOT NULL)
    ''')
    # Create detections table
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         user_id INTEGER,
         news_text TEXT NOT NULL,
         prediction TEXT NOT NULL,
         confidence TEXT NOT NULL,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
         FOREIGN KEY (user_id) REFERENCES users (id))
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password with strong requirements"""
    if len(password) < 12:
        return False, "Password must be at least 12 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter (A-Z)"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter (a-z)"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number (0-9)"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)"
    
    if any(common in password.lower() for common in ['password', '123456', 'qwerty', 'admin']):
        return False, "Password cannot contain common words like 'password', '123456', 'qwerty', 'admin'"
    
    return True, "Password is valid"

def register_user(username, email, password):
    """Register user with enhanced security"""
    if not validate_email(email):
        return False, "Invalid email format"
    
    valid_password, msg = validate_password(password)
    if not valid_password:
        return False, msg
    
    # Sanitize inputs
    username = sanitize_input(username)
    email = sanitize_input(email)
    
    if not username or not email:
        return False, "Invalid username or email format"
    
    try:
        conn = sqlite3.connect('fake_news.db')
        c = conn.cursor()
        hashed_password = hash_password_with_salt(password)
        c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                 (username, email, hashed_password))
        conn.commit()
        conn.close()
        return True, "Registration successful"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists"
    except Exception as e:
        return False, "An error occurred during registration"

def login_user(username, password):
    """Login user with enhanced security"""
    username = sanitize_input(username)
    
    if not check_rate_limit(username):
        remaining_time = LOCKOUT_TIME
        return False, f"Too many login attempts. Please try again in {remaining_time} seconds."
    
    try:
        conn = sqlite3.connect('fake_news.db')
        c = conn.cursor()
        c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
        result = c.fetchone()
        conn.close()
        
        if result and verify_password(password, result[1]):
            update_login_attempts(username, True)
            return True, "Login successful"
        
        update_login_attempts(username, False)
        return False, "Invalid username or password"
    except Exception as e:
        return False, "An error occurred during login"

def save_detection(user_id, news_text, prediction, confidence):
    conn = sqlite3.connect('fake_news.db')
    c = conn.cursor()
    c.execute('INSERT INTO detections (user_id, news_text, prediction, confidence) VALUES (?, ?, ?, ?)',
             (user_id, news_text, prediction, confidence))
    conn.commit()
    conn.close()

def get_user_id(username):
    conn = sqlite3.connect('fake_news.db')
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def get_user_history(user_id):
    conn = sqlite3.connect('fake_news.db')
    df = pd.read_sql_query('SELECT news_text, prediction, confidence FROM detections WHERE user_id = ? ORDER BY timestamp DESC',
                          conn, params=(user_id,))
    conn.close()
    return df

# Initialize database
init_db()

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_fake_news_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_fake_news(text, model):
    try:
        # Use ML model for prediction
        prediction = model.predict([text])[0]
        probability = model.predict_proba([text])[0]
        
        # Convert numeric prediction to text
        if isinstance(prediction, (int, np.integer)):
            prediction = "FAKE" if prediction == 0 else "REAL"
            
        # Determine confidence based on probability
        confidence = "High" if max(probability) > 0.7 else "Medium"
        return prediction, confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def validate_news_text(text):
    """Validate the input news text"""
    if not text or len(text.strip()) == 0:
        return False, "Please enter some text to analyze."
    
    # Count words (split by whitespace)
    words = text.strip().split()
    word_count = len(words)
    
    # Check minimum word count (25 words)
    if word_count < 25:
        return False, f"News text must contain at least 25 words for accurate analysis. Current word count: {word_count}"
    
    return True, ""

def show_main_app():
    st.title("Malay Fake News Detection")
    st.write(f"Welcome, {st.session_state.username}!")
    
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
    
    st.write("This tool uses machine learning to analyze and detect potential fake news in Malay language texts.")
    
    st.markdown("""
    ### 📝 News Text Requirements
    Please ensure your text meets these requirements:
    1. **Language**: Must be in Malay
    2. **Minimum length**: 25 words (not just the title)
    3. **Content**: Include both title and full news content
    """)
    
    st.markdown("""
    ### 📰 Example News Text:
    Below is an example of how to format your news text:
    """)
    
    example_text = """KOTA BHARU: Jabatan Pendidikan Negeri Kelantan mengumumkan penutupan sementara semua sekolah di daerah Kota Bharu bermula esok susulan peningkatan mendadak kes Covid-19 di kawasan tersebut. Pengarah Pendidikan Negeri, Dato' Ahmad bin Ibrahim berkata keputusan ini dibuat selepas berbincang dengan Jabatan Kesihatan Negeri dan pihak berkuasa tempatan. Menurut beliau, sebanyak 15 buah sekolah telah melaporkan kes positif dalam tempoh seminggu lepas. "Penutupan ini akan berlangsung selama dua minggu untuk membolehkan kerja-kerja sanitasi dijalankan secara menyeluruh. Pembelajaran akan diteruskan secara dalam talian menggunakan platform Google Classroom dan Microsoft Teams," katanya dalam satu kenyataan media hari ini. Beliau menambah bahawa pemantauan rapi akan dilakukan dan keputusan untuk membuka semula sekolah akan dibuat berdasarkan penilaian risiko oleh pihak berkuasa kesihatan."""
    
    st.code(example_text, language="text")
    
    news_text = st.text_area(
        "Enter news text to analyze:",
        help="Enter your news content in Malay language (minimum 25 words).",
        height=300
    )
    
    if st.button("Analyze"):
        # Validate input
        is_valid, error_message = validate_news_text(news_text)
        if not is_valid:
            st.error(f"❌ {error_message}")
            st.info("👆 Please check the example format above and try again.")
        else:
            model = load_model()
            if model is not None:
                prediction, confidence = predict_fake_news(news_text, model)
                if prediction is not None:
                    result_color = "#ff6b6b" if prediction == "FAKE" else "#51cf66"
                    st.markdown(f"""
                    <div style='padding: 20px; border-radius: 5px; background-color: {result_color}; color: white;'>
                        <h3>Prediction: {prediction}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    try:
                        # Save detection to history
                        user_id = get_user_id(st.session_state.username)
                        if user_id is not None:
                            save_detection(user_id, news_text, prediction, "High" if prediction == "FAKE" else confidence)
                        else:
                            st.error("Error: Could not find user ID")
                    except Exception as e:
                        st.error(f"Error saving detection: {str(e)}")
    
    # Show detection history
    st.subheader("Detection History")
    try:
        user_id = get_user_id(st.session_state.username)
        if user_id is not None:
            history = get_user_history(user_id)
            if not history.empty:
                st.dataframe(history)
            else:
                st.info("No detection history yet.")
        else:
            st.error("Error: Could not find user ID")
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")

# Authentication UI
def show_auth_ui():
    if check_session_timeout():
        st.warning("Your session has expired. Please login again.")
    
    st.title("Malay Fake News Detection")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                success, message = login_user(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Register")
            
            if submit:
                success, message = register_user(new_username, new_email, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
                    st.markdown("""
                    ### Password Requirements:
                    - At least 12 characters long
                    - Must contain uppercase letter (A-Z)
                    - Must contain lowercase letter (a-z)
                    - Must contain number (0-9)
                    - Must contain special character (!@#$%^&*(),.?":{}|<>)
                    - Cannot contain common words (password, 123456, etc.)
                    """)

# Main app flow
if not st.session_state.authenticated:
    show_auth_ui()
else:
    show_main_app() 