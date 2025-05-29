import streamlit as st
import joblib
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import hashlib
import re
import numpy as np
import bcrypt
import html
from functools import wraps
import time

# Security configurations
MAX_LOGIN_ATTEMPTS = 3
LOGIN_TIMEOUT = 300  # 5 minutes in seconds
SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Initialize session state for security
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0
if 'last_login_attempt' not in st.session_state:
    st.session_state.last_login_attempt = None
if 'last_activity' not in st.session_state:
    st.session_state.last_activity = datetime.now()

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None

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
         password TEXT NOT NULL,
         failed_attempts INTEGER DEFAULT 0,
         last_login DATETIME,
         account_locked BOOLEAN DEFAULT 0,
         lock_until DATETIME)
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
    # Add audit trail
    c.execute('''
        CREATE TABLE IF NOT EXISTS audit_log
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         user_id INTEGER,
         action TEXT NOT NULL,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
         ip_address TEXT,
         FOREIGN KEY (user_id) REFERENCES users (id))
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed):
    """Verify password using bcrypt"""
    try:
        return bcrypt.checkpw(password.encode(), hashed)
    except:
        return False

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def sanitize_input(text):
    """Sanitize user input to prevent XSS attacks"""
    if text is None:
        return ""
    # Escape HTML special characters
    text = html.escape(text)
    # Remove potentially dangerous characters
    text = re.sub(r'[^\w\s\-.,:]', '', text)
    return text

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
    
    try:
        conn = sqlite3.connect('fake_news.db')
        c = conn.cursor()
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                 (username, email, hashed_password))
        conn.commit()
        conn.close()
        return True, "Registration successful"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists"
    except Exception as e:
        return False, f"Error: {str(e)}"

def login_user(username, password):
    """Login user with enhanced security"""
    if not rate_limit_login():
        return False
    
    username = sanitize_input(username)
    
    conn = sqlite3.connect('fake_news.db')
    c = conn.cursor()
    c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    
    if result and verify_password(password, result[1]):
        # Reset login attempts on successful login
        c.execute('UPDATE users SET failed_attempts = 0, last_login = CURRENT_TIMESTAMP WHERE username = ?',
                 (username,))
        conn.commit()
        st.session_state.login_attempts = 0
        log_activity(result[0], "Successful login")
        conn.close()
        return True
    else:
        # Increment failed attempts
        st.session_state.login_attempts += 1
        st.session_state.last_login_attempt = time.time()
        if result:
            c.execute('UPDATE users SET failed_attempts = failed_attempts + 1 WHERE username = ?',
                     (username,))
            conn.commit()
        conn.close()
        return False

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
    df = pd.read_sql_query('SELECT news_text, prediction, confidence, timestamp FROM detections WHERE user_id = ? ORDER BY timestamp DESC',
                          conn, params=(user_id,))
    conn.close()
    return df

def check_session_timeout():
    """Check if the session has timed out"""
    if st.session_state.authenticated:
        if datetime.now() - st.session_state.last_activity > timedelta(seconds=SESSION_TIMEOUT):
            st.session_state.authenticated = False
            st.session_state.username = None
            return True
    st.session_state.last_activity = datetime.now()
    return False

def rate_limit_login():
    """Implement rate limiting for login attempts"""
    if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
        if st.session_state.last_login_attempt:
            time_passed = time.time() - st.session_state.last_login_attempt
            if time_passed < LOGIN_TIMEOUT:
                remaining = int(LOGIN_TIMEOUT - time_passed)
                st.error(f"Too many login attempts. Please try again in {remaining} seconds.")
                return False
            else:
                st.session_state.login_attempts = 0
    return True

def log_activity(user_id, action):
    """Log user activity for security audit"""
    conn = sqlite3.connect('fake_news.db')
    c = conn.cursor()
    c.execute('INSERT INTO audit_log (user_id, action) VALUES (?, ?)',
             (user_id, sanitize_input(action)))
    conn.commit()
    conn.close()

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
        prediction = model.predict([text])[0]
        # Convert numeric prediction to text
        if isinstance(prediction, (int, np.integer)):
            prediction = "FAKE" if prediction == 0 else "REAL"
        probability = model.predict_proba([text])[0]
        confidence = "High" if max(probability) > 0.7 else "Medium"
        return prediction, confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Authentication UI
def show_auth_ui():
    st.title("Malay Fake News Detection")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if login_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
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

def validate_news_text(text):
    """Validate the input news text"""
    if not text or len(text.strip()) == 0:
        return False, "Please enter some text to analyze."
    
    # Check minimum length (20 characters)
    if len(text.strip()) < 20:
        return False, "News text must be at least 20 characters long to be analyzed properly."
    
    # Check if it starts with a location prefix (common in Malay news)
    location_prefixes = ["MELAKA:", "KOTA BHARU:", "KUALA LUMPUR:", "JOHOR BAHRU:", "PUTRAJAYA:", "SHAH ALAM:", "IPOH:", "SEREMBAN:", "ALOR SETAR:", "GEORGE TOWN:"]
    
    # Convert text to uppercase for case-insensitive comparison
    text_upper = text.strip().upper()
    
    # Check if text starts with location prefix
    has_prefix = any(text_upper.startswith(prefix) for prefix in location_prefixes)
    if not has_prefix:
        return False, f"News text must start with a location prefix. Valid prefixes include: {', '.join(location_prefixes)}"
    
    return True, ""

# Main app UI
def show_main_app():
    # Check session timeout
    if check_session_timeout():
        st.warning("Your session has expired. Please login again.")
        return
    
    st.title("Malay Fake News Detection")
    st.write(f"Welcome, {st.session_state.username}!")
    
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
    
    st.write("This tool uses machine learning to analyze and detect potential fake news in Malay language texts.")
    
    st.markdown("""
    ### 📝 Required News Format
    Your news text **must** follow this format:
    """)
    
    example_text = """KUALA LUMPUR: Dalam satu makluman tular, sebuah universiti tempatan dikatakan menawarkan biasiswa penuh tanpa sebarang syarat kepada semua pelajar baharu."""
    
    st.code(example_text, language="text")
    
    st.markdown("""
    ### ✅ Format Requirements:
    1. **Must start** with a location prefix followed by colon (e.g., 'KOTA BHARU:', 'KUALA LUMPUR:', etc.)
    2. **Minimum length**: 20 characters
    """)
    
    news_text = st.text_area(
        "Enter news text to analyze:",
        help="Enter your news content following the format requirements above.",
        height=200
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
                        <p>Confidence: {confidence}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    try:
                        # Save detection to history
                        user_id = get_user_id(st.session_state.username)
                        if user_id is not None:
                            save_detection(user_id, news_text, prediction, confidence)
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

# Main app flow
if not st.session_state.authenticated:
    show_auth_ui()
else:
    show_main_app() 