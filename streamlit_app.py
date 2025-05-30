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
import logging
from logging.handlers import RotatingFileHandler

# Set up logging configuration
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logging
    log_file = os.path.join('logs', 'app.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                log_file, 
                maxBytes=1024*1024,  # 1MB per file
                backupCount=5  # Keep 5 backup files
            ),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Log application startup
logger.info("Application started")

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    logger.info("Initialized authentication state")
if 'username' not in st.session_state:
    st.session_state.username = None
    logger.info("Initialized username state")
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
    try:
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
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def hash_password(password):
    try:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        logger.debug("Password hashed successfully")
        return hashed
    except Exception as e:
        logger.error(f"Password hashing failed: {str(e)}")
        raise

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

def register_user(username, email, password):
    logger.info(f"Attempting to register user: {username}")
    if not validate_email(email):
        logger.warning(f"Invalid email format attempted: {email}")
        return False, "Invalid email format"
    
    valid_password, msg = validate_password(password)
    if not valid_password:
        logger.warning(f"Invalid password format for user {username}: {msg}")
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
        logger.info(f"User registered successfully: {username}")
        return True, "Registration successful"
    except sqlite3.IntegrityError:
        logger.warning(f"Registration failed - username or email exists: {username}")
        return False, "Username or email already exists"
    except Exception as e:
        logger.error(f"Registration error for {username}: {str(e)}")
        return False, f"Error: {str(e)}"

def login_user(username, password):
    logger.info(f"Login attempt for user: {username}")
    try:
        conn = sqlite3.connect('fake_news.db')
        c = conn.cursor()
        hashed_password = hash_password(password)
        c.execute('SELECT id FROM users WHERE username = ? AND password = ?',
                 (username, hashed_password))
        result = c.fetchone()
        conn.close()
        
        if result:
            logger.info(f"Login successful for user: {username}")
            return True
        else:
            logger.warning(f"Login failed for user: {username}")
            return False
    except Exception as e:
        logger.error(f"Login error for {username}: {str(e)}")
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
    df = pd.read_sql_query('SELECT news_text, prediction, confidence FROM detections WHERE user_id = ? ORDER BY timestamp DESC',
                          conn, params=(user_id,))
    conn.close()
    return df

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
        logger.info("Starting news prediction")
        prediction = model.predict([text])[0]
        if isinstance(prediction, (int, np.integer)):
            prediction = "FAKE" if prediction == 0 else "REAL"
        probability = model.predict_proba([text])[0]
        confidence = "High" if max(probability) > 0.7 else "Medium"
        logger.info(f"Prediction complete: {prediction} with {confidence} confidence")
        return prediction, confidence
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None, None

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

def show_main_app():
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
                if login_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    logger.info(f"User logged in successfully: {username}")
                    st.success("Login successful!")
                    st.rerun()
                else:
                    logger.warning(f"Failed login attempt for user: {username}")
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
                    logger.info(f"New user registered: {new_username}")
                    st.success(message)
                else:
                    logger.warning(f"Registration failed for {new_username}: {message}")
                    st.error(message)

# Main app flow
if not st.session_state.authenticated:
    show_auth_ui()
else:
    show_main_app() 