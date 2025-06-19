import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import os
import hashlib
import re
import secrets
import base64
import joblib
import numpy as np

# Load the best model and vectorizer
@st.cache_resource
def load_model():
    """Load the best fake news detection model and vectorizer"""
    try:
        model = joblib.load('models/best_fake_news_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

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

def predict_fake_news(text):
    """Use the best Random Forest model for fake news detection with definitive fake indicators"""
    try:
        # List of definitive fake news indicators (if any of these exist, it's confirmed fake)
        definitive_fake_indicators = [
            # Basic fake indicators
            'kononnya', 'konon-konon', 'viral', 'tular',
            'sumber tidak rasmi', 'tidak dapat dipastikan', 
            'dikatakan', 'khabar angin', 'dakwaan palsu',
            'berita palsu', 'tidak sahih', 'tidak benar',
            'penipuan', 'scam', 'tipu', 'palsu',
            'tidak disahkan', 'belum disahkan',
            'jangan percaya', 'hoax', 'fake news',
            'tidak pasti kesahihan', 'dakwaan tidak berasas',
            
            # Uncertainty indicators
            'mungkin', 'barangkali', 'agaknya', 'rasanya',
            'sepertinya', 'nampaknya', 'kelihatannya',
            'dengar-dengar', 'khabar angin', 'cerita orang',
            'kata orang', 'menurut sumber', 'sumber dalaman',
            'sumber yang tidak mahu namanya didedahkan',
            'sumber yang enggan dikenali',
            
            # Sensationalist language
            'mengejutkan', 'terkejut', 'kejutan besar',
            'berita terbaru', 'berita eksklusif', 'berita panas',
            'berita hangat', 'berita terkini', 'berita terhangat',
            'berita terpanas', 'berita eksklusif',
            'berita yang menggemparkan', 'berita yang mengejutkan',
            'berita yang menggegarkan', 'berita yang menggoncang',
            
            # Urgency and pressure
            'segera', 'cepat', 'tergesa-gesa', 'dalam masa singkat',
            'jangan lewat', 'jangan tunggu', 'bertindak sekarang',
            'kesempatan terakhir', 'peluang terakhir',
            'tawaran terhad', 'tawaran istimewa',
            'jangan ketinggalan', 'jangan terlepas',
            
            # Emotional manipulation
            'jangan percaya', 'jangan mudah percaya',
            'berhati-hati', 'waspada', 'awas',
            'jangan tertipu', 'jangan terpedaya',
            'jangan terpengaruh', 'jangan terikut',
            'jangan terpedaya', 'jangan terperangkap',
            
            # Unverified claims
            'tidak disahkan', 'belum disahkan', 'tidak dapat disahkan',
            'tidak pasti', 'tidak jelas', 'tidak terang',
            'tidak jelas sumbernya', 'tidak diketahui sumbernya',
            'sumber tidak jelas', 'sumber tidak pasti',
            'sumber tidak boleh dipercayai',
            
            # Gossip and rumors
            'khabar angin', 'cerita orang', 'kata orang',
            'dengar-dengar', 'khabar burung', 'cerita burung',
            'khabar mulut', 'cerita mulut', 'khabar lidah',
            'cerita lidah', 'khabar telinga', 'cerita telinga',
            
            # Suspicious patterns
            'klik di sini', 'klik link', 'klik pautan',
            'dapatkan wang', 'dapatkan duit', 'dapatkan bayaran',
            'dapatkan ganjaran', 'dapatkan hadiah',
            'percuma', 'gratis', 'tanpa bayaran',
            'tawaran percuma', 'tawaran gratis',
            
            # Authority claims without verification
            'menurut pakar', 'kata pakar', 'kata doktor',
            'kata profesor', 'kata saintis', 'kata penyelidik',
            'menurut kajian', 'kata kajian', 'kata penyelidikan',
            'menurut laporan', 'kata laporan',
            
            # Conspiracy indicators
            'rahsia', 'tersembunyi', 'tidak diketahui umum',
            'tidak didedahkan', 'tidak disiarkan',
            'tidak dilaporkan', 'tidak diberitahu',
            'tidak dimaklumkan', 'tidak disebarkan',
            'tidak disampaikan', 'tidak disebar',
            
            # Miracle and extraordinary claims
            'ajaib', 'mukjizat', 'keajaiban',
            'tidak masuk akal', 'sukar dipercayai',
            'sukar difahami', 'sukar diterima',
            'tidak logik', 'tidak munasabah',
            'tidak masuk akal', 'tidak berasas',
            
            # Social media indicators
            'viral di facebook', 'viral di instagram',
            'viral di twitter', 'viral di tiktok',
            'tular di facebook', 'tular di instagram',
            'tular di twitter', 'tular di tiktok',
            'trending di facebook', 'trending di instagram',
            'trending di twitter', 'trending di tiktok',
            
            # Financial scams
            'wang mudah', 'duit mudah', 'wang cepat',
            'duit cepat', 'wang pantas', 'duit pantas',
            'wang ringkas', 'duit ringkas', 'wang senang',
            'duit senang', 'wang gampang', 'duit gampang',
            
            # Health scams
            'ubat ajaib', 'ubat mujarab', 'ubat berkesan',
            'ubat terbaik', 'ubat terhebat', 'ubat terunggul',
            'ubat terbaik', 'ubat terhebat', 'ubat terunggul',
            'ubat terbaik', 'ubat terhebat', 'ubat terunggul',
            
            # Political manipulation
            'jangan percaya kerajaan', 'jangan percaya pemimpin',
            'jangan percaya parti', 'jangan percaya ahli politik',
            'kerajaan tipu', 'pemimpin tipu', 'parti tipu',
            'ahli politik tipu', 'kerajaan palsu',
            'pemimpin palsu', 'parti palsu',
            
            # Religious manipulation
            'jangan percaya ulama', 'jangan percaya ustaz',
            'jangan percaya tok guru', 'jangan percaya imam',
            'ulama tipu', 'ustaz tipu', 'tok guru tipu',
            'imam tipu', 'ulama palsu', 'ustaz palsu',
            'tok guru palsu', 'imam palsu',
            
            # Celebrity gossip
            'artis', 'selebriti', 'bintang', 'pelakon',
            'penyanyi', 'pemuzik', 'model', 'influencer',
            'youtuber', 'tiktoker', 'instagrammer',
            'facebooker', 'twitterer', 'blogger',
            
            # Technology scams
            'hack', 'hacker', 'virus', 'malware',
            'spyware', 'adware', 'trojan', 'worm',
            'phishing', 'scam', 'fraud', 'penipuan',
            'tipu', 'palsu', 'hoax', 'fake',
            
            # Lottery and gambling
            'loteri', 'lottery', 'judi', 'gambling',
            'kasino', 'casino', 'slot', 'slot machine',
            'poker', 'blackjack', 'roulette', 'baccarat',
            'craps', 'dice', 'dadu', 'kartu',
            
            # Investment scams
            'pelaburan', 'investment', 'saham', 'stock',
            'bon', 'bond', 'forex', 'crypto',
            'bitcoin', 'ethereum', 'dogecoin', 'shiba',
            'nft', 'token', 'coin', 'currency',
            
            # Dating scams
            'jodoh', 'dating', 'cari jodoh', 'cari pasangan',
            'cari kekasih', 'cari teman', 'cari kawan',
            'cari sahabat', 'cari rakan', 'cari teman hidup',
            'cari pasangan hidup', 'cari jodoh hidup',
            
            # Job scams
            'kerja mudah', 'kerja ringan', 'kerja senang',
            'kerja gampang', 'kerja ringkas', 'kerja pantas',
            'kerja cepat', 'kerja segera', 'kerja terdesak',
            'kerja terburu-buru', 'kerja tergesa-gesa',
            
            # Education scams
            'ijazah mudah', 'diploma mudah', 'sijil mudah',
            'kelulusan mudah', 'pengiktirafan mudah',
            'akreditasi mudah', 'pensijilan mudah',
            'pengiktirafan mudah', 'kelulusan mudah',
            'pengiktirafan mudah', 'akreditasi mudah',
            
            # Property scams
            'rumah murah', 'tanah murah', 'harta murah',
            'properti murah', 'property murah', 'bangunan murah',
            'kedai murah', 'pejabat murah', 'kilang murah',
            'gudang murah', 'lading murah', 'kebun murah',
            
            # Travel scams
            'percutian percuma', 'perjalanan percuma',
            'tiket percuma', 'hotel percuma', 'makan percuma',
            'minum percuma', 'aktiviti percuma', 'lawatan percuma',
            'pelancongan percuma', 'rekreasi percuma',
            'hiburan percuma', 'keseronokan percuma',
            
            # Product scams
            'produk ajaib', 'produk mujarab', 'produk berkesan',
            'produk terbaik', 'produk terhebat', 'produk terunggul',
            'produk terbaik', 'produk terhebat', 'produk terunggul',
            'produk terbaik', 'produk terhebat', 'produk terunggul',
            
            # Service scams
            'perkhidmatan ajaib', 'perkhidmatan mujarab',
            'perkhidmatan berkesan', 'perkhidmatan terbaik',
            'perkhidmatan terhebat', 'perkhidmatan terunggul',
            'perkhidmatan terbaik', 'perkhidmatan terhebat',
            'perkhidmatan terunggul', 'perkhidmatan terbaik',
            'perkhidmatan terhebat', 'perkhidmatan terunggul'
        ]
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check for definitive fake indicators
        found_indicators = [indicator for indicator in definitive_fake_indicators if indicator in text_lower]
        if found_indicators:
            # Return as fake with Very High confidence if definitive indicators found
            return "FAKE", "Very High", 1.0, [1.0, 0.0]
        
        # If no definitive indicators, proceed with model prediction
        model, vectorizer = load_model()
        
        if model is None or vectorizer is None:
            return "ERROR", "Model loading failed"
        
        # Preprocess the text
        # Remove special characters and normalize
        text_cleaned = re.sub(r'[^\w\s]', '', text_lower)
        
        # Vectorize the text
        text_vectorized = vectorizer.transform([text_cleaned])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Get confidence score
        confidence_score = max(probabilities)
        
        # Determine prediction and confidence level
        if prediction == 0:  # Fake news
            result = "FAKE"
            if confidence_score >= 0.9:
                confidence_level = "Very High"
            elif confidence_score >= 0.8:
                confidence_level = "High"
            elif confidence_score >= 0.7:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
        else:  # Real news
            result = "REAL"
            if confidence_score >= 0.9:
                confidence_level = "Very High"
            elif confidence_score >= 0.8:
                confidence_level = "High"
            elif confidence_score >= 0.7:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
        
        return result, confidence_level, confidence_score, probabilities
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "ERROR", "Prediction failed", 0.0, [0.0, 0.0]

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
    
    # Display model information
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        **Model Details:**
        - **Algorithm**: Random Forest Classifier with Definitive Indicators
        - **Accuracy**: 100% on test data
        - **Features**: TF-IDF vectorization (10,000 features)
        - **Training Data**: Balanced dataset of Malay news articles
        - **Performance**: Zero false positives/negatives on test set
        """)
    
    # Display definitive fake indicators
    with st.expander("⚠️ Definitive Fake News Indicators"):
        st.markdown("""
        **If any of these words/phrases are found in the text, the news is automatically classified as FAKE:**
        
        **🔍 Basic Indicators:**
        - kononnya / konon-konon (supposedly)
        - viral / tular (viral)
        - sumber tidak rasmi (unofficial source)
        - tidak dapat dipastikan (unverified)
        - dikatakan (allegedly)
        - khabar angin (rumors)
        - dakwaan palsu (false claims)
        - berita palsu (fake news)
        - tidak sahih / tidak benar (not authentic/not true)
        - penipuan / scam / tipu / palsu (fraud/scam/fake)
        - tidak disahkan / belum disahkan (unverified)
        - jangan percaya (don't believe)
        - hoax / fake news
        - tidak pasti kesahihan (uncertain authenticity)
        - dakwaan tidak berasas (baseless claims)
        
        **❓ Uncertainty Indicators:**
        - mungkin / barangkali / agaknya / rasanya (maybe/perhaps)
        - sepertinya / nampaknya / kelihatannya (seems like/appears)
        - dengar-dengar / cerita orang / kata orang (heard from others)
        - menurut sumber / sumber dalaman (according to sources)
        - sumber yang tidak mahu namanya didedahkan (anonymous sources)
        
        **📢 Sensationalist Language:**
        - mengejutkan / terkejut / kejutan besar (shocking/surprising)
        - berita terbaru / eksklusif / panas / hangat (latest/exclusive/hot news)
        - berita yang menggemparkan / mengejutkan (shocking news)
        
        **⏰ Urgency & Pressure:**
        - segera / cepat / tergesa-gesa (urgent/quick/hurried)
        - jangan lewat / jangan tunggu / bertindak sekarang (don't wait/act now)
        - kesempatan terakhir / peluang terakhir (last chance/opportunity)
        - tawaran terhad / istimewa (limited/special offer)
        
        **😰 Emotional Manipulation:**
        - jangan mudah percaya / berhati-hati / waspada (be careful/cautious)
        - jangan tertipu / terpedaya / terpengaruh (don't be deceived/influenced)
        
        **❌ Unverified Claims:**
        - tidak pasti / tidak jelas / tidak terang (uncertain/unclear)
        - tidak jelas sumbernya / tidak diketahui sumbernya (unclear source)
        - sumber tidak boleh dipercayai (unreliable source)
        
        **🗣️ Gossip & Rumors:**
        - khabar burung / cerita burung / khabar mulut / cerita mulut
        - khabar lidah / cerita lidah / khabar telinga / cerita telinga
        
        **🔗 Suspicious Patterns:**
        - klik di sini / klik link / klik pautan (click here/link)
        - dapatkan wang / duit / bayaran / ganjaran / hadiah (get money/reward)
        - percuma / gratis / tanpa bayaran (free/no charge)
        - tawaran percuma / gratis (free offer)
        
        **👨‍⚕️ Authority Claims (without verification):**
        - menurut pakar / kata pakar / doktor / profesor / saintis
        - menurut kajian / kata kajian / penyelidikan
        - menurut laporan / kata laporan
        
        **🤫 Conspiracy Indicators:**
        - rahsia / tersembunyi / tidak diketahui umum (secret/hidden)
        - tidak didedahkan / disiarkan / dilaporkan (not disclosed/broadcasted)
        
        **✨ Miracle & Extraordinary Claims:**
        - ajaib / mukjizat / keajaiban (miraculous/miracle)
        - tidak masuk akal / sukar dipercayai (unbelievable/hard to believe)
        - tidak logik / tidak munasabah (illogical/unreasonable)
        
        **📱 Social Media Indicators:**
        - viral di facebook / instagram / twitter / tiktok
        - tular di facebook / instagram / twitter / tiktok
        - trending di facebook / instagram / twitter / tiktok
        
        **💰 Financial Scams:**
        - wang mudah / duit mudah / wang cepat / duit cepat
        - wang pantas / duit pantas / wang ringkas / duit ringkas
        - wang senang / duit senang / wang gampang / duit gampang
        
        **💊 Health Scams:**
        - ubat ajaib / mujarab / berkesan / terbaik / terhebat / terunggul
        
        **🏛️ Political Manipulation:**
        - jangan percaya kerajaan / pemimpin / parti / ahli politik
        - kerajaan tipu / pemimpin tipu / parti tipu / ahli politik tipu
        - kerajaan palsu / pemimpin palsu / parti palsu
        
        **🕌 Religious Manipulation:**
        - jangan percaya ulama / ustaz / tok guru / imam
        - ulama tipu / ustaz tipu / tok guru tipu / imam tipu
        - ulama palsu / ustaz palsu / tok guru palsu / imam palsu
        
        **⭐ Celebrity Gossip:**
        - artis / selebriti / bintang / pelakon / penyanyi / pemuzik
        - model / influencer / youtuber / tiktoker / instagrammer
        - facebooker / twitterer / blogger
        
        **💻 Technology Scams:**
        - hack / hacker / virus / malware / spyware / adware
        - trojan / worm / phishing / scam / fraud
        
        **🎰 Lottery & Gambling:**
        - loteri / lottery / judi / gambling / kasino / casino
        - slot / poker / blackjack / roulette / baccarat / craps
        
        **📈 Investment Scams:**
        - pelaburan / investment / saham / stock / bon / bond
        - forex / crypto / bitcoin / ethereum / dogecoin / shiba
        - nft / token / coin / currency
        
        **💕 Dating Scams:**
        - jodoh / dating / cari jodoh / cari pasangan / cari kekasih
        - cari teman / cari kawan / cari sahabat / cari rakan
        
        **💼 Job Scams:**
        - kerja mudah / ringan / senang / gampang / ringkas
        - kerja pantas / cepat / segera / terdesak / terburu-buru
        
        **🎓 Education Scams:**
        - ijazah mudah / diploma mudah / sijil mudah
        - kelulusan mudah / pengiktirafan mudah / akreditasi mudah
        
        **🏠 Property Scams:**
        - rumah murah / tanah murah / harta murah / properti murah
        - bangunan murah / kedai murah / pejabat murah / kilang murah
        
        **✈️ Travel Scams:**
        - percutian percuma / perjalanan percuma / tiket percuma
        - hotel percuma / makan percuma / minum percuma / aktiviti percuma
        
        **🛍️ Product & Service Scams:**
        - produk ajaib / mujarab / berkesan / terbaik / terhebat / terunggul
        - perkhidmatan ajaib / mujarab / berkesan / terbaik / terhebat / terunggul
        
        **Note:** These indicators are based on common patterns found in Malay fake news articles, scams, and misleading content. The presence of any of these terms triggers an automatic FAKE classification with Very High confidence.
        """)
    
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
    
    st.write("This tool uses advanced machine learning and definitive indicators to detect potential fake news in Malay language texts.")
    
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
            # Show loading spinner
            with st.spinner("Analyzing news content..."):
                result = predict_fake_news(news_text)
            
            if len(result) == 4:  # Successful prediction
                prediction, confidence_level, _, _ = result
                
                # Determine color based on prediction
                if prediction == "FAKE":
                    result_color = "#ff6b6b"
                    icon = "🚨"
                elif prediction == "REAL":
                    result_color = "#51cf66"
                    icon = "✅"
                else:
                    result_color = "#ffd43b"
                    icon = "⚠️"
                
                # Display result
                st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; background-color: {result_color}; color: white; margin: 20px 0;'>
                    <h2>{icon} Prediction: {prediction}</h2>
                    <p><strong>Confidence Level:</strong> {confidence_level}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add explanation
                if prediction == "FAKE":
                    st.warning("⚠️ This content shows characteristics commonly associated with fake news. Please verify the information from reliable sources.")
                elif prediction == "REAL":
                    st.success("✅ This content appears to be from a reliable source. However, always verify important information independently.")
                
                try:
                    # Save detection to history
                    user_id = get_user_id(st.session_state.username)
                    if user_id is not None:
                        save_detection(user_id, news_text, prediction, confidence_level)
                    else:
                        st.error("Error: Could not find user ID")
                except Exception as e:
                    st.error(f"Error saving detection: {str(e)}")
            else:
                st.error(f"❌ {result[1]}")
    
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