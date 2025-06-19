from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import re
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fake_news.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Delete existing database file if it exists
if os.path.exists('fake_news.db'):
    os.remove('fake_news.db')

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the trained model
model = joblib.load('best_fake_news_model.pkl')

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    detections = db.relationship('Detection', backref='user', lazy=True)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    news_text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def is_valid_password(password):
    # At least 8 characters, one uppercase, one lowercase, one number
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    return True

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# Routes
@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
            
        if not is_valid_email(email):
            flash('Invalid email format')
            return redirect(url_for('register'))
            
        if not is_valid_password(password):
            flash('Password must be at least 8 characters long and contain uppercase, lowercase, and numbers')
            return redirect(url_for('register'))
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get the news text from the request
        news_text = request.form['news_text']

        # List of keywords that might indicate fake news
        fake_indicators = [
            'viral', 'kononnya', 'didakwa', 'tidak rasmi', 'sumber tidak rasmi',
            'dikatakan', 'tular', 'dakwaan', 'khabar angin',
            'palsu', 'fitnah', 'hoax', 'berita palsu', 'maklumat palsu',
            'penipuan', 'tidak benar', 'tidak sahih', 'tidak berasas',
            'sebaran palsu', 'berita tidak benar', 'berita tidak sahih',
            'berita tidak rasmi', 'gosip', 'cerita palsu', 'cerita rekaan',
            'sensasional', 'panik', 'menipu', 'tipu', 'bukan rasmi',
            'belum disahkan', 'belum rasmi', 'belum ada pengesahan',
            'belum dapat dipastikan', 'belum ada kenyataan rasmi',
            'belum ada pengumuman rasmi', 'berita tular', 'berita fitnah',
            'berita rekaan', 'berita bohong', 'berita tipu', 'berita panik',
            'berita sensasi', 'berita tidak tepat', 'berita tidak betul',
            'berita tidak lengkap', 'berita tidak logik', 'berita mengelirukan',
            'maklumat tidak benar', 'maklumat tidak sahih', 'maklumat tidak tepat',
            'maklumat tidak rasmi', 'maklumat tidak logik', 'maklumat mengelirukan',
            'mesej palsu', 'mesej tular', 'mesej tidak benar', 'mesej tidak sahih',
            'mesej tidak rasmi', 'mesej tidak tepat', 'mesej tidak logik',
            'mesej mengelirukan', 'khabar tidak benar', 'khabar tidak sahih',
            'khabar tidak rasmi', 'khabar tidak tepat', 'khabar tidak logik',
            'khabar mengelirukan', 'cerita tidak benar', 'cerita tidak sahih',
            'cerita tidak rasmi', 'cerita tidak tepat', 'cerita tidak logik',
            'cerita mengelirukan', 'amaran palsu', 'amaran tular', 'amaran tidak benar',
            'amaran tidak sahih', 'amaran tidak rasmi', 'amaran tidak tepat',
            'amaran tidak logik', 'amaran mengelirukan'
        ]
        # Convert text to lowercase for case-insensitive matching
        text_lower = news_text.lower()
        # If any fake indicator word is present, set as FAKE
        if any(indicator in text_lower for indicator in fake_indicators):
            result = "FAKE"
        else:
            # Make prediction
            prediction = model.predict([news_text])[0]
            # Convert prediction to human-readable format
            result = "FAKE" if prediction == 0 else "REAL"

        # Save the detection to database
        detection = Detection(
            user_id=current_user.id,
            news_text=news_text,
            prediction=result
        )
        db.session.add(detection)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'prediction': result,
            'confidence': 'High'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/history')
@login_required
def history():
    detections = Detection.query.filter_by(user_id=current_user.id).order_by(Detection.timestamp.desc()).all()
    return render_template('history.html', detections=detections)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 