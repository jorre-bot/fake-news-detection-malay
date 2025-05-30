import sqlite3
from tabulate import tabulate
from datetime import datetime

def init_db():
    conn = sqlite3.connect('fake_news.db')
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

def show_user_activity():
    conn = sqlite3.connect('fake_news.db')
    cursor = conn.cursor()
    
    print("\n=== USER REGISTRATION AND LOGIN ACTIVITY ===")
    print("\nUSERS TABLE:")
    cursor.execute('''
        SELECT id, username, email, 
        datetime(created_at, 'localtime') as registration_time 
        FROM users
    ''')
    users = cursor.fetchall()
    if users:
        print(tabulate(users, headers=['ID', 'Username', 'Email', 'Registration Time'], tablefmt='grid'))
    else:
        print("No users found in the database.")

    print("\n=== NEWS DETECTION ACTIVITY ===")
    print("\nDETECTIONS TABLE (Most Recent First):")
    cursor.execute('''
        SELECT 
            d.id,
            u.username,
            d.news_text,
            d.prediction,
            d.confidence,
            datetime(d.timestamp, 'localtime') as detection_time
        FROM detections d
        JOIN users u ON d.user_id = u.id
        ORDER BY d.timestamp DESC
    ''')
    detections = cursor.fetchall()
    if detections:
        print(tabulate(detections, 
                      headers=['ID', 'Username', 'News Text', 'Prediction', 'Confidence', 'Detection Time'], 
                      tablefmt='grid',
                      maxcolwidths=[5, 15, 50, 10, 10, 20]))
    else:
        print("No detections found in the database.")

    # Show statistics
    print("\n=== SYSTEM STATISTICS ===")
    
    # User statistics
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    print(f"\nTotal registered users: {total_users}")

    # Detection statistics
    cursor.execute('SELECT COUNT(*) FROM detections')
    total_detections = cursor.fetchone()[0]
    print(f"Total news detections: {total_detections}")

    # Prediction distribution
    cursor.execute('''
        SELECT prediction, COUNT(*) as count 
        FROM detections 
        GROUP BY prediction
    ''')
    prediction_stats = cursor.fetchall()
    if prediction_stats:
        print("\nPrediction Distribution:")
        print(tabulate(prediction_stats, headers=['Prediction', 'Count'], tablefmt='grid'))

    # Recent activity
    print("\n=== RECENT ACTIVITY (Last 24 Hours) ===")
    cursor.execute('''
        SELECT 
            u.username,
            d.prediction,
            datetime(d.timestamp, 'localtime') as activity_time
        FROM detections d
        JOIN users u ON d.user_id = u.id
        WHERE d.timestamp > datetime('now', '-1 day')
        ORDER BY d.timestamp DESC
        LIMIT 5
    ''')
    recent = cursor.fetchall()
    if recent:
        print(tabulate(recent, headers=['Username', 'Prediction', 'Time'], tablefmt='grid'))
    else:
        print("No recent activity in the last 24 hours.")

    conn.close()

if __name__ == "__main__":
    init_db()
    show_user_activity() 