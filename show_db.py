import sqlite3
from tabulate import tabulate

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

def show_database_contents():
    conn = sqlite3.connect('fake_news.db')
    cursor = conn.cursor()
    
    # Get users table contents
    print("\nUSERS TABLE:")
    cursor.execute('SELECT id, username, email FROM users')  # Excluding password for security
    users = cursor.fetchall()
    if users:
        print(tabulate(users, headers=['ID', 'Username', 'Email'], tablefmt='grid'))
    else:
        print("No users found in the database.")
    
    # Get detections table contents
    print("\nDETECTIONS TABLE:")
    cursor.execute('SELECT id, user_id, news_text, prediction, confidence, timestamp FROM detections')
    detections = cursor.fetchall()
    if detections:
        print(tabulate(detections, headers=['ID', 'User ID', 'News Text', 'Prediction', 'Confidence', 'Timestamp'], tablefmt='grid'))
    else:
        print("No detections found in the database.")
    
    conn.close()

if __name__ == "__main__":
    init_db()
    show_database_contents() 