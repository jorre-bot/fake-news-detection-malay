import streamlit as st
import os
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import yaml

# Admin credentials configuration
ADMIN_CONFIG_FILE = 'admin_config.yml'

def init_admin_config():
    if not os.path.exists(ADMIN_CONFIG_FILE):
        # Create default admin credentials (you should change these)
        admin_config = {
            'username': 'admin',
            # Default password: "admin123"
            'password_hash': 'c7ad44cbad762a5da0a452f9e854fdc1e0e7a52a38015f23f3eab1d80b931dd472634dfac71cd34ebc35d16ab7fb8a90c81f975113d6c7538dc69dd8de9077ec'
        }
        with open(ADMIN_CONFIG_FILE, 'w') as f:
            yaml.dump(admin_config, f)

def hash_password(password):
    return hashlib.sha512(password.encode()).hexdigest()

def check_admin_auth(username, password):
    try:
        with open(ADMIN_CONFIG_FILE, 'r') as f:
            admin_config = yaml.safe_load(f)
            return (username == admin_config['username'] and 
                   hash_password(password) == admin_config['password_hash'])
    except:
        return False

def show_login():
    st.title("Log Viewer - Admin Login")
    
    with st.form("admin_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if check_admin_auth(username, password):
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Invalid admin credentials")

def parse_log_line(line):
    try:
        # Split the line into timestamp and message
        timestamp_str = line[:23]  # Extract timestamp
        level = line[26:31].strip()  # Extract log level
        message = line[33:].strip()  # Extract message
        
        # Parse timestamp
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
        
        return {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
    except:
        return None

def load_logs():
    log_file = os.path.join('logs', 'app.log')
    if not os.path.exists(log_file):
        return pd.DataFrame()
    
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                logs.append(parsed)
    
    return pd.DataFrame(logs)

def show_log_viewer():
    st.title("Malay Fake News Detection - Log Viewer")
    
    # Add logout button
    if st.sidebar.button("Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    # Load logs
    df = load_logs()
    
    if df.empty:
        st.warning("No logs found.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Time range filter
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last Hour", "Last 24 Hours", "Last 7 Days", "All Time"]
    )
    
    # Log level filter
    available_levels = df['level'].unique()
    selected_levels = st.sidebar.multiselect(
        "Log Levels",
        available_levels,
        default=available_levels
    )
    
    # Search filter
    search_term = st.sidebar.text_input("Search Messages")
    
    # Apply filters
    now = datetime.now()
    if time_range == "Last Hour":
        df = df[df['timestamp'] > now - timedelta(hours=1)]
    elif time_range == "Last 24 Hours":
        df = df[df['timestamp'] > now - timedelta(days=1)]
    elif time_range == "Last 7 Days":
        df = df[df['timestamp'] > now - timedelta(days=7)]
    
    df = df[df['level'].isin(selected_levels)]
    
    if search_term:
        df = df[df['message'].str.contains(search_term, case=False)]
    
    # Display statistics
    st.header("Log Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Logs", len(df))
    
    with col2:
        error_count = len(df[df['level'] == 'ERROR'])
        st.metric("Errors", error_count)
    
    with col3:
        warning_count = len(df[df['level'] == 'WARNING'])
        st.metric("Warnings", warning_count)
    
    # Display logs
    st.header("Log Entries")
    
    # Sort by timestamp (most recent first)
    df = df.sort_values('timestamp', ascending=False)
    
    # Display as a table
    st.dataframe(
        df,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp"),
            "level": st.column_config.TextColumn("Level"),
            "message": st.column_config.TextColumn("Message", width="large"),
        }
    )
    
    # Download logs
    if st.button("Download Filtered Logs"):
        csv = df.to_csv(index=False)
        st.download_button(
            "Click to Download",
            csv,
            "filtered_logs.csv",
            "text/csv",
            key='download-csv'
        )

def main():
    # Initialize admin configuration
    init_admin_config()
    
    # Initialize session state
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    # Show appropriate page based on authentication
    if not st.session_state.admin_authenticated:
        show_login()
    else:
        show_log_viewer()

if __name__ == "__main__":
    main() 