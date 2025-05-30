# Malay Fake News Detection System

A secure web-based system for detecting fake news in the Malay language using machine learning.

## Project Overview

This project implements a secure web application that allows users to:
- Register and authenticate securely
- Analyze Malay news text for authenticity
- View their detection history
- Compare different ML model performances

## Security Features

### Authentication Security
- Secure password hashing with salt
- Email validation
- Strong password policy enforcement
- Session management
- Login attempt rate limiting

### Data Security
- Input sanitization
- SQL injection prevention
- XSS protection
- Secure database operations
- Error handling and logging

### Application Security
- Protected routes
- Session timeout
- Secure file handling
- Safe model loading
- Input validation

## Project Structure

### Core Files
- `streamlit_app.py`: Main Streamlit application with security implementations
- `app.py`: Alternative Flask implementation with additional security features
- `show_db.py`: Secure database inspection tool
- `fake_news.db`: SQLite database with security constraints
- `best_fake_news_model.pkl`: ML model with secure loading

### Database Schema
```sql
Users Table:
- id (PRIMARY KEY)
- username (UNIQUE)
- email (UNIQUE)
- password (Hashed)

Detections Table:
- id (PRIMARY KEY)
- user_id (FOREIGN KEY)
- news_text
- prediction
- confidence
- timestamp
```

## Security Implementation Details

### Password Security
- Minimum 8 characters
- Requires uppercase, lowercase, numbers
- Secure hashing with salt
- Rate-limited login attempts

### Input Validation
- Email format validation
- News text format requirements
- Length restrictions
- Character sanitization

### Database Security
- Parameterized queries
- Foreign key constraints
- Unique constraints
- Secure connection handling

## Setup and Deployment

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run streamlit_app.py
```

## Security Best Practices

1. User Authentication:
- Use strong passwords
- Keep login credentials secure
- Log out after each session

2. Data Input:
- Follow the required news format
- Avoid submitting sensitive information
- Report any suspicious behavior

3. System Access:
- Use secure network connections
- Keep the system updated
- Monitor for unauthorized access

## Future Security Enhancements

Planned security improvements:
1. Multi-factor authentication
2. Enhanced audit logging
3. Automated backup system
4. Advanced rate limiting
5. HTTPS enforcement

## Contributing

When contributing to this project, please:
1. Follow secure coding practices
2. Implement proper error handling
3. Use parameterized queries
4. Add appropriate input validation
5. Document security features

## License

This project is licensed under the MIT License - see the LICENSE file for details. 