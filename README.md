# Malay Fake News Detection System

A secure web-based system for detecting fake news in the Malay language using advanced machine learning.

## Project Overview

This project implements a secure web application that allows users to:
- Register and authenticate securely
- Analyze Malay news text for authenticity using Random Forest ML model
- View their detection history with confidence scores
- Get detailed probability breakdowns for predictions

## 🎯 **Model Performance**

### **Best Model: Random Forest Classifier**
- **Accuracy**: 100% on test data
- **F1 Score**: 100%
- **Features**: TF-IDF vectorization (10,000 features)
- **Training Data**: Balanced dataset of Malay news articles
- **Performance**: Zero false positives/negatives on test set

### **Model Comparison Results**
| Model | Accuracy | F1 Score | Total Errors |
|-------|----------|----------|--------------|
| **Random Forest** | **1.000** | **1.000** | **0** |
| Linear SVC | 1.000 | 1.000 | 0 |
| Naive Bayes | 0.999 | 0.999 | 1 |
| XGBoost | 0.997 | 0.997 | 2 |
| Gradient Boosting | 0.996 | 0.996 | 3 |

## Security Features

### Authentication Security
- Secure password hashing with salt
- Email validation
- Strong password policy enforcement (12+ characters, special chars)
- Session management with timeout
- Login attempt rate limiting (3 attempts, 5-minute lockout)

### Data Security
- Input sanitization
- SQL injection prevention
- XSS protection
- Secure database operations
- Error handling and logging

### Application Security
- Protected routes
- Session timeout (1 hour)
- Secure file handling
- Safe model loading with caching
- Input validation

## Project Structure

### Core Files
- `streamlit_app.py`: Main Streamlit application with ML model integration
- `models/best_fake_news_model.pkl`: Best Random Forest model
- `models/tfidf_vectorizer.pkl`: TF-IDF vectorizer for text preprocessing
- `models/best_model_metadata.json`: Model metadata and performance info
- `fake_news.db`: SQLite database with security constraints

### Results and Documentation
- `results/`: Contains model analysis, confusion matrices, and performance reports
- `train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv`: Training datasets

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

## ML Model Features

### Text Preprocessing
- TF-IDF vectorization with 10,000 features
- Text cleaning and normalization
- Case-insensitive processing
- Special character removal

### Prediction Output
- Binary classification (Fake/Real)
- Confidence levels (Very High, High, Medium, Low)
- Probability scores for both classes
- Detailed explanations and recommendations

## Setup and Deployment

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run streamlit_app.py
```

3. Access the application at `http://localhost:8501`

## Usage Instructions

### News Text Requirements
1. **Language**: Must be in Malay
2. **Minimum length**: 25 words (not just the title)
3. **Content**: Include both title and full news content

### Example News Format
```
KOTA BHARU: Jabatan Pendidikan Negeri Kelantan mengumumkan penutupan sementara semua sekolah di daerah Kota Bharu bermula esok susulan peningkatan mendadak kes Covid-19 di kawasan tersebut. Pengarah Pendidikan Negeri, Dato' Ahmad bin Ibrahim berkata keputusan ini dibuat selepas berbincang dengan Jabatan Kesihatan Negeri dan pihak berkuasa tempatan...
```

## Security Best Practices

1. User Authentication:
- Use strong passwords (12+ characters with special chars)
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

## Model Training and Evaluation

The system uses a comprehensive dataset of Malay news articles with:
- Balanced fake and real news samples
- TF-IDF feature extraction
- Multiple ML model comparison
- Cross-validation and testing
- Performance metrics and confusion matrices

## Future Enhancements

Planned improvements:
1. Multi-factor authentication
2. Enhanced audit logging
3. Automated backup system
4. Advanced rate limiting
5. HTTPS enforcement
6. Model retraining pipeline
7. Real-time model performance monitoring

## Contributing

When contributing to this project, please:
1. Follow secure coding practices
2. Implement proper error handling
3. Use parameterized queries
4. Add appropriate input validation
5. Document security features
6. Test model performance thoroughly

## License

This project is licensed under the MIT License - see the LICENSE file for details. 