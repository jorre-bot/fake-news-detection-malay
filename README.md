# Malay Fake News Detection

A web application that uses machine learning to detect fake news in Malay language texts. The application provides user authentication, history tracking, and uses state-of-the-art machine learning models for prediction.

## Features

- User authentication (login/register)
- Email validation
- Strong password requirements
- History tracking of news detections
- Machine learning-based fake news detection
- Beautiful and responsive UI

## Setup Instructions

1. Clone the repository:
```bash
git clone <your-repository-url>
cd fake-news-detection-malay
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Model Information

The fake news detection model is trained on Malay language news articles and uses advanced NLP techniques. The model achieves high accuracy in distinguishing between real and fake news articles.

## Usage

1. Register an account with a valid email
2. Login to the system
3. Enter or paste Malay news text
4. Click "Analyze" to get the prediction
5. View your detection history in the History page

## Requirements

- Python 3.7+
- See requirements.txt for full list of dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 