# Fake News Detector with AI

A powerful AI-powered fake news detection system that combines multiple verification methods to determine the authenticity of news articles.

## Features

- AI-powered news authenticity analysis using OpenAI
- Similar article search using TF-IDF vectorization
- Google Custom Search integration for verification
- User-friendly Streamlit web interface
- Large dataset of verified news articles

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Abhich05/Fake-News-Detector-GEN-AI.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY='your_openai_api_key'
GOOGLE_API_KEY='your_google_api_key'
CSE_ID='your_cse_id'
```

4. Run the application:
```bash
streamlit run fake_news.py
```

## Usage

1. Open the application in your web browser (http://localhost:8501)
2. Enter a news article in the text input
3. Click "Analyze" to get:
   - AI-generated authenticity score
   - Similar articles from the dataset
   - Google search results for verification

## Technologies Used

- Python
- OpenAI API
- Google Custom Search API
- Streamlit
- scikit-learn
- pandas

## Dataset

The project uses a dataset of news articles with verified truthfulness ratings. The dataset is included in the repository as `news_data.csv`.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
