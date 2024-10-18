import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from review_fetcher import fetch_app_reviews
import google.generativeai as genai
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt_tab', quiet=True)

sia = SentimentIntensityAnalyzer()

def get_top_negative_sentences(data, num_sentences=200, threshold=-0.5):
    sentences = nltk.sent_tokenize(data)
    scored_sentences = [(sentence, sia.polarity_scores(sentence)) for sentence in sentences]
    negative_sentences = [
        (sentence, scores) for sentence, scores in scored_sentences 
        if scores['compound'] < threshold and scores['neg'] > 2 * scores['pos']
    ]
    sorted_sentences = sorted(negative_sentences, key=lambda x: x[1]['compound'])
    top_negatives = [sentence for sentence, _ in sorted_sentences[:num_sentences]]
    return top_negatives

def analyze_with_gemini(negative_reviews, gemini_api):
    # Configure the Gemini API
    genai.configure(api_key=gemini_api)

    # Set up the model
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
                Please analyze the following negative app reviews and provide:

                1. A concise summary of the main issues users are experiencing.

                2. A numbered list of the top 5 problems mentioned, ordered by their significance.

                Negative Reviews:
                {negative_reviews}

                Your response should be formatted as:

                Summary:
                [Your summary here]

                Top 5 Problems:
                1. Problem 1
                2. Problem 2
                3. Problem 3
                4. Problem 4
                5. Problem 5
                """
    response = model.generate_content(prompt)
    return response.text


def analyze_app_reviews(app_name, stop_analysis, gemini_api):
    logging.info(f"Analyzing reviews for app: {app_name}")

    logging.info("Fetching reviews")
    reviews = fetch_app_reviews(app_name)
    if stop_analysis.is_set():
        return None, None
    logging.info(f"Fetched {len(reviews.split())} words of reviews")

    logging.info("Extracting top negative sentences")
    top_negative_sentences = get_top_negative_sentences(reviews)
    if stop_analysis.is_set():
        return None, None
    logging.info(f"Extracted {len(top_negative_sentences)} negative sentences")

    negative_text = " ".join(top_negative_sentences)

    logging.info("Analyzing negative_text with Gemini model")
    top_problems = analyze_with_gemini(negative_text, gemini_api)
    
    return negative_text, top_problems