from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
import logging
from review_fetcher import fetch_app_reviews
from llama_cpp import Llama
import time
import google.generativeai as genai
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt_tab',quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(f"Using device: {device}")

# Initialize the summarization pipeline with the device argument
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def get_top_negative_sentences(data, num_sentences=200, threshold=-0.5):
    sentences = nltk.sent_tokenize(data)
    scored_sentences = [(sentence, sia.polarity_scores(sentence)) for sentence in sentences]
    negative_sentences = [
        (sentence, scores) for sentence, scores in scored_sentences 
        if scores['compound'] < threshold and scores['neg'] > 2 * scores['pos']
    ]
    sorted_sentences = sorted(negative_sentences, key=lambda x: x[1]['compound'])
    return [sentence for sentence, _ in sorted_sentences[:num_sentences]]

from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
import nltk
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio() > 0.7

def unique_sentences(text):
    sentences = sent_tokenize(text)
    unique = []
    for sentence in sentences:
        if not any(similar(sentence.lower(), s.lower()) for s in unique):
            unique.append(sentence)
    return unique


def summarize_text(text, stop_analysis, max_length=512, batch_size=32):
    start_time = time.time()
    
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    dataset = TextDataset(chunks, summarizer.tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    summaries = []
    for batch in dataloader:
        if stop_analysis.is_set():
            return None
        batch = {k: v.squeeze(1).to(device) for k, v in batch.items()}
        summary_ids = summarizer.model.generate(batch['input_ids'], max_length=150, min_length=50, do_sample=False)
        batch_summaries = summarizer.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        summaries.extend(batch_summaries)

    summary = ' '.join(summaries)
    unique_summary = ' '.join(unique_sentences(summary))
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    logging.info(f"Summarization completed in {execution_time:.2f} seconds")
    
    return unique_summary

def analyze_with_llama(summary):
    # Initialize Llama model
    llm = Llama(model_path="/mnt/sharedssd/preet/llama.cpp/models/Meta-Llama-3-7B-29Layers.Q4_K_M.gguf", n_ctx=2048, n_gpu_layers=-1)

    # Prepare the prompt
    prompt = f"""Based on the following summary of negative reviews, identify and list the top 5 problems:

    Summary: {summary}

    Top 5 problems:
    1. """

    # Generate response
    response = llm(prompt, max_tokens=500, stop=["6.", "\n\n"], echo=True)

    # Extract and return the generated problems
    f = open("temp.txt", "w")
    f.write(str(response))
    f.close()
    generated_text = response['choices'][0]['text']
    if not generated_text.strip():
        return "No problems identified"
    top_problems = generated_text.split("1. ")[1:]
    return top_problems


def analyze_with_gemini(summary):
    # Configure the Gemini API
    genai.configure(api_key='AIzaSyA9gOLakGAP3cp0hV0EzYe_FZQ-r3dz6Y4')

    # Set up the model
    model = genai.GenerativeModel('gemini-pro')

    # Prepare the prompt
    prompt = f"""Based on the following summary of negative reviews, identify and list the top 5 problems:

    Summary: {summary}

    Top 5 problems:
    1. """

    # Generate response
    response = model.generate_content(prompt)

    # Extract and return the generated problems
    generated_text = response.text
    if not generated_text.strip():
        return "No problems identified"
    top_problems = "1. "+"\n".join(generated_text.split("1. ")[1:])
    return top_problems


def analyze_app_reviews(app_name, stop_analysis):
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
    logging.info("Generating summary of negative reviews")
    summary = summarize_text(negative_text, stop_analysis)
    if stop_analysis.is_set():
        return None, None

    logging.info("Review analysis complete")

    logging.info("Analyzing summary with Gemini model")
    top_problems = analyze_with_gemini(summary)
    
    return summary, top_problems

