import google_play_scraper
import concurrent.futures
from google_play_scraper import Sort, reviews
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_app_reviews(app_name, num_reviews=500, max_workers=5):
    logging.info(f"Starting to fetch reviews for app: {app_name}")
    
    result = google_play_scraper.search(app_name)
    if not result:
        raise ValueError(f"No app found with name: {app_name}")
    
    app_id = result[0]['appId']
    logging.info(f"Found app ID: {app_id}")
    
    def fetch_batch(continuation_token=None):
        logging.info(f"Fetching batch of reviews. Continuation token: {continuation_token}")
        return reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=min(num_reviews, 100),
            continuation_token=continuation_token
        )

    review_texts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_token = {executor.submit(fetch_batch): None}
        while len(review_texts) < num_reviews:
            for future in concurrent.futures.as_completed(future_to_token):
                token = future_to_token[future]
                try:
                    result, token = future.result()
                    review_texts.extend([review['content'] for review in result if review['content'] is not None])
                    logging.info(f"Fetched {len(review_texts)} reviews so far")
                    if token and len(review_texts) < num_reviews:
                        future_to_token[executor.submit(fetch_batch, token)] = token
                except Exception as exc:
                    logging.error(f'Generated an exception: {exc}')

    logging.info(f"Finished fetching reviews. Total reviews fetched: {len(review_texts)}")
    return " ".join(review_texts[:num_reviews])