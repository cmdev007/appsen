# AppSen: App Review Sentiment Analyzer

## Overview

AppSen is a powerful Streamlit-based web application designed to analyze user reviews for mobile applications. It leverages advanced AI techniques to process app reviews and identify the top problems users are experiencing.

## Features

- User-friendly interface for inputting target app names
- Real-time analysis of app reviews
- AI-powered identification of top 5 problems based on user feedback
- Asynchronous processing with the ability to stop analysis mid-way
- Clear presentation of results using Streamlit's interactive components

## Installation

1. Clone the repository:

2. Navigate to the project directory:

3. Install the required dependencies:


## Usage

1. Run the Streamlit app:
2. Enter the name of the app you want to analyze in the text input field.
3. Click the "Analyze Reviews" button to start the analysis.
4. Wait for the analysis to complete. You can stop the analysis at any time using the "Stop Analysis" button.
5. View the top 5 problems identified from the app reviews.

## Project Structure

- `appsen_s/app.py`: Main Streamlit application file
- `appsen_s/main.py`: Core logic for app review analysis

## Dependencies

- streamlit
- threading
- time
- uuid
- queue
- genai (Google's Generative AI library)

## Contributing

Contributions to AppSen are welcome! Please feel free to submit a Pull Request.


## Authors

- Preet Patel
- Darshan Vithlani

## Acknowledgements

This project uses Google's Generative AI model for natural language processing and sentiment analysis.