# AWS EC2 FAQ Chatbot

**Note: This chatbot is fed only FAQs from the AWS EC2 FAQ page: [https://aws.amazon.com/ec2/faqs/](https://aws.amazon.com/ec2/faqs/)**

## Live Demo

Try out the live version of the chatbot [here](https://aws-faq.streamlit.app/).

## Overview

This project builds a chatbot that answers questions based on the FAQs from the AWS EC2 FAQ page. The chatbot uses a combination of web scraping, text embedding generation, and a Streamlit app to provide answers to user queries.

You can try out the live version of the chatbot [here](https://aws-faq.streamlit.app/).
## Instructions

### Step 1: Scrape the FAQ Data

To scrape the FAQ data from the web page, run the `scraper.py` file. This script will scrape the data from the AWS EC2 FAQ page and create a CSV file named `aws_faqs.csv`.

```bash
python scraper.py
```

### Step 2: Generate Embeddings

Run the `embeddings-generator.py` file to generate the embeddings. This will create a `data` directory in the project directory. The `data` folder will contain the ChromaDB vector store.

```bash
python embeddings-generator.py
```
### Step 3: Start the Streamlit App

Finally, run the following command to start the Streamlit app of the AWS FAQ chatbot.

```bash
streamlit run aws-chatbot.py
```

## Important Note ⚠️
**Before running `embeddings-generator.py` and `aws-chatbot.py`, include your Hugging Face token in both scripts. The scripts will not run without it.**

## Implementation Details

### scraper.py

- **Functionality**: Scrapes the EC2 FAQs from the AWS website.
- **Libraries Used**: `BeautifulSoup` for web scraping, `Pandas` for creating a CSV file.
- **Output**: Creates a CSV file `aws_faqs.csv` with the scraped FAQ data.

### embeddings-generator.py

- Functionality: Generates embeddings from the scraped FAQ data.
- Process:
        - Loads the CSV file using `CSV Loader` from `langchain`.
        - Splits the text using `RecursiveCharacterTextSplitter` from `langchain`.
        - Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2` model from HuggingFace.
        - Stores the embeddings into Chroma Vector Database, saved locally in a `data` directory.

### aws-chatbot.py

- **Functionality**: Provides a Streamlit-based chatbot interface to answer user questions.
- **Process**:
    - Initializes ChromaDB with the local database.
    - Searches for similar questions in ChromaDB when a user asks a question, using `similarity_score_threshold` method with a threshold set to `0.5`.
    - Provides the similar question and its answer to the LLM.
    - Uses `mistralai/Mixtral-8x7B-Instruct-v0.1` LLM from HuggingFace to generate responses.

## Project Structure

```graphql
├── data/                     # Directory containing ChromaDB vector store
├── aws_faqs.csv              # CSV file containing scraped FAQ data
├── scraper.py                # Script to scrape FAQ data from AWS EC2 FAQ page
├── embeddings-generator.py   # Script to generate embeddings from the FAQ data
├── aws-chatbot.py            # Streamlit app script for the chatbot
├── requirements.txt          # File listing required Python libraries
└── README.md                 # This README file
```

## Requirements

Install the required libraries using the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Running the Project

1. Scrape the FAQ Data: `python scraper.py` (Optional)
2. Generate Embeddings: `python embeddings-generator.py` (Optional)
3. Start the Streamlit App: `streamlit run aws-chatbot.py`