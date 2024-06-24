# WaLL-E: Resume Q&A Chatbot

## What This Chatbot Is About

WaLL-E is an intelligent Resume Q&A chatbot designed to answer queries about my professional background directly from 
[my resume](https://drive.google.com/file/d/1WZEBLgU-35Cxh5lSMcvmL92ytwuShmE6/view?usp=drive_link).
It can handle a wide range of questions, from technical skills to project experience, providing insights and details in a conversational manner.

## How It Is Built

**WaLL-E** is built using the following technologies:

- **LangChain**: Used for chaining the LLM with the vector store.
- **RecursiveTextSplitter**: Splits the PDF into smaller chunks for efficient processing.
- **Hugging Face's sentence-transformers/all-MiniLM-L6-v2 Model**: Generates embeddings from the text chunks.
- **Chroma Vector Database**: Stores and queries embeddings.
- **LLAMA 3 LLM with HuggingFace Inference API**: Powers the language model interactions.
- **Persona Pattern Technique**: Used for crafting prompts to the LLM.
- **Streamlit**: Hosts the application and provides an interactive user interface.

## Instructions on How to Setup and Run

### Step 1: Install Required Python Libraries

First, install the required Python libraries mentioned in the `requirements.txt` file by running the following command:

```bash
pip install -r requirements.txt
```
### Step 2: Generate Embeddings
There are two Python files: `embeddings_generator.py` and `WaLL-E.py`.

1. **Generate HuggingFace API Token**: Obtain a HuggingFace API Token and place it in both Python files.
2. **Run Embeddings Generator**: Run `embeddings_generator.py` to generate the embeddings of the provided PDF file in the **"data"** directory.
```bash 
python embeddings_generator.py
```

### Step 3: Start the WaLL-E Chatbot
Once the embeddings are generated, you can start the local Streamlit app by running the following command:
```bash
streamlit run WaLL-E.py
```

### What Kind of Diverse Questions You Can Ask?
WaLL-E can handle a variety of questions, such as:

- Questions about Tapaswi's work experience
- Questions about his technical expertise with mobile app development
- Questions about the projects he has worked on
- Questions about his skills

### Hosted Web App

You can also interact with WaLL-E online at the following link:
[WaLL-E Chatbot](https://tapaswi.streamlit.app)