from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
import streamlit as st
import os, warnings
warnings.filterwarnings('ignore')
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

data_directory = os.path.join(os.path.dirname(__file__), "data")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface_api_token"] # Don't forget to add your hugging face token

llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.01, "max_new_tokens":1024},
)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

prompt_template = """
Given the following frequently asked question and its answer, 
generate a helpful response for the user based on these. 
While answering the question please adhere to the following guidelines:
1. Response should be completely based on the given FAQ and its answer don't add any extra knowledge or make any assumptions. 
2. Response should be clear and precise.
3. If the provided FAQ and answer is not related to AWS just respond that the given question is out of your knowledge base.
4. No pre-amble and post-amble is required, just answer the question.
FAQ and its answer:
{context}

User Question:{question}

Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 1, 'score_threshold': 0.5}), 
    chain_type_kwargs={"prompt": custom_prompt})


def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Remove whitespace from the top of the page and sidebar
st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=1, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )

st.header("Amazon EC2 FAQ chatbot", divider='grey')

side_bar_message = """
Hello! This chatbot is designed to answer your questions based on the AWS EC2 FAQs, 
which you can find [here](https://aws.amazon.com/ec2/faqs/).

**Note**: Currently, this chatbot is trained to answer questions solely based on the Amazon EC2 FAQs.
## What This Chatbot Does

- **Answers Questions**: Provides answers to your queries by searching through the AWS EC2 FAQs.
- **FAQ-Based Responses**: Ensures that responses are relevant and accurate by using pre-existing FAQ data.

**How It Is Built**

- **Web Scraping**: The FAQ data is scraped from the AWS EC2 FAQ page using a Python script.
- **Text Embeddings**: The scraped data is processed to generate text embeddings using a pre-trained model.
- **ChromaDB**: The embeddings are stored in a Chroma Vector Database for efficient search and retrieval.
- **LLM Integration**: When you ask a question, the chatbot searches for similar questions in the ChromaDB and uses a language model to generate a coherent response.

Enjoy using the AWS EC2 FAQ Chatbot!

For more details and to view the code, visit the [project repository](https://github.com/tapaswi-v-s/data-science/tree/aws-chatbot/large-language-models/aws-faqs-chatbot).
"""

with st.sidebar:
    st.title(':blue[Amazon EC2 chatbot]')
    st.markdown(side_bar_message)

initial_message = """
    Hi there!  What would you like to know about Amazon EC2?
    For Starters, here are some question you can ask me\n
    Can I use EC2 with S3?\n
    What occurs to my data in the event that a system fails?\n"""

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": 
                                  initial_message}]
    
# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, Let me check..."):
            response = get_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)