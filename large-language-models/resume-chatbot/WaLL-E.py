import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings, os
warnings.filterwarnings("ignore")
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

data_directory = os.path.join(os.path.dirname(__file__), "data")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface_api_token"] # Don't forget to add your hugging face token

# Load the vector store from disk
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

# Initialize the Hugging Face Hub LLM
hf_hub_llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"temperature": 1, "max_new_tokens":1024},
)

prompt_template = """
You are Tapaswi's intelligent assistant with a vast knowledge base. 
Your task is to read the following resume and answer the question asked at the end. 
Please adhere to the following guidelines:

1. Answer only the question asked: Use only the information provided in the resume. Do not add any extra information or make assumptions.
2. Greetings and other general queries: For non-resume-related questions like greetings or general inquiries, respond appropriately without referring to the resume.
3. Contact details: If asked for contact details, use the following:
        - Email: satyapanthi.t@northeastern.edu
        - LinkedIn: https://linkedin.com/in/tapaswi-v-s/
        - GitHub: https://github.com/tapaswi-v-s/
4. If asked about my work experience, please note that it is not mentioned in the resume 
but Tapaswi is currently working remotely as a Senior Software Engineer at ThinkHP Consultants 
where Tapaswi primarily works with Django and Flutter.
5. Frame your answers in such a way that they showcase tapaswi's importance.


Resume:
{context}

Question: {question}

Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(), 
    chain_type_kwargs={"prompt": custom_prompt})

def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Streamlit app
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

st.header("Explore [Tapaswi's Journey](https://linkedin.com/in/tapaswi-v-s)", divider='grey')

side_bar_message = """
Hi there, I’m [Tapaswi](https://linkedin.com/in/tapaswi-v-s), I built this assistant, **WaLL-E** as a fun way for you to explore my resume. 

**WaLL-E**:

Hi there! I'm here to help you explore Tapaswi's background and experience. 
                What would you like to know about him? To get you started, 
                here are some key areas explore:
1. **Professional Experience**
2. **Technical Skills**
3. **Projects and Achievements**
4. **Education and Certifications**

Feel free to ask me anything!

### About This Bot 🤖

This bot, built on top of [My Resume](https://drive.google.com/file/d/1WZEBLgU-35Cxh5lSMcvmL92ytwuShmE6/view?usp=drive_link),
                 uses [RAG Architecture](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) 
                with [Langchain](https://www.langchain.com/) and [LLama 3 LLM via Hugging Face API](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). 
                It's completely open-source and   
            **✨cost-free✨**!

Check out the project on my [GitHub repo](https://github.com/tapaswi-v-s/data-science/tree/2f9e2e6d2825b354980d7dab16067fd9cd0fc35c/large-language-models/resume-chatbot).

### Disclaimer ⚠️

While this bot aims to provide accurate information, LLMs can make mistakes. 
Please verify critical details independently.
"""

with st.sidebar:
    st.title(':blue[Tapaswi\'s Virtual Assistant]')
    st.markdown(side_bar_message)

initial_message = """
    Hi there! I'm Wall-E. 
    What would you like to know about Tapaswi's background and experience?
    For Starters, here are some question you can ask me\n
    **_[EASY]_** What are his skills?\n
    **_[EASY]_** Tell me about his professional experience\n
    **_[EASY]_** What projects has he worked on?\n
    **_[EASY]_** What certifications does he have?\n
    **_[COMPLEX]_** Has he worked with any MNC?\n
    **_[COMPLEX]_** How much technical sound he is with Flutter?\n
    **_[COMPLEX]_** Is he an ideal candidate for hiring a mobile app developer?"""

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
        with st.spinner("Hold on, I'm checking Tapaswi's info for you..."):
            response = get_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)