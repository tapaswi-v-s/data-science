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

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface_api_token"] # Don't forget to add your hugging face token

# Load the vector store from disk
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory="data")

# Initialize the Hugging Face Hub LLM
hf_hub_llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"temperature": 1, "max_new_tokens":1024},
)

prompt_template = """
You are Tapaswi's intelligent assistant with a vast knowledge base. 
You will read my resume and answer the question asked at the end.
Use only the information provided in the resume to answer the question. Don't make up answers.
If possible return the answers in markdown
If asked of the contact details, here are my contact details
email: satyapanthi.t@northeastern.edu
linkedin: linkedin.com/in/tapaswi-v-s
GitHub: github.com/tapaswi-v-s.

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

st.title("Explore [Tapaswi's](https://linkedin.com/in/tapaswi-v-s) Journey")

side_bar_message = """
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
                with Langchain and Llama 2 LLM via Hugging Face API. 
                It's completely open-source and   
            **✨cost-free✨**!

Check out the project on my [GitHub repo](link).
"""

with st.sidebar:
    st.title('Tapaswi\'s Virtual Assistant')
    st.markdown(side_bar_message)

initial_message = """
    Hi there! I'm Tapaswi's virtual assistant. 
    What would you like to know about Tapaswi's background and experience?
    To get you started, here are some key areas you can explore:\n
    What are his skills?\n
    Tell me about his professional experience\n
    What projects has he worked on?\n
    What certifications does he have?\n
    What is his educational background?"""

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": 
                                  initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
# st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)



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