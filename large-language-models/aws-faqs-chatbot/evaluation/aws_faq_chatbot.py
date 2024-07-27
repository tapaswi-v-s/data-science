from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceEndpoint
from typing import Tuple, List
from langchain.docstore.document import Document
import os, warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
load_dotenv()

class AwsFaqChatBot():
    current_script_dir = os.path.dirname(__file__)
    aws_faqs_chatbot_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
    data_directory = os.path.join(aws_faqs_chatbot_dir, 'embeddings')

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_new_tokens=1024,
        temperature=0.1,
        model_kwargs={'token': os.getenv("HUGGINGFACEHUB_API_TOKEN")}
    )

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

    RAG_PROMPT_TEMPLATE = """
    <|system|>
    Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    While answering the question please adhere to the following guidelines:
    1. Response should be completely based on the provided context don't add any extra knowledge or make any assumptions.
    2. Response should be clear and precise.
    3. If the context is not provided just respond that the given question is out of your knowledge base.
    4. No pre-amble and post-amble is required, just answer the question.
    </s>
    <|user|>
    Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}
    </s>
    <|assistant|>
    """

    def similarity_search(self, question):
    
        relevant_docs = self.vector_store.similarity_search(query=question, k=3)
        relevant_docs = [doc.page_content for doc in relevant_docs]

        context = ""
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
        
        return context

    def ask(self, question, k=1) -> Tuple[str, List[Document]]:
        relevant_docs = self.vector_store.similarity_search(query=question, k=k)
        relevant_docs = [doc.page_content for doc in relevant_docs]

        context = ""
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

        final_prompt = self.RAG_PROMPT_TEMPLATE.format(question=question, context=context)
        
        answer = self.llm.invoke(final_prompt)
        return (answer, relevant_docs)
