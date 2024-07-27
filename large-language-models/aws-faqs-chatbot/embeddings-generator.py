import pandas as pd
from langchain.docstore.document import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm.auto import tqdm
from langchain_community.vectorstores import Chroma
import os, warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
load_dotenv()

df = pd.read_csv('aws_faqs.csv')
df = df.dropna()

langchain_docs = [LangchainDocument(page_content=row[1]["answer"], 
                                    metadata={"faq": row[1]["question"]}) 
                  for row in tqdm(df.iterrows())]

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        encode_kwargs={"normalize_embeddings": True}  # set True to compute cosine similarity
)

vector_store = Chroma(embedding_function=embedding_model, persist_directory="embeddings")
vector_store.add_documents(documents=langchain_docs)

print("Embeddings generated and stored successfully.")