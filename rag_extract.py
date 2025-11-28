import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Connect to Chroma Cloud
client = chromadb.CloudClient(
    api_key="ck-Bz6vWE8LH4mrmKx2dfreGSzgiikFBKhm7BHNHivkDvvw",
    tenant="6db89e03-6466-4af4-ad1c-d8237a75efa7"
)

# 2. Use HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Connect vectorstore
vectorstore = Chroma(
    client=client,
    collection_name="reddit_posts",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Define prompt
prompt = PromptTemplate.from_template("""
Use the following context to answer the question.

Context:Places ,restuarants, hidden gems,things to do

Question:
Restuarants to go, 
Answer:
""")

# 5. Local LLM Pipeline (runs on your computer)
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # Smaller, faster model
    max_length=512
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 6. Build RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 7. Query
query = "Tell me about hiddne gems in  Philadelphia"
response = rag_chain.invoke(query)

print("\nAnswer:", response)