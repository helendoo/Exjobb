from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_groq import ChatGroq
import pandas as pd
import os

#Ollama Framework + Groq + HuggingFace

os.environ["GROQ_API_KEY"] = "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"  

# Groq LLM via LangChain
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    verbose=True,
)

# HuggingFace for local embedding 
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

#Load CSV data
df = pd.read_csv("errors.csv")
documents = [Document(text=row.to_string()) for _, row in df.iterrows()]

#Build index
index = VectorStoreIndex.from_documents(documents)

#conversational agent
chat_engine = index.as_chat_engine(chat_mode="context", llm=llm)


while True:
    user_input = input("You: ")
    if user_input.lower() == "q":
        break
    response = chat_engine.chat(user_input)
    print("Bot:", response.response)
