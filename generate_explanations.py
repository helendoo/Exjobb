import pandas as pd
import os
from langchain_groq import ChatGroq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, Document, VectorStoreIndex

# === Step 1: Setup ===
os.environ["GROQ_API_KEY"] = "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"  # Replace if needed

llm = ChatGroq(model="llama-3.3-70b-versatile", verbose=True)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Step 2: Load Anomaly Results ===
mwd_full = pd.read_csv("MWD_Anomaly_Results.csv")

mwd_columns = [
    'AvePenetrRate', 'AvePercPressure', 'AveFeedPressure',
    'AveDampPressure', 'AveRotPressure', 'AveFlushPressure'
]

normal_avg = mwd_full[mwd_full['anomaly_label'] == 0][mwd_columns].mean()
anomalies = mwd_full[mwd_full['anomaly_label'] == 1].head()

# === Step 3: Generate Explanations with LLM ===
explanations = []

for _, row in anomalies.iterrows():
    hole_id = int(row['Hole ID'])
    start_time = row['Start Hole Time']

    prompt = f"Anomaly detected in Hole ID {hole_id} (Start Time: {start_time}):\n\n"
    for col in mwd_columns:
        value = row[col]
        avg = normal_avg[col]
        pct_diff = ((value - avg) / avg * 100) if avg != 0 else 0
        prompt += f"- {col}: {value:.2f} (normal: {avg:.2f}, deviation: {pct_diff:+.1f}%)\n"

    prompt += "\nExplain what this might indicate in a machine diagnostic context."

    response = llm.invoke(prompt)
    explanation = f"Hole ID {hole_id} Explanation:\n{response.content}"
    explanations.append(explanation)

    print(f"\nHole ID {hole_id} Explanation:\n{response.content}\n")

# === Step 4: Save as text file (for human-readable logging) ===
with open("anomaly_explanations.txt", "w", encoding="utf-8") as f:
    for explanation in explanations:
        f.write(explanation + "\n\n---\n\n")

# === Step 5: Save as vector index (for chatbot retrieval) ===
docs = [Document(text=exp) for exp in explanations]
index = VectorStoreIndex.from_documents(docs)
index.storage_context.persist(persist_dir="anomaly_index")


df_errors = pd.read_csv("errors.csv")
error_docs = [Document(text=row.to_string()) for _, row in df_errors.iterrows()]

all_docs = error_docs + [Document(text=exp) for exp in explanations]

index = VectorStoreIndex.from_documents(all_docs)
index.storage_context.persist(persist_dir="combined_index")
