from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from basicRCA import rootAnalysis
import pandas as pd
import os

#langGraph Framework + groq + Ollama


# LLM 
os.environ["GROQ_API_KEY"] = "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"  
llm = ChatGroq(model="llama-3.3-70b-versatile", verbose=True)

# Embedding 
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

csvDF = pd.read_csv("C:/Users/wazzu/LLM/epirocData/delays.csv")
csvDocuments = [Document(text=row.to_string()) for _, row in csvDF.iterrows()]
csvIndex = VectorStoreIndex.from_documents(csvDocuments)

csvChat_engine = csvIndex.as_chat_engine(
    chat_mode="context",
    llm=llm)


with open("C:/Users/wazzu/LLM/data/expertInterviewTest.txt", "r", encoding="utf-8") as f:
    expertInterviews = f.read()
expertDocuments = [Document(text=expertInterviews)]
expertIndex = VectorStoreIndex.from_documents(expertDocuments)

expertChat_engine = expertIndex.as_chat_engine(
    chat_mode="context",
    llm=llm)


with open("C:/Users/wazzu/LLM/data/operatorInterview.txt", "r", encoding="utf-8") as f:
    operatorInterviews = f.read()
operatorDocuments = [Document(text=operatorInterviews)]
operatorIndex = VectorStoreIndex.from_documents(operatorDocuments)

operatorChat_engine = operatorIndex.as_chat_engine(
    chat_mode="context",
    llm=llm)


# State Schema 
class ChatState(TypedDict):
    user_input: Optional[str]
    response: Optional[str]
    root_cause_result: Optional[str]
    end: Optional[bool]

def get_csv_info(query: str) -> str:
    response = csvChat_engine.chat(query)
    return response.response

def get_expert_info(query: str) -> str:
    response = expertChat_engine.chat(query)
    return response.response   

def get_operator_info(query: str) -> str:
    response = operatorChat_engine.chat(query)
    return response.response 
# Graph Nodes 

def root_cause_node(state: ChatState) -> ChatState:
    root_cause_data = rootAnalysis("C:/Users/wazzu/LLM/data/errors.csv")
    return {"root_cause_result": root_cause_data}
   
def ask_node(state: ChatState) -> ChatState:
    user_input = input("You: ")
    if user_input.lower() == "q":
        return {"user_input": None, "end": True}
    return {"user_input": user_input, "end": False}

def chat_node(state: ChatState) -> ChatState:
    if state["user_input"] is None:
        return state
    
    csvInfo = get_csv_info(state["user_input"])
    expertInfo = get_expert_info(state["user_input"])
    operatorInfo = get_operator_info(state["user_input"])
    rootCauseSummary = state.get("root_cause_analysis", "No root cause analysis available.")

    finalPrompt = f"""

Error Data:
{csvInfo}

Expert Knowledge:
{expertInfo}

Operator Knowledge:
{operatorInfo}

Root Cause Analysis:
{rootCauseSummary}

User Query:
{state['user_input']}

Based on the Error data, expert advice, operator advice and root cause analysis, 
diagnose the problem and suggest next actions based on the given information. DONT give suggestions to irrelevant questions 
""" 
    response = llm.invoke(finalPrompt)   
    return {"response": response.content}


def respond_node(state: ChatState) -> ChatState:
    if state.get("end"): 
        return state
    if state.get("response"):
        print("Bot:", state["response"])
    return {}

# Build Graph
builder = StateGraph(ChatState)

builder.add_node("ask", ask_node)
builder.add_node("root_cause", root_cause_node)
builder.add_node("chat", chat_node)
builder.add_node("respond", respond_node)

builder.set_entry_point("ask")
builder.add_edge("ask", "chat")
builder.add_edge("root_cause", "chat")
builder.add_edge("chat", "respond")
builder.add_conditional_edges("respond", lambda s: "END" if s.get("end") else "ask", {"ask": "ask", "END": END})


graph = builder.compile()
graph.invoke({
    "user_input": None,
    "response": None,
    "root_cause_result": None,
    "end": False
})


