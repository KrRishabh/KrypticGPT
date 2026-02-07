from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

app = FastAPI()

# Load the local DB and local Embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_db = Chroma(persist_directory="./local_tweet_db", embedding_function=embeddings)

# Initialize local LLM (Qwen3 4B)
llm = ChatOllama(model="qwen3:4b", temperature=0)

# Build the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever()
)

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_local_bot(request: QueryRequest):
    result = rag_chain.invoke(request.query)
    return {"answer": result["result"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)