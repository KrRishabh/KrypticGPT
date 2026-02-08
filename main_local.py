from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# 1. Load the local DB and local Embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_db = Chroma(persist_directory="./local_tweet_db", embedding_function=embeddings)

# 2. Create a retriever from the vector DB
retriever = vector_db.as_retriever()

# 3. Initialize local LLM (Qwen3 4B)
llm = ChatOllama(model="qwen3:4b", temperature=0)

# 4. Define the prompt template
system_prompt = (
    "You are an assistant that answers questions based on the user's tweets. "
    "Use the following retrieved tweets as context to answer the question. "
    "If you don't know the answer, say so. Keep your answer concise.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])


class QueryRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask_local_bot(request: QueryRequest):
    # Step A: Retrieve relevant tweet chunks from the vector DB
    print(f"Retrieving relevant tweet chunks from the vector DB for query: {request.query}")
    docs = retriever.invoke(request.query)

    # Step B: Combine the retrieved chunks into a single context string
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step C: Build the prompt with context + user question
    messages = prompt.invoke({"context": context, "question": request.query})

    # Step D: Send the prompt to the LLM and get a response
    response = llm.invoke(messages)

    return {"answer": response.content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)