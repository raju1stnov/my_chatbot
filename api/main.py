from fastapi import FastAPI
from pydantic import BaseModel
from agents.orchestrator import KnowledgeOrchestrator

app = FastAPI()
orch = KnowledgeOrchestrator()

class Query(BaseModel):
    question: str

@app.post("/query")
async def handle_query(query: Query):
    return orch.process_query(query.question)