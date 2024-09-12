from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/v1/completions/")
def get_answer(question: QueryRequest):
    time.sleep(0.5)
    result = {'result': 'agi achieved internally!',
              'context': 'agi agi agi agi agi agi'}
    return {"response": result['result'], "context": result['context']}

# how to set up? uvicorn api:app --reload --port [port] 
# how to use? curl -X POST http://localhost:[port]/v1/completions/ -H "Content-Type: application/json" -d "{\"question\": \"What is AGI?\"}"