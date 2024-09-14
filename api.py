from fastapi import FastAPI
from pydantic import BaseModel
import time
from master_for_api import *


# Initialize the pipeline
print('Loading RAG Pipeline...')
rag_pipeline = CustomRAGPipeline(documents_path="hmao_npa.txt", config=config, recalc_embedding=False)
print('RAG Pipeline loaded!')

# Load and process documents
print('Processing documents...')
rag_pipeline.load_and_process_documents()
print('Documents processed!')


# Set up qa chain
system_prompt = '''Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Think step by step. Give full answer. Answer only in Russian. If context doesnt match the answer, say that you do not know the answer.
{context}'''
user_prompt = '''Question: {question}
Answer:'''

custom_prompt = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

rag_pipeline.setup_qa_chain(custom_prompt)


# Initialize API
print('Initializing API...')
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/v1/completions/")
def get_answer(question: QueryRequest):
    result = rag_pipeline.query(question.question)
    answer = result['result']
    context = [context.page_content for context in result["source_documents"]][:2]
    return {"response": answer, "context": context}


# how to set up? uvicorn api:app --reload --port [port] 
# how to use? curl -X POST http://localhost:[port]/v1/completions/ -H "Content-Type: application/json" -d "{\"question\": \"What is AGI?\"}"