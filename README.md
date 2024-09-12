# KVADR_RAG
Документация по проекту создания RAG-пайплана от команды "БЕЗБАШЕННЫЕ КВАДРЫ"

В ноутбуке rag_pipeline_test.ipynb приведены параграфы с написанным методом пайплайна и примером его запуска для нормативно правовых актов. 

## Инструкция по запуску
1. Клонирование репозитория
```
git clone https://github.com/Roman210302/KVADR_RAG
```
2. Установка зависимостей
```
pip install -r requirements.txt
```
### Advanced Baseline

config = {
    'model_name': 'llama3.1-8b-q4',  # llama3.1-8b-q4 / gemma-2-9b-it-simpo-q4 / tlite-q4
    'embed_model_name_short': 'e5l', # e5l (multilingual-e5-large) / ubgem3 (deepvk/USER-bge-m3)
    'chunk_size': 512, # либо 1024/256, 512/128, 256/64 или ?2048/256?
    'chunk_overlap': 128,
    'llm_framework': 'VLLM', # VLLM, LLamaCpp, Ollama
    'vectorstore_name': 'MILVUS', # база данных MILVUS / FAISS
    'retriever_name': 'vectorstore', # ensemble,
    'ensemble_retrievers_names': ['BM25', 'vectorstore'], # применяется только если retriever_name=ensemble
    'ensemble_retrievers_weights': [0.4, 0.6], # применяется только если retriever_name=ensemble
    'compressor_name': 'chunks_feature',
    'reranker_name': '' # to be updated
    'chain_type': 'stuff',
    
}
