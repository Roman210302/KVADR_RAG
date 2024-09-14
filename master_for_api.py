import os
from typing import List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_milvus import Milvus
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams
from langchain_community.llms import VLLM
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_utilization, context_recall, answer_correctness 
from ragas import evaluate
from ragas.run_config import RunConfig
import pandas as pd
from tqdm import tqdm
import subprocess
import pickle

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import BaseModel

import warnings
warnings.filterwarnings("ignore")


config = {
    'model_name': 'gemma-2-9b-it-simpo-q4',  # llama3.1-8b-q4 / gemma-2-9b-it-simpo-q4 / tlite-q4
    'embed_model_name_short': 'ubgem3', # e5l (multilingual-e5-large) / bgem3 (bge-m3) / ubgem3 (USER-bge-m3)
    'chunk_size': 512, # либо 512/128, 1024/256, 256/64 (или ?2048/256?)
    'chunk_overlap': 128,
    'llm_framework': 'VLLM', # VLLM, LLamaCpp, Ollama
    'vectorstore_name': 'MILVUS', # база данных MILVUS / FAISS
    'retriever_name': 'vectorstore', # 'vectorstore' / 'ensemble' (BM25 + vertorstore)
    'retriever_k': 4,
    'compressor_name': 'None', # None / 'cross_encoder_reranker' / 'gluing_chunks' 
    'chain_type': 'stuff',
}

ensemble_config = {
    'ensemble_retrievers_names': ['BM25', 'vectorstore'], # применяется только если retriever_name=ensemble
    'ensemble_retrievers_weights': [0.4, 0.6], # применяется только если retriever_name=ensemble
}

llama_config = {
    'repo_id': 'lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF',
    'filename': 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
    'tokenizer': 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4'
    }

gemma_config = {
    'repo_id': "mannix/gemma2-9b-simpo",
    'llm_framework': 'Ollama'
    }

tlite_config = {
    'repo_id': 'mradermacher/saiga_tlite_8b-GGUF',
    'filename': 'saiga_tlite_8b.Q4_K_M.gguf',
    'tokenizer': 'IlyaGusev/saiga_tlite_8b'
    }

reranker_config = {
    'reranker_model': "BAAI/bge-reranker-v2-m3",
    'retriever_k': 30
}


def update_config_with_model(config, llama_config, gemma_config, tlite_config):
    if config['model_name'] == 'llama3.1-8b-q4':
        config.update(llama_config)
    elif config['model_name'] == 'gemma-2-9b-it-simpo-q4':
        config.update(gemma_config)
    elif config['model_name'] == 'tlite-q4':
        config.update(tlite_config)
    else:
        ValueError('Incorrect model_name: choose from llama3.1-8b-q4, gemma-2-9b-it-simpo-q4, or tlite-q4')
    
    if config['embed_model_name_short'] == 'e5l':
        config['embedding_model'] = "intfloat/multilingual-e5-large"
    elif config['embed_model_name_short'] == 'bgem3':
        config['embedding_model'] = 'BAAI/bge-m3'
    elif config['embed_model_name_short'] == 'ubgem3':
        config['embedding_model'] = 'deepvk/USER-bge-m3'
    
    if config['retriever_name'] == 'ensemble':
        config.update(ensemble_config)
    
    if config['compressor_name'] == 'cross_encoder_reranker':
        config.update(reranker_config)


update_config_with_model(config, llama_config, gemma_config, tlite_config)


# внутри используется реранкер и эвристики
class ChunkCompressor(BaseDocumentCompressor):
    chunks: list
    chunk_overlap: int

    def compress_documents(self, chunks, query=None, callbacks=None):
        outputs = []
        print(chunks)
        for chunk in chunks:
            print(chunk)
            current_id = chunk.metadata['chunk_index']
            new_chunk = Document(page_content=chunk.page_content, metadata=chunk.metadata)
            
            # Добавление контекста слева
            if current_id > 0:
                left_neighbor = self.chunks[current_id - 1]
                new_chunk.page_content = left_neighbor.page_content[:-self.config['chunk_overlap']] + new_chunk.page_content
            
            # Добавление контекста справа
            if current_id < len(self.chunks) - 1:
                right_neighbor = self.chunks[current_id + 1]
                new_chunk.page_content += right_neighbor.page_content[self.config['chunk_overlap']:]
            
            outputs.append(new_chunk)
        
        return outputs


class CustomRAGPipeline:
    def __init__(self, 
                 documents_path: str,
                 config: dict,
                 recalc_embedding: bool = False,
                 recalc_chunks: bool = False,
                 ):
        
        self.config = config
        self.documents_path = documents_path
        self.embedding_model = self.config['embedding_model']
        
        self.vectorstore = None
        self.qa_chain = None

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.chunks = None
        
        self.retriever = None
        self.compressor = None #склейка соседних чанков к найденным и затем реранкер (оба - при необходимости)

        self.vectorstore_path = '_'.join([self.config['embed_model_name_short'], 
                                          self.config['vectorstore_name'], 
                                          str(self.config['chunk_size']), 
                                          str(self.config['chunk_overlap'])]
                                        )
        
        self.chunks_path = '_'.join(['chunks', 
                                     str(self.config['chunk_size']), 
                                     str(self.config['chunk_overlap'])]
                                   ) + '.pkl'

        if not recalc_embedding:
            if os.path.exists(self.vectorstore_path) and self.config['vectorstore_name'] == 'FAISS':
                self.vectorstore = FAISS.load_local(self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
            elif os.path.isfile(f"{self.vectorstore_path}.db") and self.config['vectorstore_name'] == 'MILVUS':
                self.vectorstore = Milvus(
                    self.embeddings,
                    connection_args={"uri": f"./{self.vectorstore_path}.db"},
                    collection_name="RAG",
                )
        
        if not recalc_chunks:
            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)

        if self.config['llm_framework'] == 'VLLM':
            self.llm = self.load_vllm_model()
        elif self.config['llm_framework'] == 'LLamaCpp':
            self.llm = self.load_llama_cpp_model()
        elif self.config['llm_framework'] == 'Ollama':
            self.llm = self.load_ollama_model()
            
            
    def load_vllm_model(self):
        # Load the vLLM model from HuggingFace Hub
        repo_id = self.config['repo_id']
        filename = self.config['filename']
        tokenizer = self.config['tokenizer']
        model_path = hf_hub_download(repo_id, filename=filename)
        
        # Initialize vLLM with the downloaded model
        vllm_llm = VLLM(model=model_path,
                        vllm_kwargs={"quantization": "awq", 
                                     'max_model_len': 8192,
                                     'gpu_memory_utilization': 0.7},
                        temperature=0.75,
                        stop=["<|eot_id|>"]
                        )
        
        return vllm_llm


    def load_llama_cpp_model(self):
        repo_id = self.config['repo_id']
        filename = self.config['filename']
        model_path = hf_hub_download(repo_id, filename=filename)
        
        # Инициализация модели LlamaCpp
        llama_cpp_llm = LlamaCpp(model_path=model_path,
                                temperature=0.8,
                                top_p=0.95,
                                top_k=30,
                                max_tokens=64,
                                n_ctx=13000,
                                n_parts=-1,
                                n_gpu_layers=64,
                                n_threads=8,
                                frequency_penalty=1.1,
                                verbose=True,
                                stop=["<|eot_id|>"],  # Остановка на токене EOS
                                )
        
        return llama_cpp_llm

    
    def load_ollama_model(self):
        model_name = self.config['repo_id']
        
        command = f"ollama pull {model_name}"
        
        try:
            subprocess.run(command, shell=True, check=True)
            print(f'Pullled the model {model_name} successfully')
        except subprocess.CalledProcessError as e:
            print(f"Error pulling model {model_name}: {e}")
        
        return OllamaLLM(
            model=model_name,
            temperature=0.8,
            top_p=0.95,
            top_k=30,
            max_tokens=512,
            stop=["<|eot_id|>"]
        )
    
    def load_and_process_documents(self):
        if (self.config['retriever_name'] == 'ensemble') \
            or (self.config['retriever_name'] == 'BM25') \
            or (self.config['compressor_name'] == 'gluing_chunks') \
            or (not self.vectorstore):

            loader = TextLoader(self.documents_path)
            documents = loader.load()
            
            # Split the documents into chunks
            text_splitter = CharacterTextSplitter(
                        separator=" ",
                        chunk_size=self.config['chunk_size'],
                        chunk_overlap=self.config['chunk_overlap'],
                        length_function=len,
                        is_separator_regex=False,
                    )
            texts = text_splitter.split_documents(documents)

            # Add chuck index to metadata to find neighborous-chunks after retrieving
            for index, document in enumerate(texts):
                document.metadata["chunk_index"] = index
            self.chunks = texts
            
            with open(self.chunks_path, "wb") as f:
                pickle.dump(self.chunks, f)
            
            if not self.vectorstore:
                if self.config['vectorstore_name'] == 'FAISS':
                    # Create a FAISS vector store from the documents
                    self.vectorstore = FAISS.from_documents(texts, self.embeddings)
                    self.vectorstore.save_local(self.vectorstore_path)
                elif self.config['vectorstore_name'] == 'MILVUS':
                    self.vectorstore = Milvus.from_documents(
                        texts,
                        self.embeddings,
                        collection_name="RAG",
                        connection_args={"uri": f"./{self.vectorstore_path}.db"})
    
    
    def init_retriever(self):        
        if self.config['retriever_name'] == 'ensemble':
            retrievers = []
            for retriever in self.config['ensemble_retrievers_names']:
                self.config['retriever_name'] = retriever
                retrievers.append(self.init_retriever())
                
            self.retriever = EnsembleRetriever(retrievers=retrievers,
                                              weights=self.config['ensemble_retrievers_weights'])
            self.config['retriever_name'] = 'ensemble'
            
        elif self.config['retriever_name'] == 'BM25':
            self.retriever = BM25Retriever.from_documents(documents=self.chunks)#, metadata=[chunk.metadata for chunk in self.chunks])
            self.retriever.k = self.config['retriever_k']

        elif self.config['retriever_name'] == 'vectorstore':
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config['retriever_k']})
        else:
            ValueError('Incorrect retriever name')
        
        return self.retriever
    
    
    def init_compressor(self):
        if self.config['compressor_name'] == 'gluing_chunks':
            # Инициализация класса компрессора
            self.compressor = ChunkCompressor(chunks=self.chunks, chunk_overlap=self.config['chunk_overlap'])
        elif self.config['compressor_name'] == 'cross_encoder_reranker':
            model = HuggingFaceCrossEncoder(model_name=self.config['reranker_model'])
            self.compressor = CrossEncoderReranker(model=model, top_n=4)
    
    
    def setup_qa_chain(self, custom_prompt: str = None):
        self.init_retriever()
        self.init_compressor()
        
        if self.compressor:            
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor, 
                base_retriever=self.retriever
            )
        else:
            compression_retriever = self.retriever
                
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=custom_prompt
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.config['chain_type'],
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
    
    
    def query(self, question: str) -> Dict:
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")
        
        # Run the QA chain with the provided question
        return self.qa_chain({"query": question})