import os

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import ollama

class FAISSVectorStore:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []

    def add_vectors(self, vectors, documents, metadata):
        vectors = np.array(vectors).astype('float32')
        self.index.add(vectors)
        self.documents.extend(documents)
        self.metadata.extend(metadata)

    def search(self, query_vector, k=5):
        query_vector = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        return [self.documents[i] for i in indices[0]], [self.metadata[i] for i in indices[0]]

class PDFRAGSystem:
    def __init__(self, pdf_folder, embedding_model='LaBSE'):
        """
        Initializes the RAG system.
        
        Args:
            pdf_folder: path to the folder containing PDF files
            embedding_model: embedding model to use for text embeddings
        """
        self.pdf_folder = pdf_folder
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(allow_reset=True))
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_documents",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )
        
    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from a PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Splits the text in chunks. Overlap is recommended in order to retain context."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def process_pdfs(self):
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        existing_sources = {meta['source'] for meta in self.collection.get()['metadatas']}

        for pdf_file in pdf_files:
            if pdf_file in existing_sources:
                continue

            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            print(f"Processing new file: {pdf_file}...")
            
            text = self.extract_text_from_pdf(pdf_path)
            chunks = self.chunk_text(text)
            
            ids = [f"{pdf_file}_{i}" for i in range(len(chunks))]
            self.collection.add(
                documents=chunks,
                ids=ids,
                metadatas=[{"source": pdf_file} for _ in chunks]
            )
        
    
    def retrieve_relevant_chunks(self, query, k=5):
        """Retrieves the k most relevant chunks for a given query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        relevant_chunks = results['documents'][0]
        sources = results['metadatas'][0]
        
        context = "\n\n---\n\n".join([
            f"Fonte: {src['source']}\n\n{chunk}"
            for chunk, src in zip(relevant_chunks, sources)
        ])
        
        return context
    
    def generate_answer(self, query, k=5, model='gemma3:4b'):
        """Generates a response based on the documents."""
        context = self.retrieve_relevant_chunks(query, k)
        
        prompt = f"""You are a helpful AI assistant that answers questions. If needed, utilize the following context to base you answers.
        
        Context:
        {context}
        
        Question: {query}
        
        Be concise and clear in your answer. If you still can't answer the question, tell the user that you don't know."""
        
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': 0.3,
                'num_ctx': 8000
            }
        )
        
        return response['response']
    
    def query(self, question, k=5):
        """Asks the system a question."""
        print("\nProcessing you question...")
        answer = self.generate_answer(question, k)
        print("\Response:")
        print(answer)

if __name__ == "__main__":
    PDF_FOLDER = "pdfs"
    
    print("Initializing RAG system...")
    rag_system = PDFRAGSystem(PDF_FOLDER)
    
    print("Loading documents...")
    rag_system.process_pdfs()
    
    while True:
        print("\nType your question (type 'quit' or 'exit' to exit):\n")
        user_query = input().strip()
        
        if user_query.lower() in ['quit', 'exit']:
            break
            
        if user_query:
            rag_system.query(user_query)