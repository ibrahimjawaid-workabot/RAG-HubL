import json
import os
import sys
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# 🚨 CRITICAL FIX: Use the explicit full paths from the main package (This is the stable way).
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain

# --- CONFIGURATION (UPDATED FOR YOUR MODEL) ---
DATA_FILE = "hubspot_hubl_docs.jsonl"
DB_DIR = "./hubl_vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Recommended fast, high-quality embedding model
LLM_MODEL = "deepseek-v3.1:671b-cloud" # YOUR CHOSEN OLLAMA MODEL
# ---------------------

# --- PHASE 1: INDEXING (ONE-TIME SETUP) ---

def load_data(file_path):
    """Loads JSONL data and converts it into a list of LangChain Document objects."""
    print("1. Loading data from JSONL...")
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    # Input structure: "url", "title", "section", "content"
                    documents.append(
                        Document(
                            page_content=item.get("content", ""), # Use .get() for safety
                            metadata={
                                "source": item.get("url", "N/A"),
                                "title": item.get("title", "N/A"),
                                "section": item.get("section", "N/A")
                            }
                        )
                    )
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error on line {i+1}: {e}")
                    continue

        if not documents:
             print(f"Error: No valid documents were loaded from {file_path}. Check file contents.")
             return []
        
        print(f"   -> Loaded {len(documents)} documents.")
        return documents
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}. Please ensure it is in the same directory.")
        return []

def create_index(documents):
    """Chunks, embeds, and stores the documents in ChromaDB."""
    print("2. Chunking HubL documents...")
    # Optimal settings for detailed code/documentation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Split into {len(chunks)} chunks.")

    print(f"3. Initializing Embedding Model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print(f"4. Creating/Persisting Vector Store to {DB_DIR}...")
    # This step executes the embedding and storage
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    db.persist()
    print("   -> HubL Vector Indexing Complete.")
    return db

# --- PHASE 2: QUERYING (THE RAG CHAIN) ---

def setup_rag_chain(db, embeddings):
    """Sets up the Retrieval Augmented Generation pipeline."""
    print(f"\n5. Setting up RAG Query Chain using LLM: {LLM_MODEL}...")
    
    # 🚨 CRITICAL: Use your specified Ollama model
    llm = Ollama(model=LLM_MODEL)
    
    # The System Prompt is the instruction that turns the LLM into a HubL Expert
    system_prompt = (
        "You are an expert HubL (HubSpot Template Language) developer. "
        "Your job is to provide accurate HubL code and concise explanations based "
        "ONLY on the provided context. If the answer is not in the context, "
        "state that you cannot provide the code. Always prioritize providing "
        "working HubL syntax with clear examples. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Load the retriever from the vector store
    # k=5 is a good start for a complex model like DeepSeek, giving it more context
    retriever = db.as_retriever(search_kwargs={"k": 5}) 
    
    # Chain to combine the retrieved documents with the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Full RAG chain (retrieval -> augmentation -> generation)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("   -> RAG Chain Ready.")
    return rag_chain

# --- MAIN EXECUTION ---

def run_assistant():
    """Main function to run the HubL RAG Assistant."""
    
    # Determine if we need to index or just load
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(DB_DIR):
        print(f"Found existing vector store at {DB_DIR}. Loading index...")
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        documents = load_data(DATA_FILE)
        if not documents:
            sys.exit("Exiting due to data loading error.")
        db = create_index(documents)

    rag_chain = setup_rag_chain(db, embeddings)
    
    # --- INTERACTIVE CHAT LOOP ---
    print("\n--- HubL Assistant Ready. Start chatting. ---")
    print(f"Using LLM: {LLM_MODEL}. Type 'exit' or 'quit' to end.")
    
    while True:
        user_input = input("\nYour HubL Query: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting HubL Assistant. Goodbye!")
            break
            
        print("Assistant thinking...")
        
        # Invoke the RAG chain
        response = rag_chain.invoke({"input": user_input})
        
        # Print the LLM's final answer
        print("\n" + "="*25)
        print("🤖 HubL Assistant Answer:")
        print(response["answer"])
        print("="*25)
        
        # Print the source documents retrieved (for citation)
        print("\n--- Sources Retrieved (For Verification) ---")
        for i, doc in enumerate(response.get("context", [])):
            # Safely access metadata
            title = doc.metadata.get('title', 'N/A')
            section = doc.metadata.get('section', 'N/A')
            source = doc.metadata.get('source', 'N/A')
            print(f"Source {i+1}: {title} - {section}")
            print(f"URL: {source}\n")


if __name__ == "__main__":
    run_assistant()