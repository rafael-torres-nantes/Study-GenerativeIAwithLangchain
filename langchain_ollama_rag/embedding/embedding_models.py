from langchain_community.embeddings.ollama import OllamaEmbeddings

# !curl -fsSL https://ollama.com/install.sh | sh
# !ollama run llama3.2 or !ollama pull mistral
# !ollama serve

def get_embedding_ollama():
    # Inicializa os embeddings do modelo Ollama
    model_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return model_embeddings