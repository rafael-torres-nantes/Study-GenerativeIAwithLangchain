import os, shutil
from langchain_chroma import Chroma 
from embedding.embedding_models import get_embedding_ollama

CHROMA_PATH = "./chroma"

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_chroma(chunks):
    """
    Adiciona os chunks de documentos ao banco de dados Chroma. Apenas chunks novos, 
    que ainda não existem no banco de dados, são adicionados.

    Args:
        chunks (List[Document]): Lista de chunks a serem adicionados ao banco de dados.
    """
    # Carrega a base de dados existente.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_ollama())

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Obtém os documentos existentes no banco de dados.
    existing_items = db.get(include=[])  # IDs são incluídos por padrão
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Adiciona apenas os documentos que ainda não estão no banco de dados.
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")