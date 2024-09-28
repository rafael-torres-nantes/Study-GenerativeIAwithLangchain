import os
from langchain_community.document_loaders import PyPDFLoader

# Função para extrair texto de um arquivo PDF usando LangChain
def extract_text_langchain(pdf_file_path: str):
    """
    Extrai o texto de um arquivo PDF usando o PyPDFLoader da LangChain.

    Args:
        pdf_file_path (str): Caminho para o arquivo PDF a ser processado.

    Returns:
        Dict[str, Any]: Um dicionário contendo o nome do arquivo e o texto extraído.
    """
    try:
        # Carrega o PDF usando o PyPDFLoader
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()

        return {
            "filename": os.path.basename(pdf_file_path),  # Nome do arquivo PDF
            "content_text": documents  # Conteúdo extraído do PDF
        }

    except Exception as e:
        # Captura e trata exceções que possam ocorrer durante o processamento do PDF
        raise print(status_code=500, detail=f"Erro ao processar o arquivo {pdf_file_path}: {str(e)}")