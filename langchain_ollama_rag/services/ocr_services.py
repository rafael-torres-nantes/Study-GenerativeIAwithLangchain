from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

# Função para carregar todos os PDFs de um diretório usando LangChain
def load_documents_from_directory(pdf_directory_path: str):
    """
    Carrega todos os arquivos PDF em um diretório usando o PyPDFDirectoryLoader da LangChain.

    Args:
        pdf_directory_path (str): Caminho para o diretório contendo os arquivos PDF a serem processados.

    Returns:
        List[Document]: Uma lista de documentos contendo o texto extraído de cada arquivo PDF.
    """
    try:
        # Carrega todos os PDFs do diretório usando o PyPDFDirectoryLoader
        directory_loader = PyPDFDirectoryLoader(pdf_directory_path)
        return directory_loader.load()

    except Exception as e:
        # Captura e trata exceções que possam ocorrer durante o processamento dos PDFs no diretório
        raise print(status_code=500, detail=f"Erro ao processar o diretório {pdf_directory_path}: {str(e)}")