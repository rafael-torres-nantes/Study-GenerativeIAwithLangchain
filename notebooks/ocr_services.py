import os
import re
import fitz  # PyMuPDF
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader

# Função para garantir que o diretório de uploads exista
def create_folder(folder: str):
    """Cria o diretório especificado se não existir."""
    if not os.path.exists(folder):
        os.makedirs(folder)

# Função para extrair texto de um arquivo PDF usando LangChain
def extract_text_from_pdf_langchain(pdf_file_path: str) -> Dict[str, Any]:
    """
    Extrai o texto de um arquivo PDF usando LangChain.

    Args:
        pdf_file_path (str): O caminho do arquivo PDF a ser processado.

    Returns:
        Dict[str, Any]: Um dicionário contendo o nome do arquivo e o texto extraído.
    """
    try:
        # Carrega o PDF usando o PyPDFLoader
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()

        return {
            "filename": os.path.basename(pdf_file_path),
            "content_text": documents
        }

    except Exception as e:
        # Captura quaisquer erros que possam ocorrer durante o processamento
        raise HTTPException(status_code=500, detail=f"Erro ao processar o arquivo {pdf_file_path}: {str(e)}")

# Função para extrair texto de um arquivo PDF usando PyMuPDF (fitz)
def extract_text_from_pdf(pdf_file_path: str) -> Dict[str, Any]:
    """
    Extrai o texto de um arquivo PDF usando PyMuPDF (fitz).

    Args:
        pdf_file_path (str): O caminho do arquivo PDF a ser processado.

    Returns:
        Dict[str, Any]: Um dicionário contendo o nome do arquivo e o texto extraído.
    """
    try:
        pdf_document = fitz.open(pdf_file_path)  # Abre o PDF a partir do caminho do arquivo
        full_text = ""

        # Itera sobre todas as páginas do PDF e extrai o texto
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            full_text += page.get_text()

        pdf_document.close()  # Fecha o arquivo PDF após a leitura
        
        return {
            "filename": os.path.basename(pdf_file_path),
            "content_text": full_text
        }

    except fitz.fitz.EmptyFileError:
        # Lança uma exceção se o arquivo estiver vazio ou corrompido
        raise HTTPException(status_code=400, detail=f"O arquivo {pdf_file_path} não é um PDF válido.")
    except Exception as e:
        # Captura quaisquer outros erros que possam ocorrer durante o processamento
        raise HTTPException(status_code=500, detail=f"Erro ao processar o arquivo {pdf_file_path}: {str(e)}")

# Função para percorrer o diretório e subpastas em busca de arquivos PDF
def process_pdfs_in_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Percorre um diretório e suas subpastas em busca de arquivos PDF e extrai o texto de cada um.

    Args:
        directory (str): O diretório que contém os arquivos PDF.

    Returns:
        List[Dict[str, Any]]: Uma lista de dicionários contendo o nome do arquivo e o texto extraído.
    """
    extracted_texts = []

    # Percorre o diretório e todas as subpastas
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):  # Verifica se o arquivo é um PDF
                pdf_path = os.path.join(root, file)
                
                # Extrai o texto do PDF usando uma das funções de extração
                try:
                    extracted_text = extract_text_from_pdf(pdf_path)
                    extracted_texts.append(extracted_text)
                except Exception as e:
                    print(f"Erro ao processar o arquivo {pdf_path}: {str(e)}")
    
    return extracted_texts

# Exemplo de uso:
if __name__ == "__main__":
    directory = './files/ARE1467492'  # Substitua pelo caminho do diretório que contém os PDFs
    extracted_pdfs = process_pdfs_in_directory(directory)

    # # Exibe os textos extraídos
    # for pdf in extracted_pdfs:
    #     print(f"Arquivo: {pdf['filename']}")
    #     print(f"Conteúdo extraído: {pdf['content_text'][:500]}...")  # Exibe os primeiros 500 caracteres
    #     print("====================================")
