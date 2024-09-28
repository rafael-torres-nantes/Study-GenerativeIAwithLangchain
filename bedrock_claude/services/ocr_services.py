import os
import fitz  # PyMuPDF
from typing import Dict, Any, List
from .ocr_services_langchain import extract_text_langchain

# Função para extrair texto de um arquivo PDF usando PyMuPDF (fitz)
def extract_text_pymupdf(pdf_file_path: str) -> Dict[str, Any]:
    """
    Extrai o texto de um arquivo PDF usando PyMuPDF (fitz).

    Args:
        pdf_file_path (str): Caminho para o arquivo PDF a ser processado.

    Returns:
        Dict[str, Any]: Um dicionário contendo o nome do arquivo e o texto extraído.
    """
    try:
        pdf_document = fitz.open(pdf_file_path)  # Abre o PDF
        full_text = ""

        # Itera por todas as páginas do PDF e extrai o texto
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            full_text += page.get_text()

        pdf_document.close()  # Fecha o arquivo PDF após a leitura
        
        return {
            "filename": os.path.basename(pdf_file_path),  # Nome do arquivo PDF
            "content_text": full_text  # Texto extraído do PDF
        }

    except fitz.fitz.EmptyFileError:
        # Exceção específica para PDFs vazios ou corrompidos
        raise ValueError(f"O arquivo {pdf_file_path} não é um PDF válido ou está corrompido.")
    
    except Exception as e:
        # Captura outras exceções durante o processamento
        raise RuntimeError(f"Erro ao processar o arquivo {pdf_file_path}: {str(e)}")

# Função para processar todos os PDFs em um diretório e extrair seus textos
def extract_texts_from_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Percorre o diretório e suas subpastas à procura de arquivos PDF e extrai o texto de cada um.

    Args:
        directory (str): Caminho do diretório que contém os arquivos PDF.

    Returns:
        List[Dict[str, Any]]: Lista de dicionários contendo o nome do arquivo e o texto extraído.
    """
    extracted_texts = []

    # Percorre o diretório e subpastas
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):  # Verifica se o arquivo tem extensão .pdf
                pdf_path = os.path.join(root, file)
                
                # Extrai o texto do PDF utilizando a função de extração
                try:
                    extracted_text = extract_text_langchain(pdf_path)
                    extracted_texts.append(extracted_text)
                
                except Exception as e:
                    print(f"Erro ao processar o arquivo {pdf_path}: {str(e)}")
    
    return extracted_texts

# Exemplo de uso:
if __name__ == "__main__":
    directory = './data/game_rules/'  # Substitua pelo caminho do diretório que contém os PDFs
    extracted_pdfs = extract_texts_from_directory(directory)

    # Exibe os textos extraídos (opcional)
    for pdf in extracted_pdfs:
        print(f"Arquivo: {pdf['filename']}")
        print(f"Conteúdo extraído: {pdf['content_text'][:500]}...")  # Exibe os primeiros 500 caracteres
        print("====================================")
