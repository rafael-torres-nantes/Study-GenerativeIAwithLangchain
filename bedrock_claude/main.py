from services.ocr_services import extract_texts_from_directory
from controller.controller_claude import Controller

# Bloco principal executado ao rodar o script diretamente
if __name__ == "__main__":
    """
    Bloco de execução principal:
    - Inicializa o Controller.
    - Tenta executar o modelo Bedrock se as credenciais da AWS forem válidas.
    """

    # Define o caminho do diretório onde estão localizados os arquivos PDF para processamento.
    path = '../data/legal_docs/ARE1467492' 

    # Extrai os textos de todos os arquivos PDF encontrados no diretório e suas subpastas.
    # A função `process_pdfs_in_directory` retorna uma lista de textos extraídos dos PDFs.
    extract_texts = extract_texts_from_directory(path)

    # Inicializa uma variável para armazenar o resultado final após o processamento dos documentos.
    output_text = ""

    # Inicializa uma instância da classe `Controller`, que será responsável por processar os documentos e executar o modelo Bedrock.
    controller = Controller()

    for text in extract_texts:
        # Para cada texto extraído, chama o método `process_documents` do `Controller` para processar o conteúdo.
        controller.process_documents(text)

        # Executa o modelo Bedrocke o resultado é adicionado à variável `output_text`.
        output_text += controller.execute_bedrock_model()
