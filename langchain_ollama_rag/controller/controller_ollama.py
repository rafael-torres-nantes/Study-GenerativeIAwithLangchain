from services.chromadb_service import add_to_chroma, clear_database
from services.ocr_services import load_documents_from_directory
from services.ollama_services import OllamaService

class Controller:
    def __init__(self):
        """
        Inicializa o controlador, que gerencia a interação com os serviços de OCR, 
        Chroma e o modelo Ollama. Essa classe fornece métodos para carregar documentos 
        e executar o modelo de chat.
        """
        self.ollama_service = OllamaService()  # Inicialize o serviço Ollama se necessário

    def delete_database(self):
        """
        Carrega documentos de um diretório PDF, divide em chunks e os adiciona ao 
        banco de dados Chroma.

        Args:
            pdf_directory (str): Caminho para o diretório que contém os arquivos PDF.
        """
        # Limpa documentos do diretório PDF
        clear_database()

        print("✅ Documentos removidos com sucesso")

    def process_documents(self, pdf_directory):
        """
        Carrega documentos de um diretório PDF, divide em chunks e os adiciona ao 
        banco de dados Chroma.

        Args:
            pdf_directory (str): Caminho para o diretório que contém os arquivos PDF.
        """
        # Carregar documentos do diretório PDF
        self.documents = load_documents_from_directory(pdf_directory)

        # Dividir os documentos em chunks menores
        self.chunks = self.ollama_service.split_documents(self.documents)

        # Adicionar os chunks ao banco de dados Chroma
        add_to_chroma(self.chunks)

        print("✅ Documentos processados e adicionados ao Chroma com sucesso.")

    def execute_ollama_model(self, question_text):
        """
        Executa o modelo de chat Ollama com a pergunta fornecida e retorna a resposta gerada.

        Args:
            question_text (str): Texto da pergunta a ser feita ao modelo.

        Returns:
            str: Resposta gerada pelo modelo Ollama com base na pergunta e no contexto dos documentos.
        """
        # Chama o método para invocar o modelo
        response = self.ollama_service.invoke_model(question_text)  
        return response
