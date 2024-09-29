from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Importando a classe Chroma do pacote atualizado
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from prompts.promptGameRules import promptAskGameRules as get_prompt
from embedding.embedding_models import get_embedding_ollama


class OllamaService:
    def __init__(self):
        """
        Inicializa o serviço de interação com o modelo Ollama e gerencia o armazenamento de dados 
        utilizando o Chroma como sistema de banco de dados vetorial.
        Define o modelo a ser utilizado e o caminho do diretório onde os dados do Chroma serão armazenados.
        """
        self.model_id = 'mistral'  # Identificador do modelo a ser utilizado
        self.chroma_directory = 'chroma'  # Diretório para armazenar dados do Chroma

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Divide os documentos carregados em chunks menores para facilitar o processamento.

        Args:
            documents (list[Document]): Lista de documentos a serem divididos.

        Returns:
            list[Document]: Lista de chunks gerados a partir dos documentos.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Tamanho máximo de cada chunk
            chunk_overlap=80,  # Sobreposição de caracteres entre chunks consecutivos
            length_function=len,  # Função para calcular o comprimento do texto
            is_separator_regex=False,  # Define se um regex será usado para separar textos
        )
        return text_splitter.split_documents(documents)

    def invoke_model(self, question_text):
        """
        Invoca o modelo Ollama com a pergunta fornecida e retorna a resposta gerada.

        Args:
            question_text (str): Texto da pergunta a ser feita ao modelo.

        Returns:
            str: Resposta gerada pelo modelo Ollama com base na pergunta e no contexto dos documentos.
        """
        # Define a função de embedding a ser utilizada
        embedding_function = get_embedding_ollama()  # Obtém a função de embedding específica para o Ollama

        # Cria o banco de dados Chroma com a função de embedding definida
        db = Chroma(persist_directory=self.chroma_directory, embedding_function=embedding_function)

        # Realiza a busca de similaridade no banco de dados com base na pergunta
        results = db.similarity_search_with_score(question_text, k=5)

        # Monta o texto de contexto a partir dos documentos retornados pela busca
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        # Cria um template de prompt com o contexto e a pergunta
        prompt_template = ChatPromptTemplate.from_template(get_prompt())
        prompt = prompt_template.format(context=context_text, question=question_text)

        # Inicializa o modelo Ollama e obtém a resposta
        model = Ollama(model=self.model_id)    
        response_text = model.invoke(prompt)

        # Obtém os IDs das fontes dos documentos utilizados na resposta
        source_ids = [doc.metadata.get("id", None) for doc, _score in results]

        return response_text  # Retorna a resposta gerada
