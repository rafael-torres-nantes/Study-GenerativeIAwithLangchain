KNOWLEDGE_BASE_ID = ""  # ID da base de conhecimento (precisa ser preenchido)

# --------------------------------------------------------------------
# Função que busca documentos relevantes na base de conhecimento
# --------------------------------------------------------------------
def retrieve_chunks(bedrock_agent_client, user_query, number_results=5):
    """
    Realiza uma consulta para obter os documentos mais relevantes da base de conhecimento.

    :param user_query: Pergunta do usuário.
    :param number_results: Número de resultados a recuperar.
    :return: Resposta com os resultados da busca.
    """
    # O cliente do Bedrock é usado para fazer uma busca semântica na base de conhecimento
    relevant_docs_response = bedrock_agent_client.retrieve(
        retrievalQuery={
            "text": user_query  # Usa o texto do usuário como a consulta de busca
        },
        knowledgeBaseId=KNOWLEDGE_BASE_ID,  # O ID da base de conhecimento deve ser especificado
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": number_results,  # Define o número de resultados a serem retornados
                "overrideSearchType": "SEMANTIC",  # Utiliza busca semântica (pode ser alterado para híbrida)
            }
        }
    )

    retrieval_results = relevant_docs_response['retrievalResults']
    return retrieval_results  # Retorna os documentos mais relevantes

# --------------------------------------------------------------------
# Função que extrai os textos dos resultados da busca
# --------------------------------------------------------------------
def extract_contexts_from_chunks(retrieval_results):
    """
    Extrai os textos dos documentos retornados pela busca.

    :param retrieval_results: Lista de documentos recuperados.
    :return: Lista de textos dos documentos.
    """
    # Extrai o conteúdo de texto de cada documento recuperado
    contexts = [result['content']['text'] for result in retrieval_results]
    return contexts  # Retorna uma lista de contextos extraídos dos documentos