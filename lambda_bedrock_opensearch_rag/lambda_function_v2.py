import json
import boto3

# Inicializa o cliente do Bedrock Agent para consultar e gerar respostas
bedrock_agent_client = boto3.client('bedrock-agent-runtime')
bedrock_client = boto3.client('bedrock-runtime')

# IDs dos modelos usados no processamento
CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # Modelo de geração de texto
TITAN_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"  # Modelo de embeddings (não utilizado aqui)
KNOWLEDGE_BASE_ID = ""  # ID da base de conhecimento a ser definida

# --------------------------------------------------------------------
# Função que constrói o prompt com base no contexto e na pergunta
# --------------------------------------------------------------------
def get_prompt():
    """
    Constrói o prompt genérico a ser usado no modelo.

    :return: String formatada como prompt para o modelo.
    """
    prompt = """
    $search_results$
    
    $output_format_instructions$
    """
    return prompt


# --------------------------------------------------------------------
# Função para gerar uma resposta utilizando RAG (retrieval and generation)
# --------------------------------------------------------------------
def generate_response_with_RAG(user_query, num_results=5):
    """
    Recupera documentos relevantes e gera uma resposta com base no prompt do usuário.

    :param user_query: Pergunta do usuário.
    :param num_results: Número de documentos a recuperar.
    :return: Resposta gerada pelo modelo.
    """
    response = bedrock_agent_client.retrieve_and_generate(
        input={
            "text": user_query  # Prompt fornecido pelo usuário
        },
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",  # Tipo de consulta: base de conhecimento
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,  # ID da base de conhecimento
                "modelArn": CLAUDE_MODEL_ID,  # Modelo a ser utilizado
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": num_results  # Quantidade de resultados retornados
                    }
                },
                'generationConfiguration': {
                    'promptTemplate': {
                        'textPromptTemplate': get_prompt()  # Template do prompt
                    }
                }
            }
        },
    )
    return response


# --------------------------------------------------------------------
# Função para extrair as citações e contextos das fontes
# --------------------------------------------------------------------
def extract_citations_and_contexts(response):
    """
    Extrai as citações e os contextos de referência da resposta gerada.

    :param response: Resposta gerada pelo modelo.
    :return: Lista de contextos dos documentos citados.
    """
    citations = response["citations"]
    contexts = []
    
    for citation in citations:
        references = citation["retrievedReferences"]
        for ref in references:
            contexts.append(ref["content"]["text"])

    return contexts


# --------------------------------------------------------------------
# Função Lambda principal que processa o evento recebido
# --------------------------------------------------------------------
def lambda_handler(event, context):
    """
    Função Lambda principal que recebe o evento e processa o prompt do usuário.

    :param event: Dados do evento recebido (contendo o prompt).
    :param context: Contexto de execução da Lambda.
    :return: Resposta gerada pelo modelo.
    """
    # Log do prompt recebido para depuração
    user_query = event['prompt']  # Extrai o prompt do evento
    print("Prompt do Usuário:", user_query)

    # Gera a resposta com base no prompt do usuário
    response = generate_response_with_RAG(user_query, num_results=5)

    # Extrai os contextos das citações
    contexts = extract_citations_and_contexts(response)
    print(f"Contextos extraídos: {contexts}")
    
    # Extrai o texto gerado da resposta
    generated_text = response["output"]["text"]

    # Retorna a resposta gerada com o código de status 200
    return {'statusCode': 200, 'body': generated_text}
