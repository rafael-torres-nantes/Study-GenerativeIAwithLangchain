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
# Função que constrói o template de prompt para o modelo de geração
# --------------------------------------------------------------------
def create_prompt_template(query, contexts):
    """
    Constrói o prompt com base no contexto e na pergunta do usuário.
    
    :param contexts: Textos recuperados dos documentos relevantes.
    :param query: Pergunta do usuário.
    :return: String formatada como prompt para o modelo.
    """
    prompt = f"""
    <context>
    {contexts}
    </context>

    <question>
    {query}
    </question>
    """
    return prompt

# --------------------------------------------------------------------
# Função para gerar embeddings usando o Amazon Titan V2
# --------------------------------------------------------------------
def generate_embedding(user_query):
    """
    Gera o embedding de um texto usando o modelo Titan Embedding.

    :param text: Texto a ser embeddado.
    :return: Vetor de embedding.
    """
    response = bedrock_client.invoke_model(
        modelId=TITAN_EMBEDDING_MODEL_ID,
        accept='application/json',
        contentType='application/json',
        body=json.dumps({"inputText": user_query}),
    )
    response_body = json.loads(response['body'].read())
    return response_body['embedding']

# --------------------------------------------------------------------
# Função para buscar documentos similares com base na query
# --------------------------------------------------------------------
def semantic_search(query_embedding, num_results=5):
    """
    Realiza busca semântica baseada nos embeddings da consulta.

    :param query_embedding: Embedding gerado a partir da consulta do usuário.
    :return: Documentos mais relevantes.
    """
    response = bedrock_agent_client.retrieve_and_generate(
        input={
            "text": query_embedding  # Prompt fornecido pelo usuário
        },
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",  # Tipo de consulta: base de conhecimento
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,  # ID da base de conhecimento
                "modelArn": TITAN_EMBEDDING_MODEL_ID,  # Modelo a ser utilizado
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": num_results  # Quantidade de resultados retornados
                    }
                }
            }
        },
    )
    return response

# --------------------------------------------------------------------
# Função para gerar uma resposta usando o Claude 3.5 e documentos recuperados
# --------------------------------------------------------------------
def generate_response_with_claude(query, context):
    """
    Gera uma resposta com o Claude 3.5 com base nos documentos recuperados.

    :param query: Pergunta do usuário.
    :return: Resposta gerada.
    """
    body = json.dumps({"inputText": create_prompt_template(query, context)})
    response = bedrock_client.invoke_model(
        body=body,
        modelId=CLAUDE_MODEL_ID,
        accept='application/json',
        contentType='application/json'
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['outputText']

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

    # Gera o embedding da consulta do usuário
    query_embedding = generate_embedding(user_query)

    # Realiza busca semântica usando os embeddings gerados
    relevant_text = semantic_search(query_embedding)

    # Concatena os documentos relevantes em um único contexto
    context = "\n".join(relevant_text)

    # Gera a resposta usando o Claude 3.5
    generated_text = generate_response_with_claude(user_query, context)

    # Retorna a resposta gerada com o código de status 200
    return {'statusCode': 200, 'body': generated_text}
