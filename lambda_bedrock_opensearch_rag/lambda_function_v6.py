import json
import boto3

# Inicializa o cliente do Bedrock Agent e Bedrock Runtime
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

    :param user_query: Texto a ser embeddado.
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
# Função que busca documentos relevantes na base de conhecimento
# --------------------------------------------------------------------
def retrieve_chunks(user_query, number_results=5):
    """
    Realiza uma consulta para obter os documentos mais relevantes da base de conhecimento.

    :param user_query: Pergunta do usuário.
    :param number_results: Número de resultados a recuperar.
    :return: Resposta com os resultados da busca.
    """
    relevant_docs_response = bedrock_agent_client.retrieve(
        retrievalQuery={"text": user_query},
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": number_results,
                "overrideSearchType": "SEMANTIC",  # Utiliza busca semântica
            }
        }
    )

    retrieval_results = relevant_docs_response['retrievalResults']
    return retrieval_results

# --------------------------------------------------------------------
# Função que extrai os textos dos resultados da busca
# --------------------------------------------------------------------
def extract_contexts_from_results(retrieval_results):
    """
    Extrai os textos dos documentos retornados pela busca.

    :param retrieval_results: Lista de documentos recuperados.
    :return: Lista de textos dos documentos.
    """
    contexts = [result['content']['text'] for result in retrieval_results]
    return contexts

# --------------------------------------------------------------------
# Função que gera o corpo da requisição para o Bedrock
# --------------------------------------------------------------------
def generate_request_body(prompt):
    """
    Gera o corpo da requisição para enviar ao modelo Bedrock.

    Inclui o prompt e configurações de geração de texto como o número máximo de tokens, temperatura e topP.

    :param prompt: O prompt gerado.
    :return: Corpo da requisição em formato JSON.
    """
    messages = [
        {
            "role": 'user',
            "content": [{'type': 'text', 'text': prompt}]
        }
    ]

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": messages,
        "temperature": 0.5,  # Controla a aleatoriedade da geração
        "top_p": 1  # Inclui os tokens mais prováveis
    }

    return json.dumps(request_body)

# --------------------------------------------------------------------
# Função que invoca o modelo e retorna a resposta gerada
# --------------------------------------------------------------------
def invoke_model(prompt):
    """
    Invoca o modelo Bedrock com o prompt fornecido.

    :param prompt: O prompt gerado que será enviado ao modelo.
    :return: Resposta de texto gerada pelo modelo.
    """
    response = bedrock_client.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=generate_request_body(prompt)
    )

    response_body = json.loads(response.get('body').read())
    response_text = response_body.get('content')[0]['text']

    return response_text

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
    user_query = event['prompt']
    print("Prompt do Usuário:", user_query)

    # Gera o embedding da consulta do usuário
    query_embedding = generate_embedding(user_query)

    # Busca documentos relevantes
    retrieval_results = retrieve_chunks(query_embedding)
    print("Documentos Relevantes:", retrieval_results)

    # Concatena os documentos relevantes em um único contexto
    contexts = extract_contexts_from_results(retrieval_results)
    print("Contextos extraídos:", contexts)

    # Constrói o prompt com base nos contextos
    prompt = create_prompt_template(user_query, contexts)

    # Gera a resposta final
    generated_text = invoke_model(prompt)

    # Retorna a resposta gerada com o código de status 200
    return {'statusCode': 200, 'body': generated_text}