import json
import boto3

# Inicializa o cliente do Bedrock Agent para consultar e gerar respostas
bedrock_agent_client = boto3.client('bedrock-agent-runtime')  
bedrock_client = boto3.client('bedrock-runtime')

# IDs dos modelos usados no processamento
CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # ID do modelo Claude para geração de texto
TITAN_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"  # Modelo de embeddings (não usado neste caso)
KNOWLEDGE_BASE_ID = ""  # ID da base de conhecimento (precisa ser preenchido)

# --------------------------------------------------------------------
# Função que constrói o template de prompt para o modelo de geração
# --------------------------------------------------------------------
def create_prompt_template(contexts, query):
    """
    Constrói o prompt com base no contexto e na pergunta do usuário.
    
    :param contexts: Textos recuperados dos documentos relevantes.
    :param query: Pergunta do usuário.
    :return: String formatada como prompt para o modelo.
    """
    # O prompt é formatado para incluir o contexto dos documentos recuperados e a pergunta do usuário
    prompt = f"""
    <context>
    {contexts}
    </context>

    <question>
    {query}
    </question>
    """
    return prompt  # Retorna o prompt formatado

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

# --------------------------------------------------------------------
# Função que gera o corpo da requisição para o Bedrock
# --------------------------------------------------------------------
def generate_request_body(prompt):
    """
    Gera o corpo da requisição para enviar ao modelo Bedrock.

    Inclui o prompt e configurações de geração de texto como o número máximo de tokens, temperatura e topP.

    Returns:
        str: O corpo da requisição em formato JSON.
    """
    # Define a estrutura de mensagens para o modelo
    messages = [
        { "role" : 'user',
            "content": [
                {'type' : 'text',
                'text': prompt  # Usa o prompt fornecido como conteúdo de texto
                }
            ]
        }
    ]
    
    # Define os parâmetros de geração, incluindo tokens máximos e parâmetros de temperatura
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": messages,
        "temperature": 0.5,  # Temperatura: controla a aleatoriedade da geração
        "top_p": 1  # top_p: controla a inclusão dos tokens mais prováveis
    }
    return json.dumps(request_body)  # Retorna o corpo da requisição em formato JSON

# --------------------------------------------------------------------
# Função que invoca o modelo e retorna a resposta gerada
# --------------------------------------------------------------------
def invoke_model(prompt):
    """
    Invoca o modelo Bedrock com o prompt fornecido.

    :param prompt: O prompt gerado que será enviado ao modelo.
    :return: Resposta de texto gerada pelo modelo.
    """
    # Invoca o modelo Bedrock com o corpo da requisição gerado
    response = bedrock_client.invoke_model(
        modelId=CLAUDE_MODEL_ID, 
        contentType='application/json',
        accept='application/json',
        body=generate_request_body(prompt)  # Gera o corpo da requisição
    )

    # Lê o corpo da resposta e extrai o texto gerado pelo modelo
    response_body = json.loads(response.get('body').read())
    response_text = response_body.get('content')[0]['text']

    return response_text  # Retorna o texto gerado

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

    # Busca documentos relevantes
    search_response = retrieve_chunks(user_query)  # Chama a função para buscar documentos relevantes

    # Extrai os contextos dos resultados da busca
    contexts = extract_contexts_from_chunks(search_response)  # Extrai os textos dos documentos recuperados
    
    # Exibe os resultados da busca no log para depuração
    print("Contexto extraído:", contexts)

    # Gera o prompt com os contextos extraídos e a pergunta do usuário
    prompt = create_prompt_template(user_query, context)

    # Invoca o modelo para gerar o texto com base no prompt
    generated_text = invoke_model(prompt)

    # Retorna a resposta gerada com o código de status 200
    return {'statusCode': 200, 'body': generated_text}