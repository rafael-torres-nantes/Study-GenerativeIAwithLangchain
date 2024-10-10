import json

# IDs dos modelos usados no processamento
CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # ID do modelo Claude para geração de texto

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
def invoke_model(bedrock_client, prompt):
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