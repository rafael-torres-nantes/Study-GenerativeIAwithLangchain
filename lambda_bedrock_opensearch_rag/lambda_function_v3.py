import json
import boto3

from prompts.prompts_template import create_prompt_template
from bedrock_models.inference_model import invoke_model
from bedrock_agents.retrieve_chunks import retrieve_chunks, extract_contexts_from_chunks

# Inicializa o cliente do Bedrock Agent para consultar e gerar respostas
bedrock_agent_client = boto3.client('bedrock-agent-runtime')  
bedrock_client = boto3.client('bedrock-runtime')

# IDs dos modelos usados no processamento
CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # ID do modelo Claude para geração de texto
TITAN_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"  # Modelo de embeddings (não usado neste caso)
KNOWLEDGE_BASE_ID = ""  # ID da base de conhecimento (precisa ser preenchido)

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
    search_response = retrieve_chunks(bedrock_agent_client, user_query)  # Chama a função para buscar documentos relevantes

    # Extrai os contextos dos resultados da busca
    contexts = extract_contexts_from_chunks(search_response)  # Extrai os textos dos documentos recuperados
    
    # Exibe os resultados da busca no log para depuração
    print("Contexto extraído:", contexts)

    # Gera o prompt com os contextos extraídos e a pergunta do usuário
    prompt = create_prompt_template(user_query, context)

    # Invoca o modelo para gerar o texto com base no prompt
    generated_text = invoke_model(bedrock_client, prompt)

    # Retorna a resposta gerada com o código de status 200
    return {'statusCode': 200, 'body': generated_text}