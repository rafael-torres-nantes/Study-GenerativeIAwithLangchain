import json
import boto3

from prompts.prompts_template import create_prompt_template
from bedrock_models.inference_model import invoke_model
from bedrock_models.embedding_model import generate_embedding
from bedrock_agents.retrieve_chunks import retrieve_chunks, extract_contexts_from_chunks

# Inicializa o cliente do Bedrock Agent e Bedrock Runtime
bedrock_agent_client = boto3.client('bedrock-agent-runtime')
bedrock_client = boto3.client('bedrock-runtime')

# IDs dos modelos usados no processamento
CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # Modelo de geração de texto
TITAN_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"  # Modelo de embeddings (não utilizado aqui)
KNOWLEDGE_BASE_ID = ""  # ID da base de conhecimento a ser definida

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
    query_embedding = generate_embedding(bedrock_client, user_query)

    # Busca documentos relevantes
    retrieval_results = retrieve_chunks(query_embedding)
    print("Documentos Relevantes:", retrieval_results)

    # Concatena os documentos relevantes em um único contexto
    contexts = extract_contexts_from_chunks(retrieval_results)
    print("Contextos extraídos:", contexts)

    # Constrói o prompt com base nos contextos
    prompt = create_prompt_template(user_query, contexts)

    # Gera a resposta final
    generated_text = invoke_model(bedrock_client, prompt)

    # Retorna a resposta gerada com o código de status 200
    return {'statusCode': 200, 'body': generated_text}