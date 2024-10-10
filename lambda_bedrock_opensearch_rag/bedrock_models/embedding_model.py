import json

# IDs dos modelos usados no processamento
TITAN_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"  # Modelo de embeddings

# --------------------------------------------------------------------
# Função para gerar embeddings usando o Amazon Titan V2
# --------------------------------------------------------------------
def generate_embedding(bedrock_client, user_query):
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