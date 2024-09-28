import boto3
import boto3.session
from botocore.exceptions import ClientError
from utils.import_credentials import aws_credentials

class BEDROCK_UTILS:
    def __init__(self, region_name='us-east-1'):
        # Inicializa as credenciais AWS usando a função aws_credentials() que retorna ACCESS_KEY, SECRET_KEY e SESSION_TOKEN.
        self.ACESS_KEY, self.SECRET_KEY, self.SESSION_TOKEN = aws_credentials()
        
        # Armazena a região
        self.region_name = region_name

    def list_bedrock_models(self):
        try:
            # Cria um cliente Bedrock com a sessão atual
            bedrock_client = self.session.client('bedrock')

            # Chama a operação para listar modelos
            response = bedrock_client.list_foundation_models()

            # Processa e imprime as informações dos modelos
            for model in response.get('modelSummaries', []):
                print(f"Model ARN: {model['modelArn']}")
                print(f"Model ID: {model['modelId']}")
                print(f"Model Name: {model['modelName']}")
                print(f"Provider Name: {model['providerName']}")
                print(f"Input Modalities: {', '.join(model['inputModalities'])}")
                print(f"Output Modalities: {', '.join(model['outputModalities'])}")
                print(f"Customizations Supported: {', '.join(model.get('customizationsSupported', []))}")
                print(f"Inference Types Supported: {', '.join(model.get('inferenceTypesSupported', []))}")
                print(f"Model Status: {model['modelLifecycle']['status']}")
                print("-" * 40)

        except ClientError as e:
            print("Erro ao listar modelos do Bedrock: ", e)

    def list_models_info(self):
        try:
            # Cria um cliente Bedrock com a sessão atual
            bedrock_client = self.session.client('bedrock')

            # Chama a operação para listar modelos
            response = bedrock_client.list_foundation_models()

            # Extrai metadados da resposta
            content_length = response['ResponseMetadata']['HTTPHeaders']['content-length']
            content_type = response['ResponseMetadata']['HTTPHeaders']['content-type']

            print(f"Content Length: {content_length}")
            print(f"Content Type: {content_type}")

            # Processa e imprime as informações dos modelos
            for model in response.get('modelSummaries', []):
                if model['modelLifecycle']['status'] == 'ACTIVE':
                    model_id = model['modelId']

                    print(f"Model ID: {model_id} is ACTIVE")
                    
                    # Obtém detalhes do modelo ativo
                    model_details_response = bedrock_client.get_foundation_model(modelIdentifier=model_id)
                    
                    # Extrai o content-length e content-type da resposta de get_foundation_model
                    model_content_length = model_details_response['ResponseMetadata']['HTTPHeaders']['content-length']
                    model_content_type = model_details_response['ResponseMetadata']['HTTPHeaders']['content-type']

                    print(f"Content Length for {model_id}: {model_content_length}")
                    # print(f"Content Type for {model_id}: {model_content_type}")
                    
                    # Aqui você pode imprimir ou processar as informações detalhadas do modelo
                    model_details = model_details_response['modelDetails']
                    # print(f"Details for {model_id}: {model_details}")

                    print("-" * 40)

        except ClientError as e:
            print("Erro ao listar ou obter modelos do Bedrock: ", e)

# Bloco principal que será executado ao rodar o script.
if __name__ == "__main__":

    # Inicializa a classe AWS_SERVICES, o que automaticamente cria uma sessão.
    bedrock_utils = BEDROCK_UTILS()

    # Lista os modelos do Bedrock
    bedrock_utils.list_bedrock_models()

    # Lista informações dos modelos
    bedrock_utils.list_models_info()
