import boto3
import boto3.session
from botocore.exceptions import ClientError
from utils.import_credentials import aws_credentials

class AWS_SERVICES:
    def __init__(self, region_name='us-east-1'):
        # Inicializa as credenciais AWS usando a função aws_credentials() que retorna ACCESS_KEY, SECRET_KEY e SESSION_TOKEN.
        self.ACESS_KEY, self.SECRET_KEY, self.SESSION_TOKEN = aws_credentials()
        
        # Armazena a região
        self.region_name = region_name

        # Cria uma sessão AWS automaticamente ao inicializar a classe.
        self.session = self.login_session_AWS()

    
    def login_session_AWS(self):
        # Função que cria uma sessão AWS usando as credenciais fornecidas (ACCESS_KEY, SECRET_KEY, SESSION_TOKEN).
        session = boto3.Session(aws_access_key_id=self.ACESS_KEY, 
                                aws_secret_access_key=self.SECRET_KEY, 
                                aws_session_token=self.SESSION_TOKEN,
                                region_name=self.region_name)
        
        # Retorna a sessão criada.
        return session
    
    def check_aws_credentials(self):
        # Verifica se as credenciais AWS são válidas usando o serviço STS (Security Token Service).
        try:
            # Cria um cliente STS com a sessão atual.
            sts_client = self.session.client('sts')
            
            # Obtém a identidade do chamador (conta, ARN, UserID) para verificar as credenciais.
            identity = sts_client.get_caller_identity()
            
            # Se bem-sucedido, imprime os detalhes da conta, UserID e ARN.
            print("AWS credentials are valid.")
            print(f"Account: {identity['Account']}, UserID: {identity['UserId']}, ARN: {identity['Arn']}")
            
            # Retorna True indicando que as credenciais são válidas.
            return True
        
        # Em caso de erro ao verificar as credenciais, captura a exceção e imprime uma mensagem de erro.
        except ClientError as e:
            print("Erro ao verificar a sessão: ", e)
            return False
    
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
    aws_utils = AWS_SERVICES()
        
    # Verifica se as credenciais são válidas.
    aws_utils.check_aws_credentials()

    # # Lista os modelos do Bedrock
    # aws_utils.list_bedrock_models()

    # aws_utils.list_models_info()
