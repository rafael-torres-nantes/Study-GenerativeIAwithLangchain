import json
import boto3
from botocore.exceptions import ClientError
from prompts.promptSummarizeLegalText import PromptSummarizeLegalText

"""
Caso for testar o Bedrock veja se está habilitado o modelo no AWS Bedrock (https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess)
"""

class BedrockService:
    def __init__(self):
        """
        Inicializa o serviço AWS Bedrock.

        Cria uma sessão do Boto3 e um cliente para o serviço Bedrock, com a região configurada como 'us-east-1'.
        """
        # Inicia a sessão do Boto3
        self.session = boto3.Session(region_name='us-east-1')

        # Inicia o serviço Bedrock
        self.bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
               
    def set_document(self, pdf_name, msg):
        """
        Define a mensagem a ser usada pelo serviço de geração de texto.

        Parameters:
            msg (str): Mensagem que descreve a raça do animal de estimação.

        Returns:
            bool: Retorna True após definir a mensagem.
        """
        self.name_pdf = pdf_name
        self.message_pdf = msg
        return True

     
    def generate_request_body(self):
        """
        Gera o corpo da requisição para enviar ao modelo Bedrock.

        Inclui o prompt e configurações de geração de texto como o número máximo de tokens, temperatura e topP.

        Returns:
            str: O corpo da requisição em formato JSON.
        """
        request_body = {
            "anthropic_version" : 'bedrock-2023-05-31',
            "max_tokens": 60000,
            "temperature": 0.2,         # temperature: aleatoriedade na geração de texto (quanto maior, mais aleatório e menos conservador o texto é)
            "top_p": 0.3,                 # topP: tokens que compõem o top p% da probabilidade cumulativa
            'messages' : [
                            {
                            'role' : 'user', 
                            'content': [{'type' : 'text', 'text' : PromptSummarizeLegalText(self.name_pdf, self.message_pdf)}]
                            }
                         ]  
        }
        return json.dumps(request_body)

    def invoke_model(self):
        """
        Invoca o modelo Bedrock com o corpo da requisição gerado.

        Configura os parâmetros de invocação, incluindo o ID do modelo, o tipo de conteúdo e o corpo da requisição. Processa a resposta do modelo e retorna a resposta formatada.

        Returns:
            dict: Resposta formatada com o código de status e o texto gerado pelo modelo.
        """
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

        try:
            # Invoca o modelo com o corpo da requisição gerado
            response = self.bedrock.invoke_model_with_response_stream(
                modelId=model_id, 
                contentType='application/json',
                accept="*/*",
                body=self.generate_request_body()
            )
            
            output_text = ""
            for event in response.get("body"):
                chunk = json.loads(event["chunk"]["bytes"])

                # if chunk['type'] == 'message_delta':
                #     print(f"\nStop reason: {chunk['delta']['stop_reason']}")
                #     print(f"Stop sequence: {chunk['delta']['stop_sequence']}")
                #     print(f"Output tokens: {chunk['usage']['output_tokens']}")

                if chunk['type'] == 'content_block_delta':
                    if chunk['delta']['type'] == 'text_delta':
                        output_text += f"{chunk['delta']['text']}"
            
            # # Processa a resposta do modelo
            # model_response = json.loads(response["body"].read().decode('utf-8'))
            # response_text = model_response["results"][0]["outputText"]

            # Retorna a resposta formatada
            return {'statusCode': 200, 'body': output_text}
        
        except ClientError as e:
            print(f"Error invoking model: {e}")
            return {'statusCode': 500, 'body': json.dumps(str(e))}