from ..services.bedrock_claude import BedrockService
from ..utils.check_aws import AWS_SERVICES

class Controller:
    def __init__(self):
        """
        Inicializa o controlador, que gerencia a interação com os serviços AWS e Bedrock.
        Verifica se as credenciais AWS são válidas ao instanciar os serviços.
        """
        self.aws_service_manager = AWS_SERVICES()  # Gerenciador de serviços AWS
        self.bedrock_service = BedrockService()  # Serviço de processamento de documentos via Bedrock
        self.aws_session_instance = self.aws_service_manager.session  # Sessão AWS inicializada
        self.credentials_valid = self.aws_service_manager.check_aws_credentials()  # Valida as credenciais AWS

        if not self.credentials_valid: 
            raise ValueError('Erro: As credenciais fornecidas são inválidas. Verifique suas credenciais e tente novamente.')
        

    def process_documents(self, extract_text):
        """
        Processa o texto extraído de documentos usando o serviço Bedrock e AWS Lex.
        
        Parâmetros:
            extract_text (str): Texto extraído de documentos PDF ou outras fontes.
        
        Exceções:
            Levanta ValueError se as credenciais da AWS forem inválidas.
            Levanta uma exceção genérica caso o processamento com Bedrock/Lex falhe.
        """
        if not self.credentials_valid: 
            raise ValueError('Erro: As credenciais fornecidas são inválidas. Verifique suas credenciais e tente novamente.')
        
        try:
            # Define a intenção do processamento do texto usando AWS Lex ou outro serviço apropriado
            self.bedrock_service.set_document(extract_text['filename'], extract_text['content_text'])
        
        except Exception as e:
            raise ValueError(f"Erro ao processar o documento: {e}")

    def execute_bedrock_model(self):
        """
        Executa o modelo de processamento Bedrock e retorna a resposta do serviço.
        
        Retorna:
            response (dict): Resposta do modelo Bedrock após a execução.
        
        Exceções:
            Levanta ValueError se as credenciais da AWS forem inválidas.
            Levanta uma exceção genérica se a execução do modelo Bedrock falhar.
        """
        if not self.credentials_valid: 
            raise ValueError('Erro: As credenciais fornecidas são inválidas. Verifique suas credenciais e tente novamente.')
        
        try:
            # Chama o serviço Bedrock para executar o modelo de processamento
            response = self.bedrock_service.invoke_model()

            output_text = response['body']
            return output_text
        
        except Exception as e:
            raise ValueError(f"Erro ao executar o modelo do Bedrock: {e}")
