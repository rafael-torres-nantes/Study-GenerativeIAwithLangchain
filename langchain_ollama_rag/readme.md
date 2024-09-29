# 🤖 Criação de um Chatbot usando RAG + Ollama com Langchain

## 📚 Sumário
- [Descrição](#descrição)
- [O que são RAGs?](#o-que-são-rags)
- [Langchain](#langchain)
  - [Instalação do Langchain](#instalação-do-langchain)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Componentes e Funcionalidades](#componentes-e-funcionalidades)
  - [Instalação do Ollama](#instalação-do-ollama)
- [Configuração e Testes](#configuração-e-testes)
- [Comandos Úteis para Gerenciamento do Ollama](#comandos-úteis-para-gerenciamento-do-ollama)
- [Problemas e Soluções](#problemas-e-soluções)
- [Documentação](#documentação)

## 📝 Descrição
Este projeto visa criar um chatbot avançado utilizando a abordagem de **Retrieval-Augmented Generation (RAG)**, integrando **Ollama** e **Langchain**. O chatbot é projetado para responder a perguntas de maneira inteligente, utilizando um modelo de linguagem e uma base de dados de conhecimento consultável para fornecer respostas mais precisas e contextuais.

## ❓ O que são RAGs?
**Retrieval-Augmented Generation (RAG)** é uma técnica que combina a recuperação de informações com a geração de linguagem natural. Em vez de depender apenas de um modelo de linguagem para gerar respostas, um sistema RAG recupera informações relevantes de uma base de dados antes de gerar a resposta final. Isso permite que o chatbot forneça respostas mais informadas e específicas, utilizando dados contextuais que podem ser mais atualizados e relevantes do que o conhecimento incorporado no modelo de linguagem.

## 🔗 Langchain
[Langchain](https://langchain.readthedocs.io/) é um framework poderoso que facilita a construção de aplicações de processamento de linguagem natural (NLP). Ele fornece ferramentas para integrar modelos de linguagem, pipelines de dados e sistemas de recuperação de informações. Langchain permite criar fluxos de trabalho complexos e gerenciar interações entre diferentes componentes do sistema, tornando o desenvolvimento de chatbots e outras aplicações de NLP mais eficiente e modular.

### 📦 Instalação do Langchain
Para instalar o Langchain, basta incluir a biblioteca no seu arquivo `requirements.txt` ou instalar diretamente usando pip:
```bash
pip install langchain
```

## 🏗️ Estrutura do Projeto
A estrutura do projeto é organizada da seguinte forma:

```bash
/~root
│
├── /services
│   ├── chromadb_service.py      # Funções para interação com Chroma
│   ├── ocr_services.py          # Funções para carregar documentos
│   ├── ollama_services.py       # Funções para interagir com o Ollama
│
├── /prompts
│   ├── promptGameRules.py       # Prompt para regras do jogo
│
├── /controller
│   ├── controller_ollama.py     # Controlador que gerencia o fluxo
│
├── /embedding
│   ├── embedding_models.py       # Modelos de embedding    
│
├── main.py                       # Arquivo principal
│
└── README.md                     # Arquivo de leitura
```

## 📁 Estrutura do Repositório
O repositório contém os seguintes arquivos e pastas:
- **`src/`**: Scripts e módulos necessários para o funcionamento do chatbot.
- **`data/`**: Base de conhecimento utilizada pelo chatbot.
- **`requirements.txt`**: Lista de bibliotecas e pacotes necessários para executar o projeto.
- **`README.md`**: Documentação do projeto.

## 🛠️ Componentes e Funcionalidades
- **Chatbot**: Interage com os usuários, respondendo a perguntas com base no conhecimento disponível.
- **RAG**: Integração de técnicas de recuperação de informações para aumentar a capacidade de resposta do modelo.
- **Ollama**: Utilização da API do Ollama para facilitar a comunicação com modelos de linguagem.

### 🚀 Instalação do Ollama
1. **Instale o Ollama**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   Este comando baixa e executa o script de instalação do Ollama, configurando-o em seu sistema.

2. **Execute um modelo específico**:
   ```bash
   ollama run llama3.2
   ```
   ou
   ```bash
   ollama pull mistral
   ```
   O primeiro comando executa o modelo "llama3.2" já instalado. O segundo comando baixa o modelo "mistral" para uso futuro.

3. **Inicie o servidor do Ollama**:
   ```bash
   ollama serve
   ```
   Este comando inicia o servidor do Ollama, permitindo interagir com os modelos através de requisições.

## ⚙️ Configuração e Testes
Para configurar o ambiente e executar o projeto, siga os passos abaixo:

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/seu_usuario/chatbot-rag-ollama-langchain.git
   cd chatbot-rag-ollama-langchain
   ```

2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure a base de conhecimento**:
   Adicione suas informações à `data/knowledge_base.json` de acordo com o formato especificado.

4. **Execute o chatbot**:
   ```bash
   python main.py
   ```

## 🔧 Comandos Úteis para Gerenciamento do Ollama

1. **Verifique se o Ollama está em execução**:
   ```bash
   ps aux | grep ollama
   ```
   Este comando lista todos os processos em execução e filtra aqueles relacionados ao Ollama, permitindo verificar se o serviço está ativo.

2. **Verifique qual processo está usando a porta 50267**:
   ```bash
   lsof -i :50267
   ```
   Este comando lista todos os arquivos abertos e as conexões de rede associadas à porta 50267, ajudando a identificar qual processo está utilizando essa porta.

3. **Finalizar o processo com o PID 50267**:
   ```bash
   kill -9 50267
   ```
   Este comando força a finalização do processo com o PID especificado. O sinal `-9` indica que o processo deve ser encerrado imediatamente, sem executar rotinas de limpeza.

## ⚠️ Problemas e Soluções
Se você encontrar problemas ao executar o chatbot, considere as seguintes soluções:
- **Erro de Importação**: Verifique se todas as dependências foram instaladas corretamente.
- **Conexão com a API do Ollama**: Certifique-se de que sua chave de API está configurada corretamente no arquivo `config.py`.

## 📖 Documentação
Para mais informações sobre as bibliotecas utilizadas e conceitos envolvidos, consulte:
- [Langchain Documentation](https://langchain.readthedocs.io/)
- [Ollama API Documentation](https://ollama.com/docs)