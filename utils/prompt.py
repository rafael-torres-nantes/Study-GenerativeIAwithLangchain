def Prompt1(name_pdf, message_pdf):
    """
    Cria um prompt detalhado para o modelo Bedrock.

    O prompt fornece instruções para gerar uma resposta humanizada em Português-Brasil, informando que não foi possível entender a solicitação do usuário e fornecendo as informações que podem ser lidas.

    Returns:
        str: O prompt formatado para ser enviado ao modelo.
    """

    prompt = f"""\n
        Você é um assistente jurídico especializado em resumir documentos legais. Utilize o documento jurídico como contexto único e extraia as informações de maneira concisa e objetiva: 
        \n NOME DO DOCUMENTO: {name_pdf} \n
        \n DOCUMENTO: {message_pdf} \n

        Baseie-se nas próximas diretrizes ao resumir o texto.

        Diretrizes:
        1. Tipo de recurso em julgamento (ex.: recurso extraordinário, agravo).
        2. Órgão julgador que proferiu o acórdão recorrido.
        3. Decisão anterior: se foi reformada ou confirmada.
        4. Votação: se o acórdão foi proferido por unanimidade ou por maioria de votos.
        5. Fundamentos apresentados pelo relator.
        6. Transcrição literal da ementa.
        7. Juízo de admissibilidade do recurso extraordinário: admissão ou inadmissão, com os fundamentos (ex.: matéria infraconstitucional, súmula 279, etc.).
        8. Dispositivo constitucional no qual o recurso extraordinário foi interposto (ex.: art. 102, III, a, b, c ou d da CF).
        9. Dispositivos legais indicados como violados e argumentos relevantes do recurso.
        10. Pedidos formulados no recurso.
        11. Contrarrazões e argumentos relevantes.

        ---

        Padrão de Retorno:
            # NOME DO COUMENTO [Extrair do texto do documento: <NOME DO DOCUMENTO:XXXXXXX>, exemplo:Acordo Recorrido, Agravo, ...] \n
            # RESUMO DO DOCUMENTO [Elabore um resumo abrangente baseado nas diretrizes no idioma português]

        ---

        Instruções Adicionais:
        - O resumo deve ser apresentado em parágrafos contínuos, incorporando naturalmente todas as diretrizes solicitadas.
        - Caso não consiga incorporar as diretrizes, faça um resumo abrangente.
        - O texto deve ter entre uma e duas páginas, utilizando linguagem clara e objetiva.
        - Sempre que mencionar legislação ou dispositivos legais, inclua a referência ao trecho correspondente.
        - Cite as páginas do documento de onde as informações foram extraídas.
        - Todas os resumos devem ser traduzidos para Português Brasil. 

        ---

        Reforço: O retorno deve ser um texto fluido e coeso, evitando a apresentação em forma de tópicos ou listas. Incorpore todas as informações de forma harmoniosa.
        """
    return prompt 