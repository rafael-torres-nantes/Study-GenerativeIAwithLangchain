def promptAskGameRules():

    PROMPT_TEMPLATE = """
    Você é um assistente virtual de jogos de tabuleiros. O seu dever é responder a questão do usuário baseado somente no seguinte contexto:

    {context}

    ---
    Responda a questão de forma clara, em português, baseada no contexto acima. Pense no passo a passo segundos as dicas.
    --- 
    {question} 
    """

    return PROMPT_TEMPLATE