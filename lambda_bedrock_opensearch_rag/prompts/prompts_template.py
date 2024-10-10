# --------------------------------------------------------------------
# Função que constrói o prompt com base no contexto e na pergunta
# --------------------------------------------------------------------
def get_prompt():
    """
    Constrói o prompt genérico a ser usado no modelo.

    :return: String formatada como prompt para o modelo.
    """
    prompt = """
    $search_results$
    
    $output_format_instructions$
    """
    return prompt

# --------------------------------------------------------------------
# Função que constrói o template de prompt para o modelo de geração
# --------------------------------------------------------------------
def create_prompt_template(query, contexts):
    """
    Constrói o prompt com base no contexto e na pergunta do usuário.
    
    :param contexts: Textos recuperados dos documentos relevantes.
    :param query: Pergunta do usuário.
    :return: String formatada como prompt para o modelo.
    """
    prompt = f"""
    <context>
    {contexts}
    </context>

    <question>
    {query}
    </question>
    """
    return prompt