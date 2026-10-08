class HRPrompts:
    """
    HRPrompts is a static class for generating system and user prompts for the HR Assistant RAG chatbot.

    It ensures consistent, compliant, and human-like LLM behavior by:
    - Enforcing rules like no hallucination, no markdown formatting, and empathy in tone.
    - Generating plain text responses that resemble real HR communication.
    - Structuring prompt injection-resistant behavior for all model calls.

    Usage:
        system_prompt = HRPrompts.get_system_prompt()
        user_prompt = HRPrompts.get_user_prompt(context, query)

    Other options:
        - Allow per-domain prompts: HR, Legal, Finance.
        - Customize for language or regional tone (US/UK HR policies).
        - Dynamically inject few-shot examples or templates.
        - Use a prompt templating engine like `jinja2` or `PromptTools`.
        - Store prompt templates in a DB or JSON and dynamically modify based on task.
        - Allow `get_user_prompt()` to optionally include prior turns (multi-turn memory).
    """

    @staticmethod
    def get_system_prompt():
        """
        Returns the system prompt for the HR assistant chatbot.

        Purpose:
            - Instructs the model to behave like a professional HR assistant.
            - Prevents use of external knowledge.
            - Guides tone, structure, and safety behaviors.

        Returns:
            str: System prompt string.

        Key Instructions:
            - No hallucination or guessing.
            - Professional, empathetic, and concise.
            - Plain text only (no markdown).
            - Summarize multiple options clearly.
        """

        system_prompt = """
You are a highly knowledgeable, professional, and empathetic HR Assistant. 
Answer user queries ONLY using the information provided in the context section.

Follow these instructions strictly:
1. Never answer using external knowledge; rely only on the context provided.
2. If the answer is not in the context, reply with: "Sorry, the information is not available."
3. Do not invent, hallucinate, or guess details beyond what is present in the context.
4. Avoid generic phrases like 'As an AI language model'; always respond as a real HR assistant.
5. Write in full sentences, using a professional, concise, and friendly tone.
6. Structure your answer with short, clear paragraphs or bullet points (but do NOT use markdown formatting).
7. If multiple policies or options are relevant, summarize them clearly in plain text, separated by line breaks.
8. If a question involves dates, eligibility, or exceptions, highlight these details if they exist in context.
9. Never repeat the user's question in your answer; respond directly with the information.
10. Do NOT use any markdown, code blocks, or formatting symbols like *, #, -, >, or ```. Only provide clean plain text answers.
11. If the user asks about escalation or grievances, mention the official HR process if present.
12. Always maintain a polite, supportive, and trustworthy style.
"""
        return system_prompt

    @staticmethod
    def get_user_prompt(context, query):
        """
        Constructs the user prompt by injecting context and question.

        Args:
            context (str): The retrieved relevant document chunks.
            query (str): The user's actual question.

        Returns:
            str: A formatted prompt to send to the LLM, combining context and query.

        Notes:
            - This prompt assumes a single-turn format.
            - For multi-turn memory, consider appending prior user/assistant turns above the current query.
        """
        user_prompt = f"""
        Use below context to provide output:
        Context:
        {context}

        User: {query}

        OUTPUT:
        Assistant:
        """
        return user_prompt
