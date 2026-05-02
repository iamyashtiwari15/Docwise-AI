from langchain_core.messages import AIMessage, SystemMessage
from agents.llm_loader import get_llm

llm = get_llm()


def get_general_chat_response(messages: list) -> AIMessage:
    full_messages = [
        SystemMessage(content=(
            "You are a helpful AI assistant. "
            "You can chat naturally, explain ideas clearly, and help users reason about uploaded documents or web findings. "
            "If the user asks about a document, encourage them to upload a PDF or DOCX when needed. "
            "Do not invent facts that are not provided by the user or retrieved context. "
            "\n\nKeep responses concise, practical, and in markdown when it improves readability."
        ))
    ] + messages
    try:
        result = llm.invoke(full_messages)
        return AIMessage(content=result.content)
    except Exception as error:
        print(f"[Chat Agent Error] {error}")
        return AIMessage(content="Sorry, I encountered an issue generating a response.")
