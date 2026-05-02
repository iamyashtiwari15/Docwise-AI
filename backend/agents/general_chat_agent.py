from langchain_core.messages import AIMessage, SystemMessage
from agents.llm_loader import get_llm

llm = get_llm()


def get_general_chat_response(messages: list) -> AIMessage:
    full_messages = [
        SystemMessage(content=(
            "You are a knowledgeable and articulate AI assistant.\n\n"
            "Your behaviour:\n"
            "- Explain concepts clearly with examples when helpful.\n"
            "- When a user references an uploaded document, reason about it accurately — do not invent details that were not in the conversation.\n"
            "- When a user asks about a document and none has been uploaded, suggest they upload a PDF, DOCX, or TXT file.\n"
            "- Format responses in markdown (headings, lists, code blocks) when that improves readability.\n"
            "- Give complete and useful answers. Never truncate a useful answer for the sake of brevity."
        ))
    ] + messages
    try:
        result = llm.invoke(full_messages)
        return AIMessage(content=result.content)
    except Exception as error:
        print(f"[Chat Agent Error] {error}")
        return AIMessage(content="Sorry, I encountered an issue generating a response.")
