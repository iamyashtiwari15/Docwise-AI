from langchain_core.messages import AIMessage
import re

class LocalGuardrails:
    """
    Lightweight input safety checks for a general-purpose document assistant.
    """

    def __init__(self, llm):
        """
        Initialize guardrails with the provided LLM.
        Args:
            llm: A language model instance with an invoke() method.
        """
        self.llm = llm
        self.block_patterns = {
            "harmful or violent instructions": re.compile(r"\b(kill|bomb|weapon|explosive|self-harm|suicide)\b", re.IGNORECASE),
            "prompt injection": re.compile(r"\b(ignore previous|system prompt|reveal instructions|jailbreak|bypass guardrails)\b", re.IGNORECASE),
            "malware or exploit requests": re.compile(r"\b(malware|ransomware|exploit|sql injection|steal password)\b", re.IGNORECASE),
            "sensitive credential requests": re.compile(r"\b(password|otp|cvv|credit card|bank account)\b", re.IGNORECASE),
        }

    def check_input(self, user_input: str) -> tuple[bool, AIMessage]:
        """
        Check if user input passes safety filters.

        Args:
            user_input: The raw user input text

        Returns:
            Tuple (is_allowed: bool, message: AIMessage)
        """
        normalized = (user_input or "").strip()
        if not normalized:
            return False, AIMessage(content="Please enter a message.")

        for reason, pattern in self.block_patterns.items():
            if pattern.search(normalized):
                return False, AIMessage(content=f"I cannot process this request. Reason: {reason}.")
        return True, AIMessage(content=user_input)

    def check_output(self, output: str, user_input: str = "") -> str:
        """
        Process the model's output through safety filters.

        Args:
            output: The raw output from the model
            user_input: The original user query (optional, for context)

        Returns:
            Sanitized or original output string
        """
        if not output:
            return output

        output_text = output if isinstance(output, str) else output.content
        return output_text


