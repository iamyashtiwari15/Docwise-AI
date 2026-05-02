# --- Image OCR and summarization for downstream routing ---
import os
from PIL import Image as PILImage
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate


load_dotenv()

# LLM for summarization
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
)

TEXT_SUMMARY_PROMPT = (
    "You are a workflow assistant. Your task is to read extracted text from an image and summarize the useful context for downstream question answering. Provide a clear, structured summary in bullet points, including:\n"
    "- All relevant facts, dates, and parties involved\n"
    "- The main issue, request, or topic described\n"
    "- Any terms, conditions, policies, or disclaimers mentioned\n"
    "- Important entities, deadlines, or numeric values\n"
    "- Recommended next steps for the user\n"
    "- Any information that should influence the chatbot's workflow or response\n\n"
    "Extracted text:\n{extracted_text}"
)

def analyze_image(filepath: str) -> dict:
    """
    Extract text from an image, summarize it, and forward to workflow manager for context setting.
    Returns a dict with keys: 'ocr_text', 'ocr_summary', 'workflow_response'
    """
    result = {
        "ocr_text": None,
        "ocr_summary": None,
        "workflow_response": None
    }
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        image = PILImage.open(filepath)
        if image.format not in ["JPEG", "PNG", "BMP", "GIF"]:
            result["ocr_text"] = "❌ Unsupported image format. Please upload JPG, PNG, BMP, or GIF."
            return result
        if os.path.getsize(filepath) > 10 * 1024 * 1024:
            result["ocr_text"] = "❌ Image too large. Please upload an image smaller than 10MB."
            return result
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        # --- OCR Text Extraction ---
        ocr_text = pytesseract.image_to_string(image)
        result["ocr_text"] = ocr_text.strip()

        # --- Summarize OCR Text ---
        if ocr_text.strip():
            summary_prompt = TEXT_SUMMARY_PROMPT.format(extracted_text=ocr_text)
            summary_chain = ChatPromptTemplate.from_messages([("human", summary_prompt)]) | llm | StrOutputParser()
            ocr_summary = summary_chain.invoke({"input": summary_prompt})
            result["ocr_summary"] = ocr_summary.strip()


        return result

    except PILImage.UnidentifiedImageError:
        result["ocr_text"] = "❌ The uploaded file is not a valid image."
        return result
    except Exception as e:
        result["ocr_text"] = f"⚠️ Error analyzing image: {str(e)}"
        return result
