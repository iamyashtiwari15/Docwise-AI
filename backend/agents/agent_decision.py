import logging
import os
import uuid
from agents.vision_agents.image_analysis_agent import analyze_image
from typing import TypedDict, Literal, Optional, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from agents.llm_loader import get_llm
from agents.workflow_manager import WorkflowManager
from agents.guardrails import LocalGuardrails

logging.basicConfig(level=logging.INFO) # Use INFO for cleaner production logs
llm = get_llm()
manager = WorkflowManager("agents/rag_agent/rag_db")
guard = LocalGuardrails(llm)

class GraphState(TypedDict):
    input: str
    session_id: str
    image: Optional[bytes]
    image_context: Optional[str]
    agent_name: str
    response: dict
    workflow_response: dict
    involved_agents: List[str]
    input_type: Literal["text", "image", "text_with_image"]
    bypass_guardrails: bool
    messages: List[BaseMessage]
    chat_history: Optional[List[dict]]

def guardrails_node(state: GraphState):
    if state["input_type"] != "text":
        return {**state, "bypass_guardrails": True}
    is_safe, result = guard.check_input(state["input"])
    if not is_safe:
        return {
            **state,
            "bypass_guardrails": False,
            "agent_name": "GUARDRAILS_BLOCK",
            "response": {"response": result.content, "sources": [], "confidence": 1.0},
            "involved_agents": state.get("involved_agents", []) + ["GUARDRAILS_BLOCK"],
            "messages": state.get("messages", []) + [AIMessage(content=result.content)],
        }
    return {**state, "bypass_guardrails": True}

def image_detection_router(state: GraphState):
    if state.get("image"):
        logging.info("Image detected. Routing to ImageAnalysis.")
        return "handle_image"
    else:
        logging.info("No image detected. Routing directly to ContextMerger.")
        return "handle_text_only"

def image_analysis_node(state: GraphState):
    image_bytes = state.get("image")
    if not image_bytes:
        return state

    temp_filename = f"temp_{uuid.uuid4().hex}.png"
    try:
        with open(temp_filename, "wb") as f:
            f.write(image_bytes)
        result = analyze_image(temp_filename)
        
        image_context = result.get("ocr_summary") or result.get("ocr_text")
        if "❌" in (image_context or ""):
             logging.warning(f"Image analysis returned an error: {image_context}")
        elif not image_context:
            image_context = "No relevant text was found in the image."

    except Exception as e:
        logging.error(f"Image analysis failed critically: {e}")
        image_context = "❌ Error: The image file could not be processed."
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return {**state, "image_context": image_context}

def context_merger_node(state: GraphState):
    user_text = state.get("input", "")
    image_context = state.get("image_context", "")

    if image_context and "❌" not in image_context:
        merged_input = f"User Query: {user_text}\n\nImage Context:\n{image_context}"
    else:
        merged_input = user_text

    return {**state, "input": merged_input}

def route_to_agent(state: GraphState):
    if state.get("agent_name") == "GUARDRAILS_BLOCK":
        return state
        
    chat_history = state.get("chat_history", None)
    image_context = state.get("image_context", None)
    result = manager.process_query(
        state["input"],
        chat_history=chat_history,
        image_context=image_context,
        session_id=state.get("session_id"),
    )
    
    if result.get("query_type") == "general":
        agent_name = "CONVERSATION_AGENT"
    else:
        agent_name = "DOCUMENT_AGENT"
        
    logging.info(f"WorkflowManager selected agent: {agent_name}")
    return {**state, "agent_name": agent_name, "workflow_response": result}

def call_agent(state: GraphState):
    if state.get("agent_name") == "GUARDRAILS_BLOCK":
        return state

    workflow_response = state.get("workflow_response", {})
    agent_name = state.get("agent_name")
    image_context = state.get("image_context", "")

    # Route greeting/chitchat to general chat agent
    if agent_name == "CONVERSATION_AGENT":
        from agents.general_chat_agent import get_general_chat_response
        ai_message = get_general_chat_response(state["messages"])
        workflow_response["response"] = ai_message.content
        updated_agents = state.get("involved_agents", []) + [agent_name]
        messages = state["messages"] + [ai_message]
        return {
            **state,
            "response": workflow_response,
            "involved_agents": updated_agents,
            "messages": messages,
        }

    if image_context and "❌" in image_context:
        original_response = workflow_response.get("response", "I was unable to provide a response.")
        workflow_response["response"] = f"{image_context}\n\nRegarding your text query: {original_response}"

    messages = state["messages"] + [AIMessage(content=workflow_response.get("response", ""))]
    updated_agents = state.get("involved_agents", []) + [agent_name]

    return {
        **state,
        "response": workflow_response,
        "involved_agents": updated_agents,
        "messages": messages,
    }

# --- Correct Graph Building with Conditional Edges ---
def build_assistant_graph():
    builder = StateGraph(GraphState)

    # Add all the nodes
    builder.add_node("Guardrails", guardrails_node)

    def image_detection_node(state):
        return state
    builder.add_node("ImageDetection", image_detection_node)
    
    builder.add_node("ImageAnalysis", image_analysis_node)
    builder.add_node("ContextMerger", context_merger_node)
    builder.add_node("RouteAgent", route_to_agent)
    builder.add_node("CallAgent", call_agent)

    # Entry point
    builder.set_entry_point("Guardrails")

    # Static edges
    builder.add_edge("Guardrails", "ImageDetection") # This now correctly points to the router node
    builder.add_edge("ImageAnalysis", "ContextMerger")
    builder.add_edge("ContextMerger", "RouteAgent")
    builder.add_edge("RouteAgent", "CallAgent")
    builder.add_edge("CallAgent", END)

    # Add the conditional edge for routing based on image presence
    builder.add_conditional_edges(
        "ImageDetection", # The source node that makes the decision
        image_detection_router, # The function that returns the decision string
        {
            "handle_image": "ImageAnalysis", # If it returns "handle_image", go to this node
            "handle_text_only": "ContextMerger", # If it returns "handle_text_only", go here
        }
    )

    return builder.compile()

assistant_graph = build_assistant_graph()
