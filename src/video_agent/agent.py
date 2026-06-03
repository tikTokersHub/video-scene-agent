from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode

from video_agent.schemas import AgentAnswer
from video_agent.tools import (
    check_against_rules,
    compare_two_timestamps,
    get_neighbouring_frames,
    search_scenes_by_text,
    search_scenes_by_time_range,
    compare_time_interval
)


load_dotenv()

TOOLS = [
    search_scenes_by_text,
    search_scenes_by_time_range,
    get_neighbouring_frames,
    check_against_rules,
    compare_two_timestamps,
    compare_time_interval
]

SYSTEM_PROMPT = """
You are a video scene analysis assistant.

You have access to tools that search a memory of surveillance video scenes.

Rules:
1. Use search_scenes_by_text for semantic/descriptive questions.
2. For questions asking whether an activity happened, retrieve relevant scenes first.
3. After retrieving scenes, if the activity could be normal or anomalous, call check_against_rules using the most relevant retrieved caption.
4. Use search_scenes_by_time_range for questions about a specific time window.
5. Use get_neighbouring_frames when checking temporal continuity.
6. Use compare_two_timestamps when comparing two different moments.
7. Use compare_time_interval when the question depends on duration, persistence, continuity across a period, or whether something was left unattended for long enough.
8. never claim something happened unless the tools returned supporting evidence.
8. Always mention frame indices and timestamps when evidence is available.
9. If evidence is weak or missing, say you are uncertain.

Important:
- If the retrieved evidence mentions riding a bike, riding a bicycle, running, fighting, lying on the ground, skateboarding, or vandalising, you must call check_against_rules before giving the final answer.
- The final classification must be based on retrieved evidence plus rule checking.
"""



class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iterations: int

def build_agent():
    """
    Build the LangGraph cyclic agent.

    Flow:
    user question
        -> LLM decides whether to call tools
        -> tools return observations
        -> LLM reasons again
        -> repeat until no more tool calls or max iterations reached
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    ).bind_tools(TOOLS)

    tool_node = ToolNode(TOOLS)

    def call_model(state:AgentState) -> dict:
        messages = state['messages']
        response = llm.invoke(messages)
        return {
            "messages": [response],
            "iterations": state["iterations"] + 1,
        }
    
    def should_continue(state:AgentState) -> str:
        last_message = state["messages"][-1]

        if state["iterations"] >= 8:
            return "end"
        
        if getattr(last_message, "tool_calls", None):
            return "tools"
        
        return "end"
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    graph.add_edge("tools", "agent")

    return graph.compile()
    

def ask(question: str, video_id: str | None = None) -> AgentAnswer:
    """
    Ask a question about the video and return a structured AgentAnswer.
    """

    agent = build_agent()

    scoped_question = question
    if video_id:
        scoped_question = (
            f"{question}\n\n"
            f"Use video_id='{video_id}' in scene search, neighbouring-frame, "
            "and rule-checking tool calls."
        )

    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=scoped_question),
            ],
            "iterations": 0,
        }
    )

    structurer = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    ).with_structured_output(AgentAnswer)

    trace_text = "\n\n".join(
        f"{message.type.upper()}:\n{message.content}"
        for message in result["messages"]
        if getattr(message, "content", None)
    )

    final_answer = structurer.invoke(
        f"""
Convert the following agent trace into a structured final answer.

User question:
{question}

Video ID:
{video_id or "not specified"}

Agent trace:
{trace_text}

Requirements:
- The answer must be grounded only in retrieved tool evidence.
- If no strong evidence exists, use classification='uncertain'.
- Include frame_idx, timestamp_sec, caption, similarity_score, and video_id where possible.
- confidence must be between 0 and 1.
"""
    )

    return final_answer


