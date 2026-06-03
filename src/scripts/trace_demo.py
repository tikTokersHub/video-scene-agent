import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_core.messages import HumanMessage, SystemMessage
from video_agent.agent import build_agent, SYSTEM_PROMPT

QUESTION = "When did the bicycle anomaly happen, and what was happening just before it?"
VIDEO_ID = "shanghai_01_0014"

agent = build_agent()
# trigger model load up-front so it doesn't interleave with the trace
from video_agent.tools import get_ingester
get_ingester()
print("\n" + "=" * 60)
print("AGENT REASONING TRACE")
print("=" * 60 + "\n")

state = {
    "messages": [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"{QUESTION}\n\nUse video_id='{VIDEO_ID}' in tool calls."),
    ],
    "iterations": 0,
}

for chunk in agent.stream(state, stream_mode="updates"):
    for node, update in chunk.items():
        msg = update["messages"][-1]
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"  ├─ [THINK] call {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            preview = (msg.content or "")[:110].replace("\n", " ")
            print(f"  ├─ [OBSERVE] {msg.name} → {preview}…")
        elif msg.content:
            print(f"[MODEL] {msg.content[:500]}")