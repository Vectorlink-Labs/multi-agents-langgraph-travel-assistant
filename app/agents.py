import logging
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from app.models import llm
from app.tools import pdf_search_tool, web_search_tool


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AgentGraph")

# ---------- State Type ----------
class GraphState(TypedDict):
   messages: List[BaseMessage]

# ---------- Tools ----------
tools = [pdf_search_tool, web_search_tool]
tool_map = {tool.name: tool for tool in tools}

# ---------- Create React Agent ----------
agent = create_react_agent(model=llm, tools=tools)

# ---------- Agent Node ----------
def agent_node(state: GraphState) -> GraphState:
   messages = state["messages"]

   try:
       result = agent.invoke({"messages": messages})
       
       # Extract the AI message from the result
       if isinstance(result, dict) and "messages" in result:
           ai_message = result["messages"][-1]
           if isinstance(ai_message, AIMessage):
               messages.append(ai_message)
           
   except Exception as e:
       messages.append(AIMessage(content="Sorry, I encountered an error. Please try again."))

   return {"messages": messages}

# ---------- Tool Node ----------
def tool_node(state: GraphState) -> GraphState:
   messages = state["messages"]
   last_msg = messages[-1]
   tool_outputs = []

   for call in getattr(last_msg, "tool_calls", []):
       tool = tool_map[call.name]
       
       # Get the query from tool call arguments
       if hasattr(call, 'args'):
           if isinstance(call.args, dict):
               query = call.args.get("query", call.args.get("__arg1", str(call.args)))
           else:
               query = str(call.args)
       else:
           query = "general search"
       
       try:
           result = tool.func(query)
           tool_outputs.append(ToolMessage(tool_call_id=call.id, content=result))
       except Exception as e:
           error_msg = f"Tool {tool.name} failed: {e}"
           tool_outputs.append(ToolMessage(tool_call_id=call.id, content=error_msg))

   messages.extend(tool_outputs)
   return {"messages": messages}

# ---------- Router ----------
def router(state: GraphState) -> str:
   last = state["messages"][-1]
   
   if isinstance(last, AIMessage):
       if getattr(last, "tool_calls", None):
           return "tools"
       elif last.content:
           return "end"
   
   return "end"


graph = StateGraph(GraphState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges("agent", router, {
   "tools": "tools",
   "end": END
})

graph.add_edge("tools", "agent")


app = graph.compile()