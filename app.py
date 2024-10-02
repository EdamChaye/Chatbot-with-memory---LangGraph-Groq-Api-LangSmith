from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver


from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition


from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

import os
from dotenv import load_dotenv

load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
groq_api_key=os.getenv("groq_api")

# Tools implementation
# Arxiv and Wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper)


wiki_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

wiki_result=wiki_tool.run("who is lionel messi")
#print(wiki_result)


arxiv_result=arxiv_tool.run("who is lionel messi")
#print(arxiv_result)

tools=[wiki_tool]

class State(TypedDict): 
    messages : Annotated[list,add_messages]


graph_builder=StateGraph(State)

llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-11b-text-preview")

llm_with_tools=llm.bind_tools(tools=tools)

memory = MemorySaver()


def chatbot(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

#chatbot node creation
graph_builder.add_node("chatbot",chatbot)

#connecting chatbot to Start node
graph_builder.add_edge(START,"chatbot")

#creating tool node
tool_node=ToolNode(tools=tools)
graph_builder.add_node("tools",tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
      
)

graph_builder.add_edge("tools","chatbot")
graph_builder.add_edge("chatbot",END)

graph=graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}


user_input="Do you remember my name? I already told you"

events=graph.stream(
    {"messages":[("user",user_input)]},config,stream_mode="values"  
    
)

for event in events: 
    event['messages'][-1].pretty_print()
    
    
    
