"""
# Simple bot

Objectives:

    1. Define state structure with a list of HumanMessage objects.

    2. Initialize a GPT-4o model using LangChain's ChatOpenAl

    3. Sending and handling different types of messages

    4. Building and compiling the graph of the Agent
    
Main Goal: How to integrate LLMs in our Graphs
"""


from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values

load_dotenv()

class AgentState(TypedDict):
    messages : List[HumanMessage]
    
llm = ChatOpenAI(model = "gpt-4o")

def process(state: AgentState)->AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI:{response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")