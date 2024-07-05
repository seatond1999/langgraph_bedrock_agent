# %% --------------------------------------------------------------------------
from dotenv import load_dotenv
import os
import string
import operator
from typing import Annotated, TypedDict
import uuid
import boto3
from langchain.agents import tool
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import BedrockChat
from langchain_core.agents import AgentAction
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langsmith import traceable

load_dotenv(".env")
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")

# %% --------------------------------------------------------------------------


class AgentState(TypedDict):
    intermediate_steps: Annotated[list[AnyMessage], operator.add]
    chat_history: ConversationBufferWindowMemory
    user_query: str


@traceable
class Agent:
    def __init__(
        self,
        model: BedrockChat,
        tools: list[StructuredTool],
        prompt: str,
    ):
        self.prompt = prompt
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_bedrock)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.tools_str = self.stringify_tools(tools)
        self.tools = {t.name: t for t in tools}
        self.graph = graph.compile()
        self.llm_chain = self.make_llm_chain(model)

    def exists_action(self, state: AgentState):
        return isinstance(state["intermediate_steps"][-1], AgentAction)

    def call_bedrock(self, state: AgentState):
        user_query = "User: " + state["user_query"]
        chat_history = self.stringify_messages(state["chat_history"])
        intermediate_steps = self.stringify_intermediate_steps(
            state["intermediate_steps"]
        )

        messages = {
            "prompt": self.prompt,
            "tools": self.tools_str,
            "chat_history": chat_history,
            "user_query": user_query,
            "scratchpad": intermediate_steps,
        }
        message = self.llm_chain.invoke(messages)
        return {"intermediate_steps": [message]}

    def take_action(self, state: AgentState):
        tool_call = state["intermediate_steps"][-1]
        print(f"Calling: {tool}")
        if tool_call.tool not in self.tools:
            print("\n ....bad tool name....")
            result = "bad tool name, retry"
        else:
            result = self.tools[tool_call.tool](tool_call.tool_input)
            result = ToolMessage(
                content=result, tool_call_id="call_" + str(uuid.uuid4())
            )

        print("Back to the model!")
        return {"intermediate_steps": [result]}

    def make_llm_chain(self, model):
        return (
            RunnableLambda(
                lambda x: x["prompt"].format(
                    tools=x["tools"],
                    chat_history=x["chat_history"],
                    user_query=x["user_query"],
                    scratchpad=x["scratchpad"],
                )
            )
            | model.bind(stop=["</tool_input>", "</final_answer>"])
            | XMLAgentOutputParser()
        )

    def stringify_intermediate_steps(
        self, intermediate_steps: list[AgentAction, ToolMessage]
    ) -> str:
        log = []
        for i in intermediate_steps:
            if isinstance(i, AgentAction):
                log.append(
                    f"<tool>{i.tool}</tool><tool_input>{i.tool_input}</tool_input>"
                )
            elif isinstance(i, ToolMessage):
                log.append(f"<observation>{i.content}</observation>")
            else:
                raise Exception(f"wrong response type from the llm chain: {type(i)}")
        return "".join(log)

    def stringify_messages(
        self, messages: list | ConversationBufferWindowMemory
    ) -> str:
        messages = messages.chat_memory.messages

        memory_list = [
            (
                f"User: {mem.content}"
                if isinstance(mem, HumanMessage)
                else f"Agent: {mem.content}"
            )
            for mem in messages
        ]
        return "\n".join(memory_list)

    def stringify_tools(self, tools) -> str:
        return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])


# %% --------------------------------------------------------------------------


def get_llm() -> BedrockChat:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_ACCESS_KEY,
    )

    model_kwargs = {
        "max_tokens": 1000,
        "temperature": 0.1,
    }

    model = BedrockChat(
        client=bedrock_runtime,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs=model_kwargs,
    )

    return model


@tool
def get_forename(input: str) -> str:
    """Return the forename of the user's favourite tennis player. The input to this function is the string "forename"."""
    return "Andy"


@tool
def get_surname(input: str) -> str:
    """Return the surname of the user's favourite tennis player. The input to this function is the string "surname"."""
    return "Murray"


@tool
def get_name_alphabet_positions(full_name: str) -> list[int]:
    """Returns the alphabet positions of the user's favourite tennis player. The input to this function must be the first name, second name or full name of the tennis player which you should retrieve from a Previous Conversation, or using the get_forename and/or the get_surname function."""
    name_alphabet_positions = []
    alphabet_positions = {
        letter: position + 1 for position, letter in enumerate(string.ascii_lowercase)
    }
    alphabet_positions[" "] = 0
    for i in full_name:
        name_alphabet_positions.append(alphabet_positions[i.lower()])
    return name_alphabet_positions


def get_tools() -> list[StructuredTool]:
    return [get_forename, get_surname, get_name_alphabet_positions]


def get_prompt() -> str:
    return """You are a helpful agent. You will use tools which retrieve information about the user's company and projects they have completed. You must use this information only to answer the user's questions.
    
    You have access to the following tools:
    
    {tools}
    
    You must enclose your answer between <final_answer></final_answer>

    In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>
    For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:
    
    <tool>search</tool><tool_input>weather in SF</tool_input>
    <observation>64 degrees</observation>
    
    When you are done, respond with a final answer between <final_answer></final_answer>. For example:
    
    <final_answer>The weather in SF is 64 degrees</final_answer>

    Begin!
    
    Previous Conversation:
    {chat_history}
    
    New Conversation:
    {user_query}
    {scratchpad}
    """


def invoke_agent(
    user_query: str, conversational_memory: ConversationBufferWindowMemory
) -> str:
    result = abot.graph.invoke(
        {
            "user_query": user_query,
            "chat_history": conversational_memory,
        }
    )
    answer = result["intermediate_steps"][-1].return_values["output"]
    conversational_memory.chat_memory.add_user_message(user_query)
    conversational_memory.chat_memory.add_ai_message(answer)
    return answer


# %% --------------------------------------------------------------------------

if __name__ == "__main__":
    conversational_memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=3, return_messages=True
    )
    abot = Agent(get_llm(), get_tools(), get_prompt())
    user_query = "What is my favourite tennis player's first name?"
    answer = invoke_agent(user_query, conversational_memory)
    user_query = "What are the alphabet positions of my favourite tennis player's full name?"
    answer = invoke_agent(user_query, conversational_memory)
