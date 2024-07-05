## Overview

The purpose of this repository is to create an agent using LangGraph and the AWS Bedrock API.

At the time of writing, the AWS Bedrock API does not support tool calling and structured output. This makes the LangGraph agent setup more complex, because the Bedrock API expects a string and returns a string. Therefore, the LangChain function bind_tools() is not compatible with the Bedrock API.

To get round this, we must add methods to the agent class which stringify intermediate steps and chat history. We also implement the LLM with a chain of runnables which includes a parser to parse the LLM output into a structured output.

Since the time of writing, AWS has released the Converse API, which does support tool calling and structured ouptut. This is covered ....

## Dependencies

To install the required dependencies, you can use the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

Alternatively, using poetry: 
```sh
poetry install
```