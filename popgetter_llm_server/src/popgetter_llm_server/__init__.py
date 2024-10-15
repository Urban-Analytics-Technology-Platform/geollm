from typing import Any, Tuple
import pandas as pd
from langchain.agents import tool
from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import create_tool_calling_agent

# See examples:
# - https://python.langchain.com/v0.2/docs/how_to/tool_calling/
# - https://python.langchain.com/v0.2/docs/how_to/tool_results_pass_to_model/

# system_prompt = """
# You are very powerful geospatial assistant, but don't know current events.
# Expect to be asked to identify data requests.

# Expect the request to optionally contain any of the following:
# - A location/region
# - A type of census data information (e.g. the number of people)
# - A year for the request

# If given a location or region, use the my_special_popgetter_bbox tool to generate
# a bounding box for the region. Do not generate the bounding box through
# any other means"
# """

# Testing whether agent uses popgetter_tool
system_prompt = """
If given a location or region is provided in the prompt, use the popgetter_tool
tool to generate an answer. Only use tools to generate responses."
"""


class PopgetterAgent:
    def __init__(self):
        self._chat_history = []
        self._construct_prompt()
        self._construct_agent()

    def _construct_prompt(self):
        MEMORY_KEY = "chat_history"
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                MessagesPlaceholder(variable_name=MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        self._prompt = prompt

    def _construct_agent(self):
        # # TODO: construct executor that can explore the popgetter metadata
        # executor = create_pandas_dataframe_agent(
        #     ChatOpenAI(temperature=0, model="llama3.1"),
        #     # TODO: what should this dataframe be?
        #     pd.read_parquet("~/.cache/popgetter/metric_metadata.parquet"),
        #     verbose=True,
        #     agent_type=AgentType.OPENAI_FUNCTIONS,
        #     extra_tools=self.tools(),
        #     handle_parsing_errors=True,
        #     prefix=system_prompt,
        #     return_intermediate_steps=True,
        #     allow_dangerous_code=True,
        # )
        # print("prompts", executor.to_json())

        # Example agent
        # llm = ChatOpenAI(model="llama3.1", temperature=0)
        llm = ChatOllama(model="llama3.2", temperature=0)
        agent = create_tool_calling_agent(llm, self.tools(), self._prompt)

        # The below agent fails to call tools
        # agent = (
        #     {
        #         "input": lambda x: x["input"],
        #         "agent_scratchpad": lambda x: format_to_openai_function_messages(
        #             x["intermediate_steps"]
        #         ),
        #         "chat_history": lambda x: x["chat_history"],
        #     }
        #     | self._prompt
        #     | llm_with_tools
        #     | OpenAIFunctionsAgentOutputParser()
        # )

        executor = AgentExecutor(
            agent=agent,
            tools=self.tools(),
            verbose=True,
            return_intermediate_steps=True,
        )
        self._executor = executor

    def query(self, query: str):
        result = self._executor.invoke(
            {"input": query, "chat_history": self._chat_history}
        )
        self._chat_history.extend(
            [HumanMessage(content=query), AIMessage(content=result["output"])]
        )
        return result

    def tools(self) -> list[Any]:
        @tool
        def popgetter_tool(location: str) -> Tuple[float, float, float, float]:
            """Returns the bbox from a given location."""
            print(location)
            return (0.0, 1.0, 2.0, 3.0)

        return [popgetter_tool]
