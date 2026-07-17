from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()

search = TavilySearch()

# @tool
# def search(query: str) -> str:
#     """Search information over the web

#     Args:
#         query:The query to be searched

#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return "Weather is sunny"


class Source(BaseModel):
    """Schema for source used by agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answers: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
model = create_agent(model=llm, tools=[search], response_format=AgentResponse)


def main():
    print("Hello from search-agent!")
    query = HumanMessage(
        content="search for 3 job postings for an ai engineer using langchain in india on linkedin and list their details?"
    )
    res = model.invoke({"messages": [query]})
    print(res["messages"][-1].content)


if __name__ == "__main__":
    main()
