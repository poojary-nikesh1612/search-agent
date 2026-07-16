from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def search(query: str) -> str:
    """Search information over the web

    Args:
        query:The query to be searched

    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny"


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [search]
model = create_agent(model=llm, tools=tools)


def main():
    print("Hello from search-agent!")
    query = HumanMessage(content="What is the weather in tokyo.")
    res = model.invoke({"messages": [query]})
    print(res["messages"][-1].content)


if __name__ == "__main__":
    main()
