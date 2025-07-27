from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

client = Client()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. {instructions}"),
    ("user", "{query}")
])

client.push_prompt("kb_lease_doc_agent", object=prompt)