import pandas as pd
from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents import create_pandas_dataframe_agent

file_path = 'knowledge_base_sample.csv'
df = pd.read_csv(file_path)

llm = ChatOllama(model="tinyllama", temperature=0) # Use a powerful model for better reasoning

print("Creating agent...")
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
    allow_dangerous_code=True
)
print("Agent created successfully.")

response = agent.invoke("What are the top 2 articles related to password reset issues? keep the answer in short")
print(response)