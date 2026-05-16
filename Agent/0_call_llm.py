import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage

project_root = Path(__file__).resolve().parent.parent
key_path = project_root / "key.json"
with key_path.open('r', encoding='utf-8') as f:
    config = json.load(f)

llm_model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=config['deepseek']['key'],
    openai_api_base="https://api.deepseek.com",
    temperature=0.7,
)
response = llm_model.invoke([HumanMessage(content="解释机器学习")])
print(response.content)

agent = create_agent(
    model=llm_model,
    system_prompt="你是一个有帮助的助手。请简洁准确。"
)
result = agent.invoke(
    {"messages": [HumanMessage(content="解释统计学习")]}
)
print(result["messages"][-1].content)
