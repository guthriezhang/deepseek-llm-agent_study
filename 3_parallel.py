import json
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

with open('key.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

llm = ChatOpenAI(
    model="deepseek-chat",                         # 使用 deepseek-chat 模型
    openai_api_key=config['deepseek']['key'],
    openai_api_base="https://api.deepseek.com",    # 注意参数名差异
    temperature=0.7,
)

summarize_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "请简明扼要地总结以下主题："),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)

import os
import asyncio
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# --- 配置 ---
# 确保环境变量已设置 API key（如 OPENAI_API_KEY）
try:
   llm: Optional[ChatOpenAI] = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
  
except Exception as e:
   print(f"初始化语言模型出错：{e}")
   llm = None

# --- 定义独立链 ---
# 三个链分别执行不同任务，可并行运行

summarize_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "请简明扼要地总结以下主题："),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)

questions_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "请针对以下主题生成三个有趣的问题："),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)

terms_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "请从以下主题中提取 5-10 个关键词，用逗号分隔："),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)

map_chain = RunnableParallel(
   {
       "summary": summarize_chain,
       "questions": questions_chain,
       "key_terms": terms_chain,
       "topic": RunnablePassthrough(),  # 传递原始 topic
   }
)

synthesis_prompt = ChatPromptTemplate.from_messages([
   ("system", """根据以下信息：
    摘要：{summary}
    相关问题：{questions}
    关键词：{key_terms}
    请综合生成完整答案。"""),
   ("user", "原始主题：{topic}")
])

full_parallel_chain = (
     map_chain
   | synthesis_prompt 
   | llm 
   | StrOutputParser()
)

async def run_parallel_example(topic:str) -> None :
   try:
      response = await full_parallel_chain.ainvoke(topic)
      print(response)
   except Exception as e:
      print(f"\nError:{e}")
    
if __name__ == "__main__" :
   asyncio.run(run_parallel_example("人类的历史"))
