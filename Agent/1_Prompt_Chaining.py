import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------- 1. 加载 API 密钥 ----------
project_root = Path(__file__).resolve().parent.parent
key_path = project_root / "key.json"
with key_path.open('r', encoding='utf-8') as f:
    config = json.load(f)

# ---------- 2. 初始化模型 (兼容 OpenAI 接口的 DeepSeek) ----------
llm = ChatOpenAI(
    model="deepseek-chat",                         # 使用 deepseek-chat 模型
    openai_api_key=config['deepseek']['key'],
    openai_api_base="https://api.deepseek.com",    # 注意参数名差异
    temperature=0.7,
)

# ---------- 3. 定义提示链环节 ----------
# --- 提示 1：信息提取 ---
prompt_extract = ChatPromptTemplate.from_template(
  "请从以下文本中提取技术规格：\n\n{text_input}"
)

# --- 提示 2：转为 JSON ---
prompt_transform = ChatPromptTemplate.from_template(
  "请将以下技术规格转为 JSON 格式，包含 'cpu'、'memory' 和 'storage' 三个键：\n\n{specifications}"
)

# ---------- 4. 构建链 ----------
# 使用 LCEL (LangChain Expression Language) 构建提示链
chain = (
    {
        "specifications": prompt_extract | llm | StrOutputParser(), 
        "text_input": RunnablePassthrough()
    }
    | prompt_transform
    | llm
    | StrOutputParser()
)

# ---------- 5. 测试运行 ----------
try:
    input_text = (
        "新款笔记本配备 3.5GHz 八核处理器、16GB 内存和 1TB NVMe SSD。"
    )
    result = chain.invoke({"text_input": input_text})
    print("提示链输出：")
    print(result)
except Exception as e:
    print(f"运行出错：{e}")
