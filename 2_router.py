import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# ---------- 1. 加载 API 密钥 ----------
with open('key.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# ---------- 2. 初始化模型 (兼容 OpenAI 接口的 DeepSeek) ----------
llm = ChatOpenAI(
    model="deepseek-chat",                         # 使用 deepseek-chat 模型
    openai_api_key=config['deepseek']['key'],
    openai_api_base="https://api.deepseek.com",    # 注意参数名差异
    temperature=0.7,
)

# ---------- 3. 定义模拟子智能体处理器 ----------
def booking_handler(request: str) -> str:
   """模拟预订智能体处理请求。"""
   print("\n--- 委托给预订处理器 ---")
   return f"预订处理器已处理请求：'{request}'。结果：模拟预订动作。"

def info_handler(request: str) -> str:
   """模拟信息智能体处理请求。"""
   print("\n--- 委托给信息处理器 ---")
   return f"信息处理器已处理请求：'{request}'。结果：模拟信息检索。"

def unclear_handler(request: str) -> str:
   """处理无法委托的请求。"""
   print("\n--- 处理不明确请求 ---")
   return f"协调者无法委托请求：'{request}'。请补充说明。"

# ---------- 4. 定义协调者路由链 ----------
coordinator_router_prompt = ChatPromptTemplate.from_messages([
   ("system", """分析用户请求，判断应由哪个专属处理器处理。
    - 若请求涉及预订机票或酒店，输出 'booker'。
    - 其他一般信息问题，输出 'info'。
    - 若请求不明确或不属于上述类别，输出 'unclear'。
    只输出一个词：'booker'、'info' 或 'unclear'。"""),
   ("user", "{request}")
])

# ---------- 5. 定义委托逻辑 ----------
branches = {
   "booker": RunnablePassthrough.assign(output=lambda x: booking_handler(x['request']['request'])),
   "info": RunnablePassthrough.assign(output=lambda x: info_handler(x['request']['request'])),
   "unclear": RunnablePassthrough.assign(output=lambda x: unclear_handler(x['request']['request'])),
}

delegation_branch = RunnableBranch(
   (lambda x: x['decision'].strip() == 'booker', branches["booker"]),
   (lambda x: x['decision'].strip() == 'info', branches["info"]),
   branches["unclear"] # 默认分支
)

coordinator_agent = (
   {
        "decision": coordinator_router_prompt | llm | StrOutputParser(),
        "request": RunnablePassthrough()
   } 
   | delegation_branch 
   | (lambda x: x['output']))

def main():
   if not llm:
       print("\n因 LLM 初始化失败，跳过执行。")
       return

   print("--- 预订请求示例 ---")
   request_a = "帮我预订飞往伦敦的机票。"
   result_a = coordinator_agent.invoke({"request": request_a})
   print(f"最终结果 A: {result_a}")

   print("\n--- 信息请求示例 ---")
   request_b = "意大利的首都是哪里？"
   result_b = coordinator_agent.invoke({"request": request_b})
   print(f"最终结果 B: {result_b}")

   print("\n--- 不明确请求示例 ---")
   request_c = "讲讲量子物理。"
   result_c = coordinator_agent.invoke({"request": request_c})
   print(f"最终结果 C: {result_c}")

if __name__ == "__main__":
   main()
