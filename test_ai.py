import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载那个 .env 文件里的密码
load_dotenv()

# 2. 从环境变量里获取 Key，这样代码里就没有明文密码了，安全！
api_key = os.getenv("OPENAI_API_KEY")

# 检查一下有没有拿到（为了安全，只打印前几位）
if api_key:
    print(f"Key loaded successfully: {api_key[:5]}...")
else:
    print("Error: API Key not found!")
    exit()

# 3. 初始化 OpenAI 客户端
client = OpenAI(api_key=api_key)

print("Sending message to OpenAI... (Please wait)")

try:
    # 4. 发送一个最简单的测试请求
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 先用便宜的模型测试
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Are you working?"}
        ]
    )

    # 5. 打印回复
    print("--- AI Response ---")
    print(completion.choices[0].message.content)
    print("-------------------")
    print("Test Successful! 🎉")

except Exception as e:
    print(f"Test Failed: {e}")