import openai
from typing import List, Dict, Any

# 配置客户端
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy-key"  # vllm通常不需要真实的API key
)

def completions(prompt: str):
    
    response = client.completions.create(
        model="8001vllm",
        prompt=prompt,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,   # 控制质量
            "min_p": 0,    # 无最小阈值
        }
    )

    return response.choices[0].text.strip()
    
def chat_without_thinking(prompt: str):
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 适合普通问答的配置
    response = client.chat.completions.create(
        model="8001vllm",
        messages=messages,
        temperature=0.7,    # 保持创造性
        top_p=0.8,         # 平衡多样性
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,   # 控制质量
            "min_p": 0,    # 无最小阈值
            "chat_template_kwargs": {
                "enable_thinking": False  # 快速响应
            }
        }
    )

    return response.choices[0].message.content

def chat_with_thinking(prompt: str):
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 适合复杂推理的配置
    response = client.chat.completions.create(
        model="8001vllm",
        messages=messages,
        temperature=0.6,    # 更稳定的推理
        top_p=0.95,        # 更多推理路径
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,   # 控制质量
            "min_p": 0,    # 无最小阈值
            "chat_template_kwargs": {
                "enable_thinking": True   # 启用深度思考
            }
        }
    )

    return response.choices[0].message.content

def main():
    while True:
        print("\n请选择测试模式:")
        print("1. Completions 接口测试")
        print("2. Chat 接口测试 - 不思考模式（快速响应）")
        print("3. Chat 接口测试 - 思考模式（复杂推理）")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "4":
            break
        
        if choice not in ["1", "2", "3"]:
            print("无效选择，请重新输入")
            continue
        
        # 获取用户输入的prompt
        prompt = input("\n请输入prompt: ").strip()
        
        if not prompt:
            print("输入不能为空，请重新输入")
            continue
        
        # 根据选择执行相应的测试
        if choice == "1":
            print("输出：", completions(prompt))
        elif choice == "2":
            print("输出：", chat_without_thinking(prompt))
        elif choice == "3":
            print("输出：", chat_with_thinking(prompt))
        
if __name__ == "__main__":
    main()