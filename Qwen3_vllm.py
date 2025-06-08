#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 模型测试脚本
测试本地部署在8001端口的vllm模型
包括 completions 和 chat 两种模式（思考和不思考）
"""

import openai
import json
import time
from typing import List, Dict, Any

# 配置客户端
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy-key"  # vllm通常不需要真实的API key
)

def completions(prompt: str):
    """
    测试 completions 接口
    """
    print("\n=== 测试 Completions 接口 ===")
    
    try:
        response = client.completions.create(
            model="8001vllm",
            prompt=prompt,
            temperature=0.7,
            top_p=0.8,
            extra_body={
                "top_k": 20,   # 控制质量
                "min_p": 0,    # 无最小阈值
            }
        )
        
        print(f"输入: {prompt}")
        print(f"输出: {response.choices[0].text.strip()}")
        print(f"使用tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
        
    except Exception as e:
        print(f"Completions 测试失败: {e}")

def chat_without_thinking(prompt: str):
    """
    测试 Chat 接口 - 不思考模式（快速响应）
    """
    print("\n=== 测试 Chat 接口 - 不思考模式 ===")
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        start_time = time.time()
        
        # 适合普通问答的配置
        response = client.chat.completions.create(
            model="8001vllm",
            messages=messages,
            temperature=0.7,    # 保持创造性
            top_p=0.8,         # 平衡多样性
            extra_body={
                "top_k": 20,   # 控制质量
                "min_p": 0,    # 无最小阈值
                "chat_template_kwargs": {
                    "enable_thinking": False  # 快速响应
                }
            }
        )
        
        end_time = time.time()
        
        print(f"输入: {messages[0]['content']}")
        print(f"输出: {response.choices[0].message.content}")
        print(f"响应时间: {end_time - start_time:.2f}秒")
        print(f"使用tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
        
    except Exception as e:
        print(f"Chat 不思考模式测试失败: {e}")

def chat_with_thinking(prompt: str):
    """
    测试 Chat 接口 - 思考模式（复杂推理）
    """
    print("\n=== 测试 Chat 接口 - 思考模式 ===")
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        start_time = time.time()
        
        # 适合复杂推理的配置
        response = client.chat.completions.create(
            model="8001vllm",
            messages=messages,
            temperature=0.6,    # 更稳定的推理
            top_p=0.95,        # 更多推理路径
            extra_body={
                "top_k": 20,   # 控制质量
                "min_p": 0,    # 无最小阈值
                "chat_template_kwargs": {
                    "enable_thinking": True   # 启用深度思考
                }
            }
        )
        
        end_time = time.time()
        
        print(f"输入: {messages[0]['content']}")
        print(f"输出: {response.choices[0].message.content}")
        print(f"响应时间: {end_time - start_time:.2f}秒")
        print(f"使用tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
        
    except Exception as e:
        print(f"Chat 思考模式测试失败: {e}")



def interactive_test():
    """
    交互式测试界面
    """
    print("欢迎使用 vLLM 模型测试工具 (localhost:8001)")
    print("=" * 50)

    
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
        
        try:
            # 根据选择执行相应的测试
            if choice == "1":
                completions(prompt)
            elif choice == "2":
                chat_without_thinking(prompt)
            elif choice == "3":
                chat_with_thinking(prompt)

        except Exception as e:
            print(f"发生错误: {e}")
        
if __name__ == "__main__":
    interactive_test()