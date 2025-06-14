import openai
import json
from Qwen3_vllm import chat_with_thinking

# 配置客户端
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy-key"  # vllm通常不需要真实的API key
)

def test_math_problems():
    """测试math_500.jsonl中的数学问题"""
    prompt_template = '{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.'
    
    # 读取数学问题文件
    with open('math_500.jsonl', 'r', encoding='utf-8') as f:
        problems = [json.loads(line) for line in f]
    
    print(f"加载了 {len(problems)} 个数学问题")
    
    # 测试前几个问题
    for i, item in enumerate(problems[:5]):  # 只测试前5个问题
        problem = item['problem']
        expected_answer = item['answer']
        subject = item['subject']
        level = item['level']
        
        print(f"\n=== 问题 {i+1} ===")
        print(f"难度: {level}")
        print(f"问题: {problem}")  # 只显示前200个字符
        print(f"答案: {expected_answer}")
        
        # 使用提示词模板格式化问题
        full_prompt = prompt_template.format(query=problem)
        
        # 调用思考模式进行推理
        print("\n正在使用思考模式求解...")
        print("模型回答：",chat_with_thinking(full_prompt))
        
        print("\n" + "="*50)

def main():
    test_math_problems()
        
if __name__ == "__main__":
    main()