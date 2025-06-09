import json
import time
import os
import asyncio
from tqdm import tqdm
import argparse
from openai import AsyncOpenAI

# 设置OpenAI客户端连接到本地vLLM服务
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"  # 注意端口是8001

# 读取math7500.jsonl文件
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 保存结果到新的jsonl文件
def save_results(results, output_file, mode='w'):
    with open(output_file, mode, encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 异步调用模型处理问题
async def process_problem(client, problem, retry=3):
    prompt = f"{problem}"
    
    for attempt in range(retry):
        try:
            response = await client.chat.completions.create(
                model="8001vllm",  # 使用脚本中设置的模型名称
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=32768,
                temperature=0.7,    # 保持创造性
                top_p=0.8,         # 平衡多样性
                extra_body={
                    "top_k": 20,   # 控制质量
                    "min_p": 0,    # 无最小阈值
                    "skip_special_tokens": False,  # 不跳过特殊token
                    "chat_template_kwargs": {
                        "enable_thinking": False  # 快速响应
                    }
                },
            )
            return response.choices[0].message.content, None
        except Exception as e:
            if attempt < retry - 1:
                print(f"\n尝试 {attempt+1}/{retry} 失败: {str(e)[:100]}... 等待5秒后重试")
                await asyncio.sleep(5)  # 异步等待
            else:
                print(f"\n所有 {retry} 次尝试均失败! 错误: {str(e)[:100]}...")
                return None, str(e)
    
    return None, "模型调用失败"

# 异步处理单个问题
async def process_item(client, idx, item, repeat, max_retries=3):
    try:
        # 提取问题
        problem = item.get('problem', '')
        problem_id = item.get('id', idx)
        
        # 调用模型处理问题
        answer, error = await process_problem(client, problem, max_retries)
        
        # 返回结果
        return idx, {
            "id": problem_id,
            "problem": problem,
            "answer": answer if answer else "模型调用失败",
            "error": error,
            "round": repeat
        }
    except Exception as e:
        # 处理异常
        return idx, {
            "id": problem_id if 'problem_id' in locals() else idx,
            "problem": problem if 'problem' in locals() else "未知问题",
            "answer": "处理过程出现错误",
            "error": str(e),
            "round": repeat
        }

# 使用信号量限制并发数
async def run_with_concurrency_limit(client, problems, repeat, batch_size, max_retries):
    # 创建信号量限制并发
    semaphore = asyncio.Semaphore(batch_size)
    
    async def limited_process(idx, item):
        async with semaphore:
            return await process_item(client, idx, item, repeat, max_retries)
    
    # 创建所有任务
    tasks = [limited_process(idx, item) for idx, item in enumerate(problems)]
    
    # 创建进度条
    progress_bar = tqdm(total=len(tasks), desc=f"第{repeat}轮处理")
    
    # 结果列表
    results = []
    batch_results = []
    success_count = 0
    failure_count = 0
    
    # 使用as_completed处理完成的任务
    for completed_task in asyncio.as_completed(tasks):
        try:
            idx, result = await completed_task
            
            # 检查是否成功
            if result.get("error") is None:
                success_count += 1
            else:
                failure_count += 1
            
            # 保存结果
            results.append(result)
            batch_results.append(result)
            
            # 每10条数据保存一次临时结果
            if len(batch_results) >= 10:
                save_results(batch_results, f"math7500_results/math7500_results_round{repeat}_temp.jsonl", mode='a')
                batch_results = []
            
            # 更新进度条
            progress_bar.update(1)
            
        except Exception as e:
            print(f"\n处理任务时发生错误: {e}")
            progress_bar.update(1)
    
    # 关闭进度条
    progress_bar.close()
    
    # 保存剩余的临时结果
    if batch_results:
        save_results(batch_results, f"math7500_results/math7500_results_round{repeat}_temp.jsonl", mode='a')
    
    # 按ID排序结果
    sorted_results = sorted(results, key=lambda x: x["id"] if isinstance(x["id"], int) else int(x["id"]))
    
    return sorted_results, success_count, failure_count

async def main_async(input_file, output_dir, n_repeats=1, batch_size=256, max_retries=3):
    # 创建异步OpenAI客户端
    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    problems = load_jsonl(input_file)
    print(f"加载了 {len(problems)} 个问题")
    
    # 对每次重复进行处理
    for repeat in range(1, n_repeats+1):
        print(f"开始第 {repeat}/{n_repeats} 轮处理")
        
        try:
            # 使用并发控制运行所有任务
            sorted_results, success_count, failure_count = await run_with_concurrency_limit(
                client, problems, repeat, batch_size, max_retries
            )
            
            # 保存最终结果
            output_file = f"{output_dir}/math7500_results_round{repeat}.jsonl"
            save_results(sorted_results, output_file)
            
            # 打印统计信息
            print(f"\n第 {repeat} 轮处理完成:")
            print(f"  总计处理: {len(sorted_results)}/{len(problems)} 个问题")
            print(f"  成功: {success_count} 个问题 ({success_count/len(sorted_results)*100:.1f}%)")
            print(f"  失败: {failure_count} 个问题 ({failure_count/len(sorted_results)*100:.1f}%)")
            print(f"  结果已保存到: {output_file}")
            
        except KeyboardInterrupt:
            print("\n接收到中断信号，停止处理...")
            break
        except Exception as e:
            print(f"\n处理第 {repeat} 轮时发生错误: {e}")
            continue

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='异步协程处理数学问题')
    parser.add_argument('--input', type=str, default="math7500.jsonl", help='输入文件路径')
    parser.add_argument('--output_dir', type=str, default="math7500_results", help='输出目录')
    parser.add_argument('--repeats', type=int, default=1000, help='重复次数')
    parser.add_argument('--batch_size', type=int, default=256, help='并发数量')
    parser.add_argument('--max_retries', type=int, default=3, help='最大重试次数')
    
    args = parser.parse_args()
    
    # 运行异步主函数
    asyncio.run(main_async(
        input_file=args.input,
        output_dir=args.output_dir,
        n_repeats=args.repeats,
        batch_size=args.batch_size,
        max_retries=args.max_retries
    ))

if __name__ == "__main__":
    main()