import os
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"


import csv
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import time

start_time = time.time()   #

model_dir = "/home/mbr/vllm-build/Qwen/Qwen3-0.6B"
output_csv = "xiaohongshu_comments_vllm.csv"

# 初始化 vLLM 引擎（离线模式）
llm = LLM(
    model=model_dir,
    trust_remote_code=True,
    max_model_len=2048,
    enforce_eager=True,               # 兼容性
    gpu_memory_utilization=0.8,
    max_num_seqs=32,
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
sampling_params = SamplingParams(
    temperature=0.9,
    top_p=0.9,
    top_k=50,
    max_tokens=200,
    stop=["<|im_end|>", "<|endoftext|>"]
)

def generate_comment(sentiment: str, topic: str = "") -> str:
    topic_hint = f"关于{topic}" if topic else "任意产品/服务"
    messages = [
        {"role": "system", "content": "你是一个小红书资深用户，擅长用小红书文体写评论。你的回复只包含评论本身，不包含任何额外说明。"},
        {"role": "user", "content": f"请写一条{topic_hint}的{sentiment}评价，模仿小红书文案风格，语气自然，带emoji，约50-100字。"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False   # 关闭思考模式
    )
    outputs = llm.generate([prompt], sampling_params)
    comment = outputs[0].outputs[0].text.strip()
    # 简单清理可能遗留的 <|im_end|> 等
    comment = comment.split("<|im_end|>")[0].strip()
    return comment

topics = ["美食", "旅行", "美妆", "穿搭", "家居好物", "数码产品", "书籍", "健身", "咖啡", "护肤"]
all_comments = []

# 生成正面评论
for i in range(100):
    topic = topics[i % len(topics)]
    try:
        comment = generate_comment("正面", topic)
        all_comments.append((comment, "正面"))
        print(f"正面 [{i+1}/100] ({topic}): {comment[:50]}...")
    except Exception as e:
        print(f"生成第{i+1}条正面评论失败: {e}")
        all_comments.append(("", "正面"))

# 生成负面评论
for i in range(100):
    topic = topics[i % len(topics)]
    try:
        comment = generate_comment("负面", topic)
        all_comments.append((comment, "负面"))
        print(f"负面 [{i+1}/100] ({topic}): {comment[:50]}...")
    except Exception as e:
        print(f"生成第{i+1}条负面评论失败: {e}")
        all_comments.append(("", "负面"))

with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["评论内容", "情感标签"])
    writer.writerows(all_comments)
    
    

end_time = time.time()

print(f"共生成 {len(all_comments)} 条评论，已保存到 {output_csv}")

print(f"总耗时: {end_time - start_time:.2f} 秒")