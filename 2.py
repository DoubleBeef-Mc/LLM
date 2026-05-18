import os
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ["VLLM_DISABLE_FLASHINFER"] = "1"

import csv
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

start_time = time.time()   # 开始计时

model_dir = "/home/mbr/vllm-build/Qwen/Qwen3-0.6B"
input_csv = "xiaohongshu_comments.csv"
output_csv = "classified_comments_offline_vllm.csv"

# 初始化 vLLM（离线）
llm = LLM(
    model=model_dir,
    trust_remote_code=True,
    max_model_len=2048,
    enforce_eager=True,
    gpu_memory_utilization=0.8,
    max_num_seqs=32,
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1024,
    stop=["<|im_end|>", "<|endoftext|>"]
)

def build_prompt(comment: str) -> str:
    system_prompt = (
        "你是一个评论审核专家。请判断下面的小红书评论是正面还是负面。"
        "请严格按 JSON 格式输出，不要添加任何其他文字。"
        '格式示例：{"result": 1, "reason": "这里写判断理由"}'
        "其中 result 为 1 表示正面，0 表示负面。"
    )
    user_prompt = f"评论：{comment}\n请输出JSON："
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

def extract_json_from_text(text: str):
    if '</think>' in text:
        _, after = text.rsplit('</think>', 1)
    else:
        after = text
    match = re.search(r'({.*?})', after, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"JSON解析失败: {json_str[:200]}")
    return None

# 读取 CSV
data = []
with open(input_csv, "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) >= 2:
            data.append((row[0].strip(), row[1].strip()))

# 批量构造 prompts
prompts = [build_prompt(comment) for comment, _ in data]

print(f"正在对 {len(prompts)} 条评论进行批量推理...")
outputs = llm.generate(prompts, sampling_params)

results = []
failures = 0
for idx, (output, (comment, true_label)) in enumerate(zip(outputs, data), 1):
    generated = output.outputs[0].text.strip()
    parsed = extract_json_from_text(generated)
    if parsed is None:
        failures += 1
        pred_num = -1
        reason = "解析失败"
    else:
        pred_num = parsed.get("result", -1)
        reason = parsed.get("reason", "")
    results.append((comment, true_label, pred_num, reason))
    if idx % 20 == 0:
        print(f"处理进度: {idx}/{len(data)} (失败: {failures})")

with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["评论内容", "真实标签", "预测标签", "判断理由"])
    writer.writerows(results)

print(f"完成，结果保存至 {output_csv}，失败数: {failures}")
end_time = time.time()
print(f"总耗时: {end_time - start_time:.2f} 秒")