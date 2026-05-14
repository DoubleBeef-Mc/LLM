import csv
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ================== 配置区域 ==================
model_dir = "/home/mbr/vllm-build/Qwen/Qwen3-0.6B"   # WSL 内模型绝对路径
input_csv = "xiaohongshu_comments.csv"              # 输入的评论文件（放在当前目录）
output_csv = "classified_comments_offline.csv"      # 输出结果文件

vllm_config = {
    "model": model_dir,
    "trust_remote_code": True,
    "max_model_len": 2048,
    "enforce_eager": True,           # 保持与服务端一致
    "gpu_memory_utilization": 0.8,
    "max_num_seqs": 32,
}

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=80,
    stop=["<|im_end|>", "<|endoftext|>"]
)

print("正在加载 vLLM 引擎...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
llm = LLM(**vllm_config)
print("vLLM 引擎加载完成！")

def extract_json_from_text(text: str):
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    return match.group(0) if match else None

def classify_comment_to_json(comment: str, max_retries: int = 2):
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
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    for _ in range(max_retries):
        outputs = llm.generate([prompt], sampling_params)
        output_text = outputs[0].outputs[0].text.strip()
        json_str = extract_json_from_text(output_text)
        if json_str:
            try:
                parsed = json.loads(json_str)
                if "result" in parsed and "reason" in parsed:
                    return parsed
            except:
                continue
    return None

# 读取 CSV
data = []
with open(input_csv, "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) >= 2:
            data.append((row[0].strip(), row[1].strip()))

results = []
true_labels = []
pred_labels_num = []   # 存储 0/1 数字
parse_failures = 0

for idx, (comment, true_label) in enumerate(data, 1):
    parsed = classify_comment_to_json(comment)
    if parsed is None:
        parse_failures += 1
        pred_num = -1      # 用 -1 表示解析失败
        reason = "解析失败"
    else:
        pred_num = parsed["result"]   # 1 或 0
        reason = parsed["reason"]

    true_labels.append(true_label)
    pred_labels_num.append(pred_num)
    results.append((comment, true_label, pred_num, reason))

    if idx % 20 == 0:
        print(f"进度: {idx}/{len(data)} （解析失败: {parse_failures}）")

# 保存结果（预测标签直接写数字）
with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["评论内容", "真实标签", "预测标签", "判断理由"])
    writer.writerows(results)

print(f"结果已保存到 {output_csv}")