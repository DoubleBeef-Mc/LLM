import csv
import json
import re
import requests
import time

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "/home/mbr/vllm-build/Qwen/Qwen3-0.6B"
input_csv = "xiaohongshu_comments.csv"
output_csv = "classified_comments_api.csv"

def extract_json_from_text(text: str):
    if '</think>' in text:
        _, after = text.rsplit('</think>', 1)
    else:
        after = text
    # 匹配第一个 { 到第一个 } 之间的内容
    match = re.search(r'({.*?})', after, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 打印调试信息，查看未解析的字符串片段
            print(f"JSON解析失败，原始片段: {json_str[:200]}")
    return None

def call_classify(comment):
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
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
        "stop": ["<|im_end|>", "<|endoftext|>"]
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=(10, 120))
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"响应长度: {len(content)} 字符，前200字符: {content[:200]}...")
            parsed = extract_json_from_text(content)
            if parsed:
                return parsed
            else:
                print(f"无法提取JSON，完整响应: {content}")
        else:
            print(f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"请求异常: {e}")
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
failures = 0
for idx, (comment, true_label) in enumerate(data, 1):
    print(f"处理 {idx}/{len(data)}: {comment[:30]}...")
    parsed = call_classify(comment)
    if parsed is None:
        failures += 1
        pred_num = -1
        reason = "解析失败"
    else:
        pred_num = parsed.get("result", -1)
        reason = parsed.get("reason", "")
    results.append((comment, true_label, pred_num, reason))
    time.sleep(0.3)

with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["评论内容", "真实标签", "预测标签", "判断理由"])
    writer.writerows(results)

print(f"完成，结果保存至 {output_csv}，失败数: {failures}")