import csv
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 本地模型路径
model_dir = r"C:\Users\mbr\Desktop\LLM\LLM\Qwen3-0.6B"


tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model.eval()


def extract_json_from_text(text: str):
    # 从文本中提取第一个看起来像 JSON 对象的字符串
    # 尝试匹配 {...} 的最外层结构，允许跨行，但不允许嵌套大括号
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)# 这个正则表达式会匹配第一个大括号内的内容，且不允许嵌套大括号。如果输出中有多余的文字或者格式不正确，这个方法可能无法提取到正确的 JSON，但在大多数情况下应该能正常工作。
    if match:
        return match.group(0)# 返回匹配到的 JSON 字符串
    return None

def classify_comment_to_json(comment: str, max_retries: int = 3):
    #让模型输出 JSON，包含 result 和 reason。返回解析后的 dict，如果多次尝试都失败则返回 None。
    
    system_prompt = (
    "你是一个评论审核专家。请判断下面的小红书评论是正面还是负面。"
    "严格按照 JSON 格式输出，不要添加任何其他文字。\n"
    "格式：{\"result\": 1, \"reason\": \"你的判断理由\"}\n"
    "规则：\n"
    "- result 为 1 表示正面，0 表示负面\n"
    "- reason 必须结合评论内容，用简短的句子解释为什么这样判断，不要只写'正面'或'负面'，也不要写'这里写判断理由'\n"
    "示例1：评论'这家店太好吃了！' → {\"result\": 1, \"reason\": \"直接表达了对食物的赞美\"}\n"
    "示例2：评论'等了两小时才上菜' → {\"result\": 0, \"reason\": \"对服务速度表达不满\"}"
)
    user_prompt = f"评论：{comment}\n请输出JSON："

    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        output_ids = generated_ids[0][inputs.input_ids.shape[1]:].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # 尝试提取 JSON
        json_str = extract_json_from_text(output_text)
        if json_str is None:
            # 没找到大括号，重试
            continue

        try:
            parsed = json.loads(json_str)# 解析 JSON，可能会失败
            # 确保字段存在
            if "result" in parsed and "reason" in parsed:# 如果解析成功且字段完整，返回结果
                return parsed
            else:
                # 字段不全，重试
                continue
        except (json.JSONDecodeError, ValueError):
            # 解析失败，重试
            continue

    # 所有重试都失败
    return None


# 读取数据
data = []
try:
    with open("xiaohongshu_comments.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                data.append((row[0].strip(), row[1].strip()))
except FileNotFoundError:
    print("错误：找不到 xiaohongshu_comments.csv")
    exit()


true_labels = []
pred_labels = []
results = []
parse_failures = 0

for idx, (comment, true_label) in enumerate(data, 1):
    if not comment:
        continue

    parsed = classify_comment_to_json(comment)# 解析评论，得到 dict 或 None
    if parsed is None:
        # 放弃该条
        parse_failures += 1
        pred = "未知"
        reason = "解析失败"
    else:
        pred_num = parsed["result"]
        pred = "正面" if pred_num == 1 else "负面"
        reason = parsed["reason"]

    true_labels.append(true_label)
    pred_labels.append(pred)
    results.append((comment, true_label, pred, reason))

    if idx % 20 == 0:
        print(f"进度: {idx}/{len(data)} （解析失败累计: {parse_failures}）")

# 计算指标（忽略“未知”标签，或者视其为错误）
# 这里我们把“未知”当作预测错误来处理
correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
accuracy = correct / len(true_labels) if true_labels else 0

tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "正面" and p == "正面")
fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "正面" and p == "负面")
fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "负面" and p == "正面")
tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "负面" and p == "负面")

# 如果预测为“未知”，不计入任何混淆矩阵元素，但会降低准确率
unknown_count = sum(1 for p in pred_labels if p == "未知")

def calc_metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

p_pos, r_pos, f1_pos = calc_metrics(tp, fp, fn)
p_neg, r_neg, f1_neg = calc_metrics(tn, fn, fp)

print("\n=== 分类评估结果 ===")
print(f"总样本数: {len(true_labels)}")
print(f"成功解析数: {len(true_labels) - unknown_count}")
print(f"解析失败数: {unknown_count}")
print(f"准确率 (Accuracy): {accuracy:.4f} ({correct}/{len(true_labels)})")
print(f"\n混淆矩阵（不含未知）:")
print(f"              预测正面  预测负面")
print(f"实际正面        {tp:4d}      {fn:4d}")
print(f"实际负面        {fp:4d}      {tn:4d}")
print(f"\n正面类别: P={p_pos:.4f}, R={r_pos:.4f}, F1={f1_pos:.4f}")
print(f"负面类别: P={p_neg:.4f}, R={r_neg:.4f}, F1={f1_neg:.4f}")

# 保存结果（新增 reason 列）
with open("classified_comments_json.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["评论内容", "真实标签", "预测标签", "判断理由"])
    writer.writerows(results)
print("结果已保存到 classified_comments_json.csv")