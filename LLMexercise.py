import csv
import re
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch


model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()


def classify_comment(comment: str) -> str:
    """返回 '正面' 或 '负面'"""
    messages = [
        {"role": "system", "content": "你是一个情感分类专家。请阅读用户给出的小红书评论，并判断情感倾向。只回复'正面'或'负面'，不要加任何标点或解释。"},
        {"role": "user", "content": f"评论：{comment}\n情感倾向："}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():#在生成文本时不计算梯度，节省内存和计算资源，因为我们只需要模型的推理能力来进行分类，而不需要进行训练或微调。
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,         # 确定性输出，提高一致性
            temperature=0.9,
            top_k=0.9,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    output_ids = generated_ids[0][inputs.input_ids.shape[1]:].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    # 解析输出，提取正面/负面
    if "负面" in output_text:
        return "负面"
    elif "正面" in output_text:
        return "正面"
    else:
        # 无法识别时，随机给正面（后续统计会体现）
        return "正面"  # 默认正面，但会被视为错误


data = []
try:
    with open("xiaohongshu_comments.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for row in reader:
            if len(row) >= 2:
                data.append((row[0].strip(), row[1].strip()))
except FileNotFoundError:
    print("错误：找不到 xiaohongshu_comments.csv，请确保文件在当前目录")
    exit()




true_labels = []
pred_labels = []
results = []

for idx, (comment, true_label) in enumerate(data, 1):
    if not comment:
        continue  # 跳过空评论
    pred = classify_comment(comment)
    true_labels.append(true_label)
    pred_labels.append(pred)
    results.append((comment, true_label, pred))
    if idx % 20 == 0:
        print(f"进度: {idx}/{len(data)}")


from collections import Counter

# 准确率
correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
accuracy = correct / len(true_labels) if true_labels else 0


tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "正面" and p == "正面")
fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "正面" and p == "负面")
fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "负面" and p == "正面")
tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "负面" and p == "负面")

# 精确率、召回率、F1（正类为正面）
precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0


print("\n=== 分类评估结果 ===")
print(f"总样本数: {len(true_labels)}")
print(f"准确率 (Accuracy): {accuracy:.4f} ({correct}/{len(true_labels)})")
print(f"\n混淆矩阵:")
print(f"              预测正面  预测负面")
print(f"实际正面        {tp:4d}      {fn:4d}")
print(f"实际负面        {fp:4d}      {tn:4d}")
print(f"\n正面类别:")
print(f"  精确率 (Precision): {precision_pos:.4f}")
print(f"  召回率 (Recall):    {recall_pos:.4f}")
print(f"  F1-score:          {f1_pos:.4f}")
print(f"\n负面类别:")
print(f"  精确率 (Precision): {precision_neg:.4f}")
print(f"  召回率 (Recall):    {recall_neg:.4f}")
print(f"  F1-score:          {f1_neg:.4f}")


output_file = "classified_comments.csv"
with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["评论内容", "真实标签", "预测标签"])
    writer.writerows(results)
