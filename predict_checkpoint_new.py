from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import pandas as pd
import torch
import transformers
import json
import os

# 加载原始基础模型
# base_model_path = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/phi-4"
# tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
# base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.bfloat16)

# # 加载 LoRA 微调权重
# peft_checkpoint_path = "/home/qust521/Projects/task10/phi-4/output/phi-4_4096_epoch/checkpoint-850"
# model = PeftModel.from_pretrained(base_model, peft_checkpoint_path)
# 预测函数
all_fine_grained_roles_list= ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", 
                              "Virtuous","Instigator", "Conspirator", "Tyrant", "Foreign Adversary", 
                              "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent","Terrorist", "Deceiver", 
                              "Bigot","Forgotten", "Exploited", "Victim", "Scapegoat"]
def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []
    # 读取旧的JSONL文件
    with open(origin_path, "r", encoding='UTF-8') as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            context = data["text"]
            entity_mention = data["entity"] #实体
            answer = data["fine_grained_role"]
            message = {
                "instruction":f"You are an expert in the field of multi label classification. Given an article and an entity within that article. What you need to do is analyze this article and the entity, and provide the most appropriate fine-grained role of the entity.Respond only with{all_fine_grained_roles_list},without providing any additional context or explanation.",
                "input": f"article:{context},entity:{entity_mention},",
                "output": answer,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
pipeline = transformers.pipeline(
    "text-generation",
    model="/home/qust521/Projects/task10/phi-4/output/phi-4_4096_epoch/checkpoint-258",
    model_kwargs={"torch_dtype": "bfloat16"},
    device_map="auto",
)

test_dataset_path = "test.jsonl"
test_jsonl_new_path = "new_test_new.jsonl"
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

test_df = pd.read_json(test_jsonl_new_path, lines=True)
for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"},
    ]
    outputs = pipeline(messages, max_new_tokens=512)
    response=outputs[0]["generated_text"][-1]
    result = response['content']
    print("result:",result)
    with open("result_new_epoch3.txt", "a",encoding='utf-8') as f:
        f.writelines(result)
        f.write("\n")