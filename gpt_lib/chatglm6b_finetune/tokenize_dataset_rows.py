# -- coding: utf-8 --
# @Time : 2023/4/24
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
ref https://github.com/mymusise/ChatGLM-Tuning/blob/master/tokenize_dataset_rows.py
"""
import argparse
import json
from tqdm import tqdm
import datasets
import transformers
from cv2box import CVFile

# init chatglm-6b model
# model_name = 'THUDM/chatglm-6b'
model_name = '/mnt/ljt/models/hugging_face/chatglm-6b'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True, device_map='auto')

# global init
prompt_row_name = 'instruction'
target_row_name = 'output'


def preprocess(example: dict, max_seq_length):
    prompt = example[prompt_row_name]
    target = example[target_row_name]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


# def read_jsonl(path, max_seq_length, skip_overlength=False):
#     with open(path, "r") as f:
#         for line in tqdm(f.readlines()):
#             example = json.loads(line)
#             feature = preprocess(example, max_seq_length)
#             if skip_overlength and len(feature["input_ids"]) > max_seq_length:
#                 continue
#             feature["input_ids"] = feature["input_ids"][:max_seq_length]
#             yield feature


def read_json(path, max_seq_length, skip_overlength=False):
    """
    for alpaca-COT(https://github.com/PhoebusSi/Alpaca-CoT) format datasets
    """
    json_data = CVFile(path).data
    for example in tqdm(json_data):
        feature = preprocess(example, max_seq_length)
        if skip_overlength and len(feature["input_ids"]) > max_seq_length:
            continue
        feature["input_ids"] = feature["input_ids"][:max_seq_length]
        yield feature



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/ljt/dataset/NLP/liurun_99.json")
    parser.add_argument("--save_path", type=str, default="/mnt/ljt/dataset/NLP/liurun_99")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: read_json(args.data_path, args.max_seq_length, args.skip_overlength)
    )
    dataset.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
