# -- coding: utf-8 --
# @Time : 2023/4/17
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
hugging face use case
"""

from huggingface_hub import Repository
repo = Repository(local_dir="/mnt/models/hugging_face/Alpaca-CoT", clone_from="QingyiSi/Alpaca-CoT")