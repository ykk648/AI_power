# -- coding: utf-8 --
# @Time : 2023/5/9
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from transformers import AutoModel, AutoTokenizer
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers
from accelerate import init_empty_weights
from accelerate.utils import get_balanced_memory, infer_auto_device_map

import torch
import os
import platform
import signal

MODEL_ZOO = {
    'moss': {
        'model_path': '/mnt/ljt/models/hugging_face/moss-moon-003-sft',
        'model': AutoModel,
        'tokenizer': AutoTokenizer,
        'config': AutoConfig,
    },
}

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


class ChatGLM(LLM):
    def __init__(self, model_name, load_in_8bit=False):
        self.model_path = MODEL_ZOO[model_name]['model_path']
        self.config = MODEL_ZOO[model_name]['config'].from_pretrained(self.model_path, return_unused_kwargs=True,
                                                                      trust_remote_code=True)[0]
        super().__init__(MODEL_ZOO[model_name], load_in_8bit, self.get_device_map)
        self.model = self.model.half.cuda()

    def get_device_map(self):
        return 'auto'

    def stream_chat(self):
        self.model = self.model.eval()
        history = []
        global stop_stream
        print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
        while True:
            query = input("\n用户：")
            if query.strip() == "stop":
                break
            if query.strip() == "clear":
                history = []
                os.system(clear_command)
                print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
                continue
            count = 0
            for response, history in self.model.stream_chat(self.tokenizer, query, history=history):
                if stop_stream:
                    stop_stream = False
                    break
                else:
                    count += 1
                    if count % 8 == 0:
                        os.system(clear_command)
                        print(build_prompt(history), flush=True)
                        signal.signal(signal.SIGINT, signal_handler)
            os.system(clear_command)
            print(build_prompt(history), flush=True)
