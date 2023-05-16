# -- coding: utf-8 --
# @Time : 2023/5/12
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import os
import platform
from accelerate import init_empty_weights
from accelerate.utils import get_balanced_memory, infer_auto_device_map
from transformers import AutoConfig, GenerationConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers

from gpt_lib.models.llm_base import LLM

MODEL_ZOO = {
    'llama-7b': {
        'model_path': '/mnt/ljt/models/hugging_face/llama-7b-hf',
        'model': LlamaForCausalLM,
        'tokenizer': LlamaTokenizer,
        'config': AutoConfig,
        'prompt_template': """Below is an instruction that describes a task. Write a response that appropriately completes the request.
                    ### Instruction:
                    {instruction}
                    ### Response:""",
    },
}


class LLAMA(LLM):
    def __init__(self, model_name, load_in_8bit):
        self.model_path = MODEL_ZOO[model_name]['model_path']
        self.config = MODEL_ZOO[model_name]['config'].from_pretrained(self.model_path, return_unused_kwargs=True,
                                                                      trust_remote_code=True)[0]
        super().__init__(MODEL_ZOO[model_name], load_in_8bit, self.get_device_map)

    def get_device_map(self):
        return 'auto'

    def generate(self, prompt):
        generation_config = GenerationConfig(temperature=0.1,
                                             top_p=0.75,
                                             top_k=40,
                                             num_beams=4,
                                             max_new_tokens=512,
                                             do_sample=True,
                                             no_repeat_ngram_size=6,
                                             repetition_penalty=1.8,
                                             )
        result = self.generate_base(prompt, generation_config)
        return result


if __name__ == '__main__':
    llama = LLAMA('llama-7b', load_in_8bit=True)
    print(llama.generate('你好'))
