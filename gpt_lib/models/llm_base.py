# -- coding: utf-8 --
# @Time : 2023/5/12
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import torch
from transformers import GenerationConfig


class LLM:
    def __init__(self, model_info, load_in_8bit, device_map):
        self.model_path = model_info['model_path']
        if 'self.config' not in locals():
            self.config = \
                model_info['config'].from_pretrained(self.model_path, return_unused_kwargs=True,
                                                     trust_remote_code=True)[0]
        self.model = model_info['model'].from_pretrained(self.model_path, trust_remote_code=True,
                                                         load_in_8bit=load_in_8bit, device_map=device_map())
        self.tokenizer = model_info['tokenizer'].from_pretrained(self.model_path, trust_remote_code=True)
        self.prompt_template = model_info['prompt_template']

    def generate(self, *args, **kwargs):
        pass

    def generate_prompt(self, prompt_in):
        return self.prompt_template.format(instruction=prompt_in)

    def generate_base(self, prompt, generation_config=None, **kwargs, ):
        prompt_format = self.generate_prompt(prompt)
        inputs = self.tokenizer(prompt_format, return_tensors="pt")

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=inputs["input_ids"].cuda(),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        return output.split("### Response:")[1].strip()
