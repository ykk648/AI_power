# -- coding: utf-8 --
# @Time : 2023/4/20
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import os
import tqdm
import json
import torch
import loralib as lora
# import lora_utils.insert_lora
# import dataset.GLM as GLM_Data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import get_linear_schedule_with_warmup

checkpoint = "THUDM/chatglm-6b"
mixed_precision = 'bf16'
lora_config = {
    'r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'enable_lora': [True, True, True],
}
max_length = 256
LR = 2e-5
NUM_EPOCHS = 2
batch = 1
accumulate_step = 8
warm_up_ratio = 0.1

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, revision='main')
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, revision='main')
model = lora_utils.insert_lora.get_lora_model(model, lora_config)

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step,
                          deepspeed_plugin=deepspeed_plugin)
device = accelerator.device
GLM_Data.device = device

import dataset.Alpaca as Alpaca_Data

pairs = Alpaca_Data.load('./data/alpaca_data.json')
pairs_encoded = GLM_Data.encode_pairs(pairs, tokenizer)
pairs_encoded = list(filter(lambda pair: len(pair['prompt']) + len(pair['completion']) <= max_length, pairs_encoded))
train_dataset = GLM_Data.GLMDataset(pairs_encoded)
train_dataloader = DataLoader(dataset=train_dataset, collate_fn=GLM_Data.collate_fn, shuffle=True, batch_size=batch)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) / accumulate_step * warm_up_ratio),
    num_training_steps=(int(len(train_dataloader) / accumulate_step) * NUM_EPOCHS),
)

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
model.to(device).train()

total_step = 0
effective_step = 0

for epoch in range(NUM_EPOCHS):
    epoch_loss_local = 0
    for step, batch in enumerate(t := tqdm.tqdm(train_dataloader)):
        outputs = model(**batch)
        loss_d = outputs.loss.detach()
        epoch_loss_local += loss_d
        t.set_description(f"loss: {epoch_loss_local.cpu().float() / step}")
        loss = outputs.loss / accumulate_step
        accelerator.backward(loss)
        if (step + 1) % accumulate_step == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    accelerator.wait_for_everyone()
    all_epoch_loss, all_step = accelerator.gather((epoch_loss_local, torch.tensor(step, device=device)))

    if accelerator.is_main_process:
        model_id = f"finetune_{epoch}"
        accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), '/saved/' + model_id + '.pt')

        epoch_loss = all_epoch_loss.float().sum() / (all_step + 1).sum()
        total_step += (all_step + 1).sum()
        effective_step += ((all_step + 1) // accumulate_step).sum()
        print(f'epoch: {epoch}, step {effective_step.cpu().numpy()}, training_loss: {epoch_loss.cpu().numpy()}')

    accelerator.wait_for_everyone()