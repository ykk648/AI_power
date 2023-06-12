# -- coding: utf-8 --
# @Time : 2023/4/27
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
https://github.com/suno-ai/bark
"""
from bark import SAMPLE_RATE, generate_audio, preload_models
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)