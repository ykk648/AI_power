# -- coding: utf-8 --
# @Time : 2023/2/6
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import os
import openai as ai


# Get the key from an environment variable on the machine it is running on
# ai.api_key = os.environ.get("OPENAI_API_KEY")


def generate_gpt3_response(user_text, print_output=False):
    """
    Query OpenAI GPT-3 for the specific key and get back a response
    :type user_text: str the user's text to query for
    :type print_output: boolean whether or not to print the raw output JSON
    """
    completions = ai.Completion.create(
        engine='text-davinci-003',  # Determines the quality, speed, and cost.
        temperature=0.5,  # Level of creativity in the response
        prompt=user_text,  # What the user typed in
        max_tokens=3500,  # Maximum tokens in the prompt AND response
        n=1,  # The number of completions to generate
        stop=None,  # An optional setting to control response generation
    )

    # Displaying the output can be helpful if things go wrong
    if print_output:
        print(completions)

    # Return the first choice's text
    return completions.choices[0].text


if __name__ == '__main__':
    os.environ.setdefault("OPENAI_API_KEY", '')
    ai.api_key = os.environ.get("OPENAI_API_KEY")

    text = '写一篇乔·拜登的演讲稿'
    results = generate_gpt3_response(text)
    print(results)
    # print(results.decode())
