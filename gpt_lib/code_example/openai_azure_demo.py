
import openai  # 需要pip安装
import prompt_toolkit  # 需要额外安装这个库，用于命令行交互

openai.api_type = "azure"
openai.api_base = "https://shanghai-free-test.openai.azure.com/"  # 这里需要根据自己的资源进行更改
# openai.api_version = "2022-12-01"
openai.api_version = "2023-03-15-preview"

# 配置OpenAI API密钥
openai.api_key = ''  # 这里根据自己的API KEY更改
# 设定OpenAI的模型和引擎
model_engine = "gpt-35-turbo"  # 这里就是创建的模型名称更改
prompt_prefix = "我: "
response_prefix = "AI: "

# import requests
# url = openai.api_base + "/openai/deployments?api-version=2022-12-01"
# r = requests.get(url, headers={"api-key": openai.api_key})
# print(r.text)

# 定义一个函数，用于向OpenAI API发送请求并返回结果
def generate_response(prompt):
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=["\n"],
        temperature=0,
    )
    message = response.choices[0].text
    return message.strip()


# 通过Prompt Toolkit库来实现命令行交互
def prompt_user():
    while True:
        try:
            # 读取用户输入的信息
            user_input = prompt_toolkit.prompt(prompt_prefix)
            # user_input = 'What\'s the difference between garbanzo beans and chickpeas? '
            print(user_input)
            # 将用户输入发送给OpenAI API，并返回结果
            response = generate_response(user_input)
            # 打印OpenAI API返回的结果
            print(response_prefix + response)
        except KeyboardInterrupt:
            # 如果用户按下Ctrl-C，则退出程序
            print("\n再见!")
            break


# 运行程序
if __name__ == "__main__":
    prompt_user()
