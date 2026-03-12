# 调用开源大语言模型

> [第三章 大语言模型基础](https://datawhalechina.github.io/hello-agents/#/./chapter3/第三章 大语言模型基础?id=_323-调用开源大语言模型)

**Hugging Face Transformers** 是一个强大的开源库，它提供了标准化的接口来加载和使用数以万计的预训练模型。我们将使用它来完成本次实践。

**配置环境与选择模型**：为了让大多数读者都能在个人电脑上顺利运行，这里选择一个小规模但功能强大的模型：`Qwen/Qwen1.5-0.5B-Chat`。这是一个由阿里巴巴达摩院开源的拥有约 5 亿参数的对话模型，它体积小、性能优异，非常适合入门学习和本地部署。

安装依赖

```bash
uv pip install transformers torch
```

在 `transformers` 库中，我们通常使用 `AutoModelForCausalLM` 和 `AutoTokenizer` 这两个类来自动加载与模型匹配的权重和分词器。下面这段代码会自动从 Hugging Face Hub 下载所需的模型文件和分词器配置，这可能需要一些时间，具体取决于你的网络速度。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型ID
model_id = "Qwen/Qwen1.5-0.5B-Chat"

# 设置设备，优先使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("模型和分词器加载完成！")
```

我们来创建一个对话提示，Qwen1.5-Chat 模型遵循特定的对话模板。然后，可以使用上一步加载的 `tokenizer` 将文本提示转换为模型能够理解的数字 ID（即 Token ID）。

```python
# 准备对话输入
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请介绍你自己。"}
]

# 使用分词器的模板格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 编码输入文本
model_inputs = tokenizer([text], return_tensors="pt").to(device)

print("编码后的输入文本:")
print(model_inputs)

>>>
{'input_ids': tensor([[151644, 8948, 198, 2610, 525, 264,  10950, 17847, 13,151645, 198, 151644, 872, 198, 108386, 37945, 100157, 107828,1773, 151645, 198, 151644, 77091, 198]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       device='cuda:0')}
```

现在可以调用模型的 `generate()` 方法来生成回答了。模型会输出一系列 Token ID，这代表了它的回答。

最后，我们需要使用分词器的 `decode()` 方法，将这些数字 ID 翻译回人类可以阅读的文本。

```python
# 使用模型生成回答
# max_new_tokens 控制了模型最多能生成多少个新的Token
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

# 将生成的 Token ID 截取掉输入部分
# 这样我们只解码模型新生成的部分
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码生成的 Token ID
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n模型的回答:")
print(response)

>>>
我叫通义千问，是由阿里云研发的预训练语言模型，可以回答问题、创作文字，还能表达观点、撰写代码。我主要的功能是在多个领域提
供帮助，包括但不限于:语言理解、文本生成、机器翻译、问答系统等。有什么我可以帮到你的吗？
```

当你运行完所有代码后，你将会在本地电脑上看到模型生成的关于Qwen模型的介绍。恭喜你，你已经成功地在本地部署并运行了一个开源大语言模型！