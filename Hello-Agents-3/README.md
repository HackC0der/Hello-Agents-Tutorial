# 构建大模型智能体基础编程



## 一、环境准备

安装基础依赖库

```python
uv pip install openai python-dotenv
```

- `openai`用于提供标准的与LLM进行交互的接口
- `python-dotenv`用于提供读写环境变量的工具，安全存储敏感`key`\\`token`数据

`.env`

```text
# .env file
LLM_API_KEY="YOUR-API-KEY"
LLM_MODEL_ID="YOUR-MODEL"
LLM_BASE_URL="YOUR-URL"
```

## 二、openai 库基础函数

连接大模型的接口函数`OpenAI`

```python
client = OpenAI(
    api_key="这里是获取的api_key",
    base_url="https://api.aihao123.cn/luomacode-api/open-api/v1"
)
```

1. 文本生成模型对话

```python 
response = client.chat.completions.create(
    # 必选参数
    messages=[
        {"role": "system", "content": "你是一位专业的科技撰稿人，语言风格简洁易懂。"},  # 系统角色，设定模型行为
        {"role": "user", "content": "写一篇关于人工智能的文章。"}  # 用户问题
    ],
    model="gpt-3.5-turbo",
    
    # 可选参数
    stream=False,  # 非流式返回
    max_tokens=500,  # 最大生成token数
    temperature=0.7,  # 适度的随机性
    top_p=1.0,  # 不限制核采样
    n=1,  # 生成1个结果
    stop=None,  # 无停止序列
    presence_penalty=0.0,  # 不调整话题倾向
    frequency_penalty=0.0,  # 不调整重复倾向
)
# 非流式，输出结果
print(response.choices[0].message.content)

## 如果stream = True
## 逐行打印流式结果
#for chunk in response:
#    if chunk.choices[0].delta.content is not None:
#        print(chunk.choices[0].delta.content, end="", flush=True)
```

- 必选参数
  - `messages`：对话上下文
  - `model`：模型版本
- 关键可调参数
  - `temperature`：随机性
  - `max_tokens`：生成的最大长度
  - `stream`：是否以流式返回
- 上下文控制
  - `system` 角色的 `messages` 能有效设定模型的行为，是优化回复质量的关键技巧

2. 代码补全模型

与文本生成模型很类似，是专门为代码补全调优过的生成模型，例如：

```python
response = client.chat.completions.create(
    messages=[
        {'role': 'user', 'content': "1+1"}, ],
    model='gpt-3.5-turbo',
    stream=False,
    max_tokens=200
)
print(response.choices[0].message.content)
#>> 1+1 equals 2.
```

3. 图像多模态大模型

```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "这是什么？",
                    "type": "text"
                },
                {
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJUAAABNCAYAAACvzyYNAAAKnUlEQVR4nO3ceXgU9R3H8ffM7GavHOQi94UhBAtiURQrilgFeVAoiGCliq1FKqXWo/Wp1Av7UMQHi/WgqBQfrKg8FWwRH7EqV1G5HkCOkGxCyEXuO7vZc2b6x0IwLgF8nNYl/b3+y87Mb36ZfPb7+81vJyu1tLbrCIKB5O+6A0L/I0IlGM7k86soskRsjP277ovQT5hUTUfTdKxWy3fdF6GfEMOfYDgRKsFwIlSC4USoBMOJUAmGE6ESDCdCJRhOhEownAiVYDgRKsFwIlSC4USoBMOJUAmGE6ESDCdCJRhOhEownMnIxh5cVMQHnzYY2aTwP+TcPM6QdkSlEgwnQiUYToRKMJwIlWA4ESrBcCJUguFEqATDiVAJhhOhEgwnQiUYToRKMJwIlWA4ESrBcBEVqpxsB9MnZWC39354wmZViB8QRVKihdRkC+kpVkZ8bwDzZg9i+aJLGHVpvGF9yM+NZtjQOJBCP9sdJh6ak89llww463FpqVaWL7qE++8Z1Oc+d92axabVV5KX4zCsv5HI0EdfvhUJhg+JYe7tWQwviGHFWxXUNXgBmHxjKhPHJmO3mbBZZWwWBVXTaWkPUNvoxWFTQiE4+e2liklmYOKZv2/LG9Bob/eja+FfdWq1KTw6L5+keDNT5+5B0yEzzcYt1ydjtUgcLunE59PO2O7gHAf5WXaOVXYTGxcFgMcTJOA/vf+AGBM2i4LFElHvZcNFTqh0+Hh7I6lJFmZNzsDhUFj4fAldriDpyRZyM+288Y8TVNR4KC7rQlM1fEGdgF/D41V7AgUwrDCWJ+cPPuNpGpp9LHyhhPpGX9i2px8oJD/bzktvVqKdzILzWBc7v+zguisSWb+pnuKyrvBGJSjMj8ESJTPuygRGXxqqah9ua2TV2iqjrtAFI3JCBfh8GivfrkSWJSaMSSI7086R4s6e7aveqTyvdvJzHCTEmamu89DpCvba1tTmRw32rlKSLHHDmGQuHx7L4dIutu9uOb1RhxdeP87LC4ex+LeFzFlwkOaW3oG0WhVuvDqJylovVbXuntcbW/08PDef/OzQV1+mJoWq54JfDMbtCfVr664W1r5/oteb4kIXUaE6ZfW7VezY00JJuavX66lptj6P6eoK4D4ZoOx0Kz6/xoq3Kvlsb+s5zzdyeBw/uy0LVdVZsaaKhkZvr+31zV7WbDjB3NuzeWz+YP60spyqE9092++enk20XeHZV4+x50Bbz+vJCVFM+WEKKUkW9hd1EhdjBqCtM0B7Z4CxVyRQetyNJEnoev9JVUSGyufTKHL2HmYkYPWSEX0es3lnC0uWlwKQkmTB69dwe9VznusHoxJ56v7BSMCcBYeoqHaH7aOrOus/qiMvy87N1w3kifsLeODpQ7jcKmmpNmZOTKWmwcvFBTFclHd6Er57fzsAXxZ38vQLJfzqzjymTkhl5dpKikq7+Nfq0d/oulwoIiZUliiZx389hLSU0BDR1OJn8culeH0qMdEm/AG9Z1iSJIlxoxOpqffiPB6qZvsOd/S0lRBnxh/QyEy1UZAXTUqSBWe5i/omL4edXahBHUmRmDUlk5/emklbR4A179cSDGpknqUavrPhBHabwqhhcaxfMYoFS0tobvWx90gHXp/GzIlpOGwKbo+Kx6tSVeMBQNNBDepoJ6tRQNXDhuD+JGJCpelQWeuhvTPAkDwHQ/OiMZtlZEUiOdFCdb2HRS+FKpFikrjq+/Fs3tnM6r9Xh7UVF20iLtrEfXfk4LAraCooCnS5g7y2tpoNn9QDEFA1tu9pY9eBVqZNSGPy9QPP2U9nhZulfy3nrqmZtHcEqKju5rHnSrDbFGbenMGMm1LZuKWRLV80094Z+C9cqcgXMaEKBDReW1MBwLRJGcyekgGAw6aQGG9m7+HOc7RwmsUsYzbL7D7Uxu+fKyZKkZg5OYN7pmcxb1YOh0o6Ka90s/afJwDITLcxdnQidpsJkwy5GXbsVpkjZS7UkwUlxq6Qm26juc3Pti+a2fZFc8/5gkENl0unodmHDjS2+CircJM4IDSHkqTQG0GWQotfZkVCMUnGXbwIEzGh6kuMw0RCnJldX5kAn5UEG7c24PFqbPy0AUnTCWg6b66rJi7axIyJaeRm2imvPD13qqn18OgzRwEYlOtg0UNDaGjRmf/EoZ59LsqLZunvhhIIhA9br/5xBNlpVhRFQpEl5s3KYe7t2ewrCr0Rrrp0AG8uG0lcdOhy/+HBIXh8GorSP4MV8atw469JxqTIHC4+z0qlw8q3q1jzXg0dXxt+3t1UB8BVIxP6PDw73UZKooWNW3r//6LDpqAoUs9SwFe9uLqCx5c52fBpI6qqs3FzI48vc/Luh7UAtHeqfPJ5C9X1obvKPYc6+OTzlrB2+ouIrlR2m8KUG1LYuqsVV3f4H/NMsjLtPPtIIRU1Hha+6MTrOX0HWJAXDUBx+RkWMAFJkbjjlgx8AY11m+r4ah2JjTFhNkm0doTPkw4cCd3lDUyyoOtQXtPNzn2t2KwK/97bSl2zn4+3NuCw5FGY52DdR3UUO7uw22SOV3ej6dCfalbkhUqCnHQbZrPERTkOXG6VrTubz/tuqb7Bi6bBiKGxTB2fxjvvh+ZNFxfEcN+Ps/EHNA4WhVc9s1nmzmmZFOQ6eH19DdLXPsZJiDVjNsnUNvnP+1cJqjpvrKvuc2HzxVXlp37lfiWyQiXB/LvyuOmaJCxmmZ9MzuDxP5fgLHOF7erxqXR1h69DBQIaS1cdZ+kjhcyelsnEa5MJqjqJA8zExZj5YGsjVbXdvY4xm2XumZHNzEnpFB1z8cHm3kOfLkkkxUdhNkm9li764rCbuHdWLoWDHCx55VjYYmp/FzmhkmD8tQOZNG4gVXUejpa5GH91Ek/ML+DtjbUUOTsJnKpWOjz67FGQJAYPiu6ZGDor3Oiazv6DbTz8TBE/n5FNQpwZi0Wmqc3P5/vaeX5VOYFA6IM9SZbISrNy76xcxoyM50SDl0XLy/B6VQblOEIVRoLUZAvTJ6bR4QpSXRO+OHqq/8MGR2MyScy5LYtOdxDncTfmfnyX15eICZXZJDPm8gRc3SpPLnNS1+jlQFEHE8cOZPaPMnDYs5FlqWeo0PXQrfopgaDOhLt3op78IPjA4Q5+eeQw6SkWrBaFmjpPrycGAEwmmaceGEJWqpVN25pY/V4NdfUeLhsRz+LfDEHXQuewRMm4PSrL1/T92aOORLdPo6yqm+27WzhS6uLLoo6eAP8/iZhQBQIaf3uvBr+qU1sfWone/Fkzn+1txWSWkSWQzzL70AhfpZZ0nbr6voeegF9lyavlJMZHsWd/a08ADh7tYPErx3r28/h0So510t7e92KmpOv85Y3jSIqEz6ed8dGafUWdxMZF0dTSvxdFIyZUAKXl4XMnn0/r8xkmI5SUhk/aA36NLTuavnFbPv/Z+7ljdws7dvffpYRTIn6dSrjwiFAJhhOhEgwnQiUYToRKMJwIlWA4ESrBcCJUguFEqATDiVAJhhOhEgwnQiUYTqqubdYlICMt8Vs3pqqn/7dNuPCYTcbUGEOfUlAUCaXfPRwrfFNi+BMMJ0IlGE6ESjCcCJVgOBEqwXAiVILhRKgEw4lQCYYToRIMJ0IlGO4//znZnKvJJTsAAAAASUVORK5CYII="
                    },
                    "type": "image_url"
                }
            ]
        }
    ],
    model='gpt-4o-2024-05-13',
    stream=False,
    max_tokens=200
)

print(response.choices[0].message.content)
```

- `image_url`：参数里面的`image_url`传入的是经过`Base64`编码的图像，也可以传入`URL`，但是传`Base64`响应会非常的快。
- 图像转`base64`有很多工具，例如：[图片-Base64互转](https://www.67tool.com/images/convert/base64?identity=undefined)

## 三、简单Agent类封装

首先定义一个配置类，用于传参

```python
class Config:
    """配置常量"""
    API_KEY = "API_KEY"
    BASE_URL = "BASE_URL"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    MODEL_ID = "MODEL_ID"
    DEFAULT_TIMEOUT = 10
```

定义一个简单的`Agent`类，在`__init__`时初始化连接`llm`，`think`方法接受上下文内容、输出大模型反馈。利用`logging`库对日志进行记录或回显

```python
class Agent:
    """
    基础Agent类，包括初始化连接远程大模型，处理请求与输出大模型响应
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: Optional[int] = None, model_id: Optional[str] = None):
        """
        初始化各种参数，优先采用传入的参数，若未传入则从环境变量中获取。建立与远程大模型的连接
        :param api_key: 大模型API密钥，默认从环境变量中获取
        :param base_url: 大模型API基础URL，默认从环境变量中获取
        :param timeout: 请求超时时间，默认从环境变量中获取，若未指定则为10秒
        :param model_id: 大模型模块ID，默认从环境变量中获取
        """
        api_key = api_key or os.getenv(Config.API_KEY)
        base_url = base_url or os.getenv(Config.BASE_URL)
        timeout = timeout or int(os.getenv(Config.LLM_TIMEOUT, Config.DEFAULT_TIMEOUT))
        self.model = model_id or os.getenv(Config.MODEL_ID)

        missing = [k for k, v in [("API_KEY", api_key), ("BASE_URL", base_url), ("MODEL_ID", self.model)] if not v]
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
```

然后在`think()`方法中封装与大模型交互的过程

```python
class Agent:
    def think(self, messages: List[Dict[str, str]], temperature: float = 0, verbose: bool = False) -> Optional[str]:
        """
        接收来自用户的请求，调用大模型API进行处理，返回大模型的响应
        :param messages: 包含用户请求的消息列表，每个消息是一个字典，包含"role"和"content"键
        :param temperature: 大模型的温度参数，用于控制响应的随机性，默认值为0
        :param verbose: 是否打印大模型的响应，默认值为False
        :return: 大模型的响应字符串，若发生异常则返回None
        """
        if verbose:
            logger.info("大模型开始思考...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=temperature,
            )
            if verbose:
                logger.info("大语言模型响应成功")

            # 处理流式响应输出
            content = []
            for chunk in response:
                # 安全地获取delta内容，避免重复访问
                delta_content = chunk.choices[0].delta.content if chunk.choices else None
                if delta_content:
                    if verbose:
                        print(delta_content, end="", flush=True)
                    content.append(delta_content)

            if verbose:
                print()
            return "".join(content)

        except openai.APIError as e:
            logger.error(f"OpenAI API错误: {e}")
            return None
        except openai.APIConnectionError as e:
            logger.error(f"连接错误: {e}")
            return None
        except Exception as e:
            logger.error(f"未预期的异常: {e}")
            return None
```

最后进行测试：

```python
if __name__ == "__main__":
    agent = Agent()
    messages = [{"role": "user", "content": "请你解释三体运动的求解难在哪"}]
    response = agent.think(messages, verbose=True)
    if response:
        logger.info("大模型已正确响应")
    else:
        logger.warning("大模型响应为空")
```

```text
INFO:__main__:大模型开始思考...
INFO:httpx:HTTP Request: POST https://api.huiyan-ai.cn/v1/chat/completions "HTTP/1.1 200 OK"
INFO:__main__:大语言模型响应成功
三体运动的问题指的是在经典力学中，研究三个物体在相互引力作用下如何运动的数学和物理问题。虽然两体问题（例如地球绕太阳运动）有解析的精确解，但三体问题极其复杂，至今没有一般的解析解。三体运动求解难的主要原因包括以下几点：

1. **非线性耦合方程**  
   三个物体之间的引力相互作用通过牛顿运动方程描述，形成一组非线性耦合的二阶微分方程。非线性使得系统的行为非常复杂，微小的初始条件变化可能导致完全不同的轨迹（即混沌特性）。

2. **无解析通解**  
   两体问题由于可以通过开普勒定律和中心力场分析，得出精确解析解。但三体问题的运动方程无法化简为单独积分的形式，没有已知的一般解析表达式。这是因为三个质点的相互作用较两体更为复杂，无法用简单的数学函数描述。

3. **混沌动力学特性**  
   三体系统典型表现出混沌行为，系统对初始条件极端敏感，长期预测极难。即使非常接近的起始状态，经过一段时间后轨迹也会完全不同，这限制了数值计算的有效性和可靠性。

4. **高维相空间和复杂运动模式**  
   三体系统的相空间维度更大，运动模式丰富多样，包括稳定轨道、周期轨道、准周期轨道以及不规则运动。识别和分类各种可能的解非常困难。

5. **数值计算的挑战**  
   由于无解析解，常用数值方法（如Runge-Kutta方法）求解。但数值积分的误差积累和混沌特性导致长期模拟误差难以控制。此外，当三体接近发生近距离“碰撞”或“绕转”时，数值方法的稳定性和精度难以保证。

总结来说，三体运动的问题求解难主要是因为系统的高度非线性、多体交互的复杂性、混沌行为的存在及缺乏通用的解析解，导致理论分析和数值模拟都非常具有挑战性。
INFO:__main__:大模型已正确响应
```

完整代码在`AgentDemo.py`中实现

## 四、ReAct范式

- **思维链 (Chain-of-Thought)**，它能引导模型进行复杂的逻辑推理，但无法与外部世界交互，容易产生事实幻觉
- “纯行动”型，模型直接输出要执行的动作，但缺乏规划和纠错能力

ReAct的巧妙之处在于，它认识到**思考与行动是相辅相成的**。思考指导行动，而行动的结果又反过来修正思考。为此，ReAct范式通过一种特殊的提示工程来引导模型，使其每一步的输出都遵循一个固定的轨迹：

1. **Thought (思考)：** 这是智能体的“内心独白”。它会分析当前情况、分解任务、制定下一步计划，或者反思上一步的结果。
2. **Action (行动)：** 这是智能体决定采取的具体动作，通常是调用一个外部工具，例如 `Search['华为最新款手机']`。
3. **Observation (观察)：** 这是执行`Action`后从外部工具返回的结果，例如搜索结果的摘要或API的返回值。

智能体将不断重复这个 **Thought -> Action -> Observation** 的循环，将新的观察结果追加到历史记录中，形成一个不断增长的上下文，直到它在`Thought`中认为已经找到了最终答案，然后输出结果。这个过程形成了一个强大的协同效应：**推理使得行动更具目的性，而行动则为推理提供了事实依据。**

![ReAct范式中的“思考-行动-观察”协同循环](https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/4-figures/4-1.png)

我们将构建一个具备**使用外部工具**能力的ReAct智能体，来回答一个大语言模型仅凭自身知识库无法直接回答的问题。例如：“华为最新的手机是哪一款？它的主要卖点是什么？” 这个问题需要智能体理解自己需要上网搜索，调用工具搜索结果并总结答案。

### （一）定义工具

如果说大语言模型是智能体的大脑，那么**工具 (Tools)** 就是其与外部世界交互的“手和脚”。为了让ReAct范式能够真正解决我们设定的问题，智能体需要具备调用外部工具的能力。

[教程]([第四章 智能体经典范式构建](https://datawhalechina.github.io/hello-agents/#/./chapter4/第四章 智能体经典范式构建?id=_422-工具的定义与实现))中使用**SerpApi**库进行网页信息的获取。由于该库是`google`的搜索接口，国内会被墙（可能吧，没试过），故代码实现过程我采用`Web Search API`，详见：[Bocha Web Search API：国内可用的SerpAPI，可以平替Bing Web Search API，Google Web Search API_serpapi 国内使用-CSDN博客](https://blog.csdn.net/cxk19980802/article/details/142979103)

接下来，我们通过代码来定义和管理这个工具。我们将分步进行：首先实现工具的核心功能，然后构建一个通用的工具管理器。

#### 1. 实现搜索工具的核心逻辑

一个良好定义的工具应包含以下三个核心要素：

1. **名称 (Name)**： 一个简洁、唯一的标识符，供智能体在 `Action` 中调用，例如 `Search`。
2. **描述 (Description)**： 一段清晰的自然语言描述，说明这个工具的用途。**这是整个机制中最关键的部分**，因为大语言模型会依赖这段描述来判断何时使用哪个工具。
3. **执行逻辑 (Execution Logic)**： 真正执行任务的函数或方法。

我们的第一个工具是 `search` 函数，它的作用是接收一个查询字符串，然后返回搜索结果。

```python
import requests
import json
import os
import dotenv

dotenv.load_dotenv()



def search(query: str) -> str:
	"""
	一个基于博查API的网页搜索引擎工具，
	"""
	try:
		url = "https://api.bochaai.com/v1/web-search"
		payload = json.dumps({
				"query": query,
				"freshness": "oneYear",
				"summary": True,
				"count": 8
			})
		headers = {
			'Authorization': os.getenv('BOCHAAI_API_KEY'),
			'Content-Type': 'application/json',
		}
		result = ''
		response = requests.request("POST", url, headers=headers, data=payload).json()
		if 'data' in response and 'webPages' in response['data'] and 'value' in response['data']['webPages']:
			response = response['data']['webPages']['value']
			for r in response:
				result = result + '\n----------\n'+ ': ' + r.get('summary')
		return result
	except Exception as e:
		return f'搜索时发生错误：{e}'


if __name__ == "__main__":
	result = search("什么是Multi Agent架构")
	print(result)
```

#### 2. 构建通用工具调度执行器

当智能体需要使用多种工具时（例如，除了搜索，还可能需要计算、查询数据库等），我们需要一个统一的管理器来注册和调度这些工具。为此，我们创建一个 `ToolExecutor` 类

```python
class ToolExecutor:
	"""
	工具管理调度器
	"""
	def __init__(self):
		self.tools : Dict[str, Dict[str, Any]] = {}

	def registerTool(self, name: str, description: str, func: callable):
		"""
		向工具箱中注册一个新的工具
		"""
		if name in self.tools:
			pass
		self.tools[name] = {"description": description, "func": func}
	def getTool(self, name: str) -> callable:
		"""
		依据名称获取一个函数
		"""
		return self.tools.get(name,{}).get("func")
	def getAllAvailableTools(self) -> str:
		"""
		获取所有可用工具的格式化描述字符串
		"""
		return '\n'.join([
			f"- {name}: {info['description']}"
		] for name, info in self.tools.items())
```

#### 3. 测试

```python
import requests
import json
import os
import dotenv
from typing import Dict, Any
dotenv.load_dotenv()

class ToolExecutor:
	"""
	工具管理调度器
	"""
	def __init__(self):
		self.tools : Dict[str, Dict[str, Any]] = {}

	def registerTool(self, name: str, description: str, func: callable):
		"""
		向工具箱中注册一个新的工具
		"""
		if name in self.tools:
			pass
		self.tools[name] = {"description": description, "func": func}
		print(f"\n-------已注册工具：{name}-------")
	def getTool(self, name: str) -> callable:
		"""
		依据名称获取一个函数
		"""
		print(f"\n-------调用工具：{name}-------")
		return self.tools.get(name,{}).get("func")
	def getAllAvailableTools(self) -> str:
		"""
		获取所有可用工具的格式化描述字符串
		"""
		print(f"\n-------所有工具-------")
		return '\n'.join([
			f"- {name}: {info['description']}"
		 for name, info in self.tools.items()])
def search(query: str) -> str:
	"""
	一个基于博查API的网页搜索引擎工具
	"""
	try:
		url = "https://api.bochaai.com/v1/web-search"
		payload = json.dumps({
				"query": query,
				"freshness": "oneYear",
				"summary": True,
				"count": 8
			})
		headers = {
			'Authorization': os.getenv('BOCHAAI_API_KEY'),
			'Content-Type': 'application/json',
		}
		result = ''
		response = requests.request("POST", url, headers=headers, data=payload).json()
		if 'data' in response and 'webPages' in response['data'] and 'value' in response['data']['webPages']:
			response = response['data']['webPages']['value']
			for r in response:
				result = result + '\n----------\n'+ ': ' + r.get('summary')
		return result
	except Exception as e:
		return f'搜索时发生错误：{e}'


if __name__ == "__main__":
	toolBox = ToolExecutor()
	toolBox.registerTool(name = "search",
					  description="一个网页搜索的工具",
					  func=search
					  )
	print(toolBox.getAllAvailableTools())
	result = toolBox.getTool('search')("什么是Multi Agent架构")
	print(result)
```

### （二）ReActAgent

将所有独立的组件，LLM客户端和工具执行器组装起来，构建一个完整的 ReAct 智能体。我们将通过一个 `ReActAgent` 类来封装其核心逻辑。

#### 1. ReActAgent核心prompt

提示词是整个 ReAct 机制的基石，它为大语言模型提供了行动的操作指令。我们需要精心设计一个模板，它将动态地插入可用工具、用户问题以及中间步骤的交互历史。

```prompt
# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""
```

这个模板定义了智能体与LLM之间交互的规范：

- **角色定义**： “你是一个有能力调用外部工具的智能助手”，设定了LLM的角色。
- **工具清单 (`{tools}`)**： 告知LLM它有哪些可用的“手脚”。
- **格式规约 (`Thought`/`Action`)**： 这是最重要的部分，它强制LLM的输出具有结构性，使我们能通过代码精确解析其意图。
- **动态上下文 (`{question}`/`{history}`)**： 将用户的原始问题和不断累积的交互历史注入，让LLM基于完整的上下文进行决策。

#### 2. 输出解释器

LLM 返回的是纯文本，我们需要从中精确地提取出`Thought`和`Action`。这是通过几个辅助解析函数完成的，它们通常使用正则表达式来实现。

```PYTHON
    def _parse_output(self, text: str):
        """
        提取Thought与Action两个字段
        """
        # Thought： 匹配到Action或到结尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s(.*?)(?=$)", text, re.DOTALL)
        thought_str = thought_match.group(1).strip() if thought_match else None
        action_str = action_match.group(1).strip() if action_match else None
        return thought_str, action_str
    def _parse_action(self, text: str):
        """
        提取tool_name与tool_input两个字段
        """
        toolinfo_match = re.search(r"(\w+)\[(.*?)\]", text, re.DOTALL)
        if toolinfo_match:
            return toolinfo_match.group(1).strip(), toolinfo_match.group(2).strip()
        return None, None
```



#### 3. 核心循环

`ReActAgent` 的核心是一个循环，它不断地“格式化提示词 -> 调用LLM -> 执行动作 -> 整合结果”，直到任务完成或达到最大步数限制。

```python
    def run(self, question: str = ""):
        self.history = []   # 每个问题重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"------- 第{current_step}步执行 -------")

            # 1. 格式化提示词
            alltool_desc = self.tool_executor.getAllAvailableTools()
            print(alltool_desc)
            history_str = '\n'.join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools = alltool_desc,
                question=question,
                history=history_str,
                query_times = current_step
            )
            print(prompt)

            # 2. 调用LLM进行思考
            message = [{'role':'user','content':prompt}]
            response_text = self.llm.think(messages=message)
            print(response_text)
            input()
            if not response_text:
                print("大模型调用失败！")
                return None
            
            # 3. 解析大模型输出
            thought, action = self._parse_output(response_text)
            if thought:
                print("thought:"+thought)
            
            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                observation = "未能解析出有效的Action"

            # 4. 处理action Finish
            if action.startswith("Finish"):
                # final_answer = action.strip("Finish")
                final_answer = re.match(r"Finish\[(.*?)\]", action).group(1)
                print(f"最终答案：「\n{final_answer}\n」")
                return final_answer
            
            # 5. 处理action tool-call
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print("警告:未能解析出有效的tool calls，流程终止。")
                observation = "未能解析出有效的tool calls"
                # continue
            else:
                print(f"执行动作：「{tool_name}({tool_input})」")
                tool_func = self.tool_executor.getTool(tool_name)
                if not tool_func:
                    observation = f"错误：不存在名称为{tool_name}的工具"
                else:
                    observation = tool_func(tool_input)
            print("观察结果：",observation)

            # 6. 记忆上下文更新
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("到达最大调用次数，动作停")
        return None
```

#### 4. 测试

```python
...
if __name__ == "__main__":
    llm = LLM()
    tool_executor = ToolExecutor()
    tool_executor.registerTool(name = "search",
					  description="一个网页搜索的工具，函数名为search，参数为{query: str})",
					  func=search)
    agent = ReActAgent(llm=llm,tool_executor=tool_executor)
    agent.run("华为最新的手机有什么特性")
========
最终答案：「
华为最新的手机特性包括：搭载新一代麒麟芯片，性能显著提升；配备大容量电池与超快充技术，续航能力增强；Pura 80系列摄像头硬件和算法升级，拍照功能更强；Pura X首发鸿蒙操作系统5.0，具有独特的屏幕和交互创新，如AI眼动翻页功能；采用16:10比例阔型屏设计，支持高刷新率和HDR Vivid显示技术；具备IPX8级防水功能等。
」
```

### （三）ReAct特点、局限与调试

#### 1. ReAct 的主要特点

- **高可解释性**：ReAct 最大的优点之一就是透明。通过 `Thought` 链，我们可以清晰地看到智能体每一步的“心路历程”——它为什么会选择这个工具，下一步又打算做什么。这对于理解、信任和调试智能体的行为至关重要。

- **动态规划与纠错能力**：与一次性生成完整计划的范式不同，ReAct 是“走一步，看一步”。它根据每一步从外部世界获得的 `Observation` 来动态调整后续的 `Thought` 和 `Action`。如果上一步的搜索结果不理想，它可以在下一步中修正搜索词，重新尝试。
- **工具协同能力**：ReAct 范式天然地将大语言模型的推理能力与外部工具的执行能力结合起来。LLM 负责运筹帷幄（规划和推理），工具负责解决具体问题（搜索、计算），二者协同工作，突破了单一 LLM 在知识时效性、计算准确性等方面的固有局限。

#### 2. ReAct 的固有局限性

- **对LLM自身能力的强依赖**：ReAct 流程的成功与否，高度依赖于底层 LLM 的综合能力。如果 LLM 的逻辑推理能力、指令遵循能力或格式化输出能力不足，就很容易在 `Thought` 环节产生错误的规划，或者在 `Action` 环节生成不符合格式的指令，导致整个流程中断。
- **执行效率问题**：由于其循序渐进的特性，完成一个任务通常需要多次调用 LLM。每一次调用都伴随着网络延迟和计算成本。对于需要很多步骤的复杂任务，这种串行的“思考-行动”循环可能会导致较高的总耗时和费用。
- **提示词的脆弱性**：整个机制的稳定运行建立在一个精心设计的提示词模板之上。模板中的任何微小变动，甚至是用词的差异，都可能影响 LLM 的行为。此外，并非所有模型都能持续稳定地遵循预设的格式，这增加了在实际应用中的不确定性。
- **可能陷入局部最优**：步进式的决策模式意味着智能体缺乏一个全局的、长远的规划。它可能会因为眼前的 `Observation` 而选择一个看似正确但长远来看并非最优的路径，甚至在某些情况下陷入“原地打转”的循环中。

#### 3. 调试技巧

当你构建的 ReAct 智能体行为不符合预期时，可以从以下几个方面入手进行调试：

- **检查完整的提示词**：在每次调用 LLM 之前，将最终格式化好的、包含所有历史记录的完整提示词打印出来。这是追溯 LLM 决策源头的最直接方式。
- **分析原始输出**：当输出解析失败时（例如，正则表达式没有匹配到 `Action`），务必将 LLM 返回的原始、未经处理的文本打印出来。这能帮助你判断是 LLM 没有遵循格式，还是你的解析逻辑有误。
- **验证工具的输入与输出**：检查智能体生成的 `tool_input` 是否是工具函数所期望的格式，同时也要确保工具返回的 `observation` 格式是智能体可以理解和处理的。
- **调整提示词中的示例 (Few-shot Prompting)**：如果模型频繁出错，可以在提示词中加入一两个完整的“Thought-Action-Observation”成功案例，通过示例来引导模型更好地遵循你的指令。
- **尝试不同的模型或参数**：更换一个能力更强的模型，或者调整 `temperature` 参数（通常设为0以保证输出的确定性），有时能直接解决问题。

## 五、Plan-and-Solve范式

与 ReAct 将思考和行动融合在每一步不同，Plan-and-Solve 将整个流程解耦为两个核心阶段，如图4.2所示：

1. **规划阶段 (Planning Phase)**： 首先，智能体会接收用户的完整问题。它的第一个任务不是直接去解决问题或调用工具，而是**将问题分解，并制定出一个清晰、分步骤的行动计划**。这个计划本身就是一次大语言模型的调用产物。
2. **执行阶段 (Solving Phase)**： 在获得完整的计划后，智能体进入执行阶段。它会**严格按照计划中的步骤，逐一执行**。每一步的执行都可能是一次独立的 LLM 调用，或者是对上一步结果的加工处理，直到计划中的所有步骤都完成，最终得出答案。

这种“先谋后动”的策略，使得智能体在处理需要长远规划的复杂任务时，能够保持更高的目标一致性，避免在中间步骤中迷失方向。

Plan-and-Solve 尤其适用于那些结构性强、可以被清晰分解的复杂任务，例如：

- **多步数学应用题**：需要先列出计算步骤，再逐一求解。
- **需要整合多个信息源的报告撰写**：需要先规划好报告结构（引言、数据来源A、数据来源B、总结），再逐一填充内容。
- **代码生成任务**：需要先构思好函数、类和模块的结构，再逐一实现。

### （一）规划阶段

我们通过提示词的设计，完成一个推理任务。这类任务的特点是，答案无法通过单次查询或计算得出，必须先将问题分解为一系列逻辑连贯的子步骤，然后按顺序求解。**我们的目标问题是：**“一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？”

智能体需要：

1. **规划阶段**：首先，将问题分解为三个独立的计算步骤（计算周二销量、计算周三销量、计算总销量）。
2. **执行阶段**：然后，严格按照计划，一步步执行计算，并将每一步的结果作为下一步的输入，最终得出总和。

````prompt
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

````

这个提示词通过以下几点确保了输出的质量和稳定性：

- **角色设定**： “顶级的AI规划专家”，激发模型的专业能力。
- **任务描述**： 清晰地定义了“分解问题”的目标。
- **格式约束**： 强制要求输出为一个 Python 列表格式的字符串，这极大地简化了后续代码的解析工作，使其比解析自然语言更稳定、更可靠。

```python
from LLM import llm
import re
import ast
class planner:
    """
    任务分解与规划专家
    """
    def __init__(self, llm):
        self.llm = llm
        self.PLANNER_PROMPT_TEMPLATE = """
            你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
            请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
            你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

            问题: {question}

            请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
            ```python
            ["步骤1", "步骤2", "步骤3", ...]
            ```
        """
    def plan(self, task: str) -> list[str]:
        """
        填充prompt模板，重新规划任务为
        """
        prompt = self.PLANNER_PROMPT_TEMPLATE.format(question = task)
        message = [{'role': 'user', 'content': prompt}]
        print("正在生成计划......")
        response_str = self.llm.think(message=message) or ""
        print("计划已经生成：", response_str)

        # 解析plan str为list[str]
        try:
            #找到```python与```之间的部分
            plan_match = re.match(r"```python(.*?)```",response_str)
            plan_list_str = plan_match.group(1).strip() if plan_match else "[]"
            plan_list = ast.literal_eval(plan_list_str)
            return plan_list if isinstance(plan_list, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_str}")
            return []            
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []            
```

### （二）执行器与状态管理

在规划器 (`Planner`) 生成了清晰的行动蓝图后，我们就需要一个执行器 (`Executor`) 来逐一完成计划中的任务。执行器不仅负责调用大语言模型来解决每个子问题，还承担着一个至关重要的角色：**状态管理**。它必须记录每一步的执行结果，并将其作为上下文提供给后续步骤，确保信息在整个任务链条中顺畅流动

执行器的提示词与规划器不同。它的目标不是分解问题，而是**在已有上下文的基础上，专注解决当前这一个步骤**。因此，提示词需要包含以下关键信息：

- **原始问题**： 确保模型始终了解最终目标。
- **完整计划**： 让模型了解当前步骤在整个任务中的位置。
- **历史步骤与结果**： 提供至今为止已经完成的工作，作为当前步骤的直接输入。
- **当前步骤**： 明确指示模型现在需要解决哪一个具体任务。

```prompt
self.EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，思考如何求解，写出算数求解过程（数学运算）与这一步的答案。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""
```

我们将执行逻辑封装到 `Executor` 类中。这个类将循环遍历计划，调用 LLM，并维护一个历史记录（状态）。

```python
class Executor:
    """
    多阶段任务执行与调度管理器
    """
    def __init__(self, llm: LLM):
        self.llm = llm
        self.EXECUTOR_PROMPT_TEMPLATE = """
            你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
            你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
            请你专注于解决“当前步骤”，思考如何求解，写出算数求解过程（数学运算）与这一步的答案。

            # 原始问题:
            {question}

            # 完整计划:
            {plan}

            # 历史步骤与结果:
            {history}

            # 当前步骤:
            {current_step}

            请仅输出针对“当前步骤”的回答:
        """
    def execute(self,question: str, tasks: List[str]) -> str:
        """
        生成专注于这一步的结果
        """
        print(f"开始处理任务：{{\n\tquestion: {question}\n\t",tasks,f"\n}}")
        history = ""
        for i, step in enumerate(tasks):
            print(f"正在执行步骤 {i+1}/{len(tasks)}: {step}")
            prompt = self.EXECUTOR_PROMPT_TEMPLATE.format(
                question = question,
                plan = tasks,
                history = history,
                current_step = step
            )
            message = [{"role": "user", "content": prompt}]
            response_str = self.llm.think(messages=message) or ""
            history += f"\n步骤 {i+1}-{step} 结果：{response_str}"
            print(f"步骤 {i+1}-{step} 已完成：{response_str}")
        final_answer = response_str
        return final_answer
```

现在已经分别构建了负责“规划”的 `Planner` 和负责“执行”的 `Executor`。最后一步是将这两个组件整合到一个统一的智能体 `PlanAndSolveAgent` 中，并赋予它解决问题的完整能力。我们将创建一个主类 `PlanAndSolveAgent`，它的职责非常清晰：接收一个 LLM 客户端，初始化内部的规划器和执行器，并提供一个简单的 `run` 方法来启动整个流程。

```python
class PlanAndSolveAgent:
    """
    接受一个llm输入，初始化planner和executor，然后run执行
    """
    def __init__(self, llm: LLM):
        self.llm = llm
        self.planner = Planner(self.llm)
        self.executor = Executor(self.llm)
    def run(self, question: str):
        """
        先规划，后执行
        """
        plan_list = self.planner.plan(question)
        if plan_list == []:
            print("未生成有效计划列表")
            return
        result = self.executor.execute(question=question,tasks=plan_list)
        print(f"最终的结果为：{result}")

if __name__ == "__main__":
    llm = LLM()
    agent = PlanAndSolveAgent(llm=llm)
    agent.run("一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？")
```

## 六、Reflection

在我们已经实现的 ReAct 和 Plan-and-Solve 范式中，智能体一旦完成了任务，其工作流程便告结束。然而，它们生成的初始答案，无论是行动轨迹还是最终结果，都可能存在谬误或有待改进之处。Reflection 机制的核心思想，正是为智能体引入一种**事后（post-hoc）的自我校正循环**，使其能够像人类一样，审视自己的工作，发现不足，并进行迭代优化。

其核心工作流程可以概括为一个简洁的三步循环：**执行 -> 反思 -> 优化**。

1. **执行 (Execution)**：首先，智能体使用我们熟悉的方法（如 ReAct 或 Plan-and-Solve）尝试完成任务，生成一个初步的解决方案或行动轨迹。这可以看作是“初稿”。

2. 反思 (Reflection)

   ：接着，智能体进入反思阶段。它会调用一个独立的、或者带有特殊提示词的大语言模型实例，来扮演一个“评审员”的角色。这个“评审员”会审视第一步生成的“初稿”，并从多个维度进行评估，例如：

   - **事实性错误**：是否存在与常识或已知事实相悖的内容？
   - **逻辑漏洞**：推理过程是否存在不连贯或矛盾之处？
   - **效率问题**：是否有更直接、更简洁的路径来完成任务？
   - **遗漏信息**：是否忽略了问题的某些关键约束或方面？ 根据评估，它会生成一段结构化的**反馈 (Feedback)**，指出具体的问题所在和改进建议。

3. **优化 (Refinement)**：最后，智能体将“初稿”和“反馈”作为新的上下文，再次调用大语言模型，要求它根据反馈内容对初稿进行修正，生成一个更完善的“修订稿”。

与前两种范式相比，Reflection 的价值在于：

- 它为智能体提供了一个内部纠错回路，使其不再完全依赖于外部工具的反馈（ReAct 的 Observation），从而能够修正更高层次的逻辑和策略错误。
- 它将一次性的任务执行，转变为一个持续优化的过程，显著提升了复杂任务的最终成功率和答案质量。
- 它为智能体构建了一个临时的**“短期记忆”**。整个“执行-反思-优化”的轨迹形成了一个宝贵的经验记录，智能体不仅知道最终答案，还记得自己是如何从有缺陷的初稿迭代到最终版本的。更进一步，这个记忆系统还可以是**多模态的**，允许智能体反思和修正文本以外的输出（如代码、图像等），为构建更强大的多模态智能体奠定了基础。

这一步的目标任务是：“编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。”

这个任务是检验 Reflection 机制的绝佳场景：

1. **存在明确的优化路径**：大语言模型初次生成的代码很可能是一个简单但效率低下的递归实现。
2. **反思点清晰**：可以通过反思发现其“时间复杂度过高”或“存在重复计算”的问题。
3. **优化方向明确**：可以根据反馈，将其优化为更高效的迭代版本或使用备忘录模式的版本。

### （一）记忆模块

Reflection 的核心在于迭代，而迭代的前提是能够记住之前的尝试和获得的反馈。因此，一个“短期记忆”模块是实现该范式的必需品。这个记忆模块将负责存储每一次“执行-反思”循环的完整轨迹。

实现一个简单的`Memory` 类，主体是这样的：

- 使用一个列表 `records` 来按顺序存储每一次的行动和反思。
- `add_record` 方法负责向记忆中添加新的条目。
- `get_trajectory` 方法是核心，它将记忆轨迹“序列化”成一段文本，可以直接插入到后续的提示词中，为模型的反思和优化提供完整的上下文。
- `get_last_execution` 方便我们获取最新的“初稿”以供反思。

```python
class Memory():
    def __init__(self):
        """
        records记录长期记忆
        """
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆列表中插入一条记忆
        参数：
        - record_type (str) 记录的类型：['execution','reflection']
        - content  (str) 记录的具体内容：如对生成代码的反思
        """
        record = {'type': record_type, 'content': content}
        self.records.append(record)
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")
    def get_trajectory(self) -> str:
        """
        将所有的记录都扁平化为一个字符串文本，构建提示词
        """
        trajectory = []
        for record in self.records:
            if record['type'] == 'execution':
                trajectory.append(f"--------上一轮执行结果---------\n{record['content']}")
            else:
                trajectory.append(f"--------对该结果的反思---------\n{record['content']}")
        return "\n\n".join(trajectory)
    def get_last_execution(self) -> Optional[str]:
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None
```

### （二）Reflection提示词设计

有了 `Memory` 模块作为基础，我们现在可以着手构建 `ReflectionAgent` 的核心逻辑。整个智能体的工作流程将围绕我们之前讨论的“执行-反思-优化”循环展开，并通过精心设计的提示词来引导大语言模型扮演不同的角色。

与之前的范式不同，Reflection 机制需要多个不同角色的提示词来协同工作。

1. **初始执行提示词 (Execution Prompt)** ：这是智能体首次尝试解决问题的提示词，内容相对直接，只要求模型完成指定任务。

```prompt
INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""
```

2. **反思提示词 (Reflection Prompt)** ：这个提示词是 Reflection 机制的灵魂。它指示模型扮演“代码评审员”的角色，对上一轮生成的代码进行批判性分析，并提供具体的、可操作的反馈。

````prompt
REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""
````

3. **优化提示词 (Refinement Prompt)** ：当收到反馈后，这个提示词将引导模型根据反馈内容，对原有代码进行修正和优化。

```prompt
REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""
```

### （三）Reflection智能体构建

```python
class ReflectionAgent:
    def __init__(self, llm: LLM, max_iterations = 3):
        self.llm = llm
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, question: str) -> Optional[str]:
        """
        首次执行-反思-重新生成-反思-...迭代
        """
        # 1. 首次任务执行
        init_prompt = INITIAL_PROMPT_TEMPLATE.format(task = question)
        response = self.easyCallLLM(init_prompt)
        self.memory.add_record("execution",response)

        # 反思-迭代生成
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")
            
            # 2. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task = question, code = last_code)
            feedback = self.easyCallLLM(reflect_prompt)
            self.memory.add_record("reflection", feedback)
            # 检查是否需要改进
            if "无需改进" in feedback:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # 3. 迭代改进
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task = question,
                last_code_attempt = last_code,
                feedback = feedback
            )
            refined_code = self.easyCallLLM(refine_prompt)
            self.memory.add_record("execution", refined_code)
        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code

    def easyCallLLM(self, prompt: str) -> str:
        message = [{'role': 'user', 'content': prompt}]
        response = self.llm.think(messages=message) or ""
        return response
```

