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
                        "url": "data:image/png;base64,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"
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
