import openai
import dotenv
import os
import logging
from typing import Optional, List, Dict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载.env环境变量
dotenv.load_dotenv()


class Config:
    """配置常量"""
    API_KEY = "API_KEY"
    BASE_URL = "BASE_URL"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    MODEL_ID = "MODEL_ID"
    DEFAULT_TIMEOUT = 10


class LLM:
    """
    基础LLM类，包括初始化连接远程大模型，处理请求与输出大模型响应
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

if __name__ == "__main__":
    llm = LLM()
    messages = [{"role": "user", "content": "请你解释三体运动的求解难在哪"}]
    response = llm.think(messages, verbose=True)
    if response:
        logger.info("大模型已正确响应")
    else:
        logger.warning("大模型响应为空")