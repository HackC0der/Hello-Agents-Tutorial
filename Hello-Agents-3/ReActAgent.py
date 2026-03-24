# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[你的完整最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

尽可能在<5次的询问中得到Finish的最终答案，目前询问次数={query_times}
现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

import requests
import json
import os
import re
import dotenv
from typing import Dict, Any
dotenv.load_dotenv()
from LLM import LLM
from ReAct import ToolExecutor, search
class ReActAgent:
    """
    ReAct不断“推理-执行”迭代直至超过最大限度或解决问题的智能体
    """
    def __init__(self, llm: LLM, tool_executor: ToolExecutor, max_steps : int = 5):
        self.llm = llm
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
    
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

if __name__ == "__main__":
    llm = LLM()
    tool_executor = ToolExecutor()
    tool_executor.registerTool(name = "search",
					  description="一个网页搜索的工具，函数名为search，参数为{query: str})",
					  func=search)
    agent = ReActAgent(llm=llm,tool_executor=tool_executor)
    agent.run("华为最新的手机有什么特性")

