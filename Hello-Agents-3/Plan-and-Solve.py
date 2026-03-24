from LLM import LLM
import re
import ast
from typing import List

class Planner:
    """
    任务分解与规划专家
    """
    def __init__(self, llm: LLM):
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
        response_str = self.llm.think(messages=message) or ""
        print("计划已经生成：", response_str)

        # 解析plan str为list[str]
        try:
            #找到```python与```之间的部分
            plan_match = re.search(r"```python(.*?)```",response_str,re.DOTALL)
            plan_list_str = plan_match.group(1).strip() if plan_match else "[]"
            plan_list = ast.literal_eval(plan_list_str)
            print(plan_list)
            print(plan_list_str)
            print(plan_match)
            return plan_list if isinstance(plan_list, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_str}")
            return []            
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []            

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