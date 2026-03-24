from typing import List, Dict, Any, Optional
from LLM import LLM 
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

INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

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
        print(self.memory.get_trajectory())
        return final_code

    def easyCallLLM(self, prompt: str) -> str:
        message = [{'role': 'user', 'content': prompt}]
        response = self.llm.think(messages=message) or ""
        return response
    
if __name__ == "__main__":
    agent = ReflectionAgent(LLM(), 3)
    code = agent.run("编写一个Python函数，找出1到n之间所有的素数 (prime numbers)")
    with open("./tmp_output.txt",'w') as file:
        file.write(code)