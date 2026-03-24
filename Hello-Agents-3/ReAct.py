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
		print(response)
		if 'data' in response and 'webPages' in response['data'] and 'value' in response['data']['webPages']:
			response = response['data']['webPages']['value']
			for r in response:
				result = result + '\n----------\n'+ ': ' + r.get('summary')
		return result
	except Exception as e:
		return f'搜索时发生错误：{e}'


# if __name__ == "__main__":
# 	toolBox = ToolExecutor()
# 	toolBox.registerTool(name = "search",
# 					  description="一个网页搜索的工具",
# 					  func=search)
# 	print(toolBox.getAllAvailableTools())
# 	result = toolBox.getTool('search')("什么是Multi Agent架构")
# 	print(result)