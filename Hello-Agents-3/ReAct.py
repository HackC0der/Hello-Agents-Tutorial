import requests
import json
import os
import dotenv

dotenv.load_dotenv()
url = "https://api.bochaai.com/v1/web-search"

payload = json.dumps({
  "query": "什么是Multi Agent架构",
  "freshness": "oneYear",
  "summary": True,
  "count": 8
})
headers = {
  'Authorization': os.getenv('BOCHAAI_API_KEY'),
  'Content-Type': 'application/json',
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.json())