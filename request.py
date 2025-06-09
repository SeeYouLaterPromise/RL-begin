import requests
r = requests.head("https://www.zhihu.com")
print(r.headers)  # 查看所有响应头