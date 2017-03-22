# coding:utf8
#requests库学习
import requests
import json

#1.基本请求
r=requests.post("http://httpbin.org/post")
r=requests.put("http://httpbin.org/put")
r=requests.delete("http://httpbin.org/delete")
r=requests.head("http://httpbin.org/head")
r=requests.options("http://httpbin.org/get")
#2.基本get请求   
r = requests.get("http://httpbin.org/get")
#加参数
payload = {'key1': 'value1', 'key2': 'value2'}
r = requests.get("http://httpbin.org/get", params=payload)
print (r.url)
#请求JSON文件
r = requests.get("a.json")
print (r.text)
print (r.json())
#加headers参数
payload = {'key1': 'value1', 'key2': 'value2'}
headers = {'content-type': 'application/json'}
r = requests.get("http://httpbin.org/get", params=payload, headers=headers)
print (r.url)
#3.基本post请求
payload = {'key1': 'value1', 'key2': 'value2'}
r = requests.post("http://httpbin.org/post", data=payload)
print (r.text)
#将表单数据序列化
url = 'http://httpbin.org/post'
payload = {'some': 'data'}
r = requests.post(url, data=json.dumps(payload))
print (r.text)
#上传文件
url = 'http://httpbin.org/post'
files = {'file': open('test.txt', 'rb')}
r = requests.post(url, files=files)
print (r.text)
#3.cookies
url = 'http://example.com'
r = requests.get(url)
print (r.cookies)
print (r.cookies['example_cookie_name'])
#向服务器发送cookies信息
url = 'http://httpbin.org/cookies'
cookies = dict(cookies_are='working')
r = requests.get(url, cookies=cookies)
print (r.text)
#4.超时配置
requests.get('http://github.com', timeout=0.001)
#5.会话，在这里我们请求了两次，一次是设置 cookies，一次是获得 cookies
s = requests.Session()
s.get('http://httpbin.org/cookies/set/sessioncookie/123456789')
r = s.get("http://httpbin.org/cookies")
print(r.text)
#全局配置
s = requests.Session()
s.headers.update({'x-test': 'true'})
r = s.get('http://httpbin.org/headers', headers={'x-test2': 'true'})
#消除全局配置
r = s.get('http://httpbin.org/headers', headers={'x-test': None})
#6.SSH证书验证
r = requests.get('https://kyfw.12306.cn/otn/', verify=True)
print (r.text)
#跳过验证
r = requests.get('https://kyfw.12306.cn/otn/', verify=False)
#7.代理
proxies = {
  "https": "http://41.118.132.69:4433"
}
r = requests.post("http://httpbin.org/post", proxies=proxies)
#也可以通过环境变量 HTTP_PROXY 和 HTTPS_PROXY 来配置代理
#export HTTP_PROXY="http://10.10.1.10:3128"
#export HTTPS_PROXY="http://10.10.1.10:1080"


print (r.text)