# coding:utf8
#猫眼电影抓取,保存名称、主演、时间、评分、图片连接到文件

import requests
import re
import json
from multiprocessing.dummy import Pool


def get_one_page(url):
    print('Crawling ' + url)
    response=requests.get(url)
    if response.status_code==200:
        return response.text
    return None

    
def parse_one_page(html):
    pattern = re.compile(
        '<dd>.*?board-index.*?>(.*?)</i>.*?data-src="(.*?)".*?name.*?a.*?>(.*?)</a>.*?star.*?>(.*?)</p>.*?releasetime.*?>(.*?)</p>.*?integer.*?>(.*?)</i>.*?fraction.*?>(.*?)</i>.*?</dd>',
        re.S)
    items=re.findall(pattern, html)
    for item in items:
        yield{
              'index':item[0],
              'image':item[1],
              'title':item[2].strip(),
              'actor':item[3].strip()[3:]if len(item[3])>3 else '',
              'time': item[4].strip()[5:] if len(item[4]) > 5 else '',
              'score': item[5].strip() + item[6].strip()
              }

def write_to_json(content):
    with open('result.txt','a') as f:
        #print(type(json.dumps(content)))
        f.write(json.dumps(content, ensure_ascii=False) + '\n')
        f.close()
        
def main(offset):
    url = 'http://maoyan.com/board/4?offset=' + str(offset)
    html = get_one_page(url)
    for item in parse_one_page(html):
        #print(item)
        write_to_json(item)
       
pool =Pool()
pool.map(main, [i * 10 for i in range(10)])#产生10到100的数，传到main函数中
print('Finished') 