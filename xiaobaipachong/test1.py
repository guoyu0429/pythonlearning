# coding:utf8
import requests
from bs4 import BeautifulSoup
import os


#第一步：获取网页的全部内容
#第二步：提取想要的内容（分析网页代码，找出想提取网页规则）使用beautifulsoup
#第三步：将下载的图片的地址保存在文件夹中
#第四步：解决反爬虫问题
#浏览器请求头
headers = {'User-Agent':"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.76 Mobile Safari/537.36"}
#开始的URL地址ַ
all_url = 'http://www.mzitu.com/all'
#用requests中的get方法来获取all_url的内容
start_html=requests.get(all_url,headers=headers)
#print(start_html.text)#打印网页的的内容施工text,下载图片、视频、音频、等多媒体内容是使用concent
#使用BeautifulSoup来解析我们获取到的网页（‘lxml’是指定的解析器
soup=BeautifulSoup(start_html.text,'lxml')
#使用BeautifulSoup解析网页后就可以找表现(先查找 class为 all的div标签，然后查找所有的<a>标签。)
all_a = soup.find('div', class_='all').find_all('a')
#提取<a>标签的href属性和文本
for a in all_a:
    title=a.get_text()#取出a标签的文本
    href=a['href']#取出a标签的href属性
    html=requests(href,headers-headers)#从当前的href继续进行爬取
    html_soup=BeautifulSoup(html.text,'lxml')
    max_span=html_soup.find_all('span')[10].get_text()#查找所有的<span>标签获取第十个的<span>标签中的文本也就是最后一个页面了。
    for page in range(1,int(max_span)+1):
        page_url=href+'/'+str(page)
        img_html=requests(page_url,headers=headers)
        img_soup=BeautifulSoup(img_html.text,'lxml')
        img_url = img_soup.find('div', class_='main-image').find('img')['src']
        name=img_url[-9:-4]#取URL，倒数第四位到倒数第九位做图片的名字
        img=requests.get(img_url,headers=headers)
        f=open(name +'.jpg','ab')#写入多媒体文件必须要b这个参数
        f.write(img.content)
        f.close()
      


#可以将代码封装在函数中
#函数：获取网页内容
def request(url):
    headers = {'User-Agent':"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.76 Mobile Safari/537.36"}
    content=requests.get(url,headers=headers)
    return content
#函数：创建文件
def mkdir(self,path):
    path=path.strip()
    isExists=os.path.exists(os.path.join("D:\mzitu",path))
    if not isExists:
        print(u'建了一个名字叫做',path,u'的文件夹')
        os.makedirs((os.path.join("D:\mzitu",path)))
        return True
    else:
        print(u'名字叫做',path,u'文件夹已经存在了！')
        return False
  
   


