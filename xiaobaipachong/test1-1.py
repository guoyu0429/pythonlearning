# coding:utf8
import requests
from bs4 import BeautifulSoup
import os
from test2 import request#这个函数获取网页的response 然后返回
from xiaobaipachong.test1 import page_url, img_url

class mzitu(download):
    def all_url(self,url):
       html=request.get(url,3)#调用request函数把套图地址传进去会返回给我们一个response
       all_a = BeautifulSoup(html.text, 'lxml').find('div', class_='all').find_all('a')
       for a in all_a:
           title=a.get_text()
           print(u'开始保存：',title)
           path=str(title).replace("?",'_')#我注意到有个标题带有 ？  这个符号Windows系统是不能创建文件夹的所以要替换掉
           self.mddir(path)#调用mkdir函数创建文件夹！这儿path代表的是标题title哦！！！！！不要糊涂了哦！
           os.chdir(os.path.join('D:\mzitu', path))#切换到目录
           href=a['href']
           self.html(href)#调用html函数把href参数传递过去！href是啥还记的吧？ 就是套图的地址哦！！
           
    def html(self,href):#这个函数是处理套图地址获得图片的页面地址  地址形如 http://www.mzitu.com/78538/1
        html = request.get(href,3)
        max_span = BeautifulSoup(html.text, 'lxml').find_all('span')[10].get_text()
        for page in range(1, int(max_span) + 1):
            page_url = href + '/' + str(page)
            self.img(page_url) ##调用img函数
    
    def img(self,href):#这个函数处理图片页面地址获得图片的实际地址    我们需要的地址在<div class=”main-image”>中的<img>标签的src属性中
        img_html = request.get(page_url,3)
        img_url = BeautifulSoup(img_html.text, 'lxml').find('div', class_='main-image').find('img')['src']
        self.save(img_url)
        
    def save(self,path):#这个函数保存图片
        name = img_url[-9:-4]
        print(u'开始保存：',img_url)
        img = request.get(img_url,3)
        f = open(name + '.jpg', 'ab')
        f.write(img.content)
        f.close()
        
    def mkdir(self,path):#这个函数创建文件夹
        path = path.strip()
        isExists = os.path.exists(os.path.join("D:\mzitu", path))
        if not isExists:
            print(u'建了一个名字叫做', path, u'的文件夹！')
            os.makedirs(os.path.join("D:\mzitu", path))
            return True
        else:
            print(u'名字叫做', path, u'的文件夹已经存在了！')
            return False
    
#    def request(self,url): #这个函数获取网页的response 然后返回
#       headers = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1"}
#       content = requests.get(url, headers=headers)
#       return content  
        
Mzitu = mzitu() ##实例化
Mzitu.all_url('http://www.mzitu.com/all') ##给函数all_url传入参数  你可以当作启动爬虫（就是入口）
        