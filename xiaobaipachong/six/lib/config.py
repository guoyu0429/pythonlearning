# coding:utf8
#配置文件

'''
URLS_FILE保存链接单的文件
OUT_FILE输出文本EXCEL路径
COUNT_TXT计数文件
DRIVER浏览器驱动
TIMEOUT采集超时时间
MAX_SCROLL_TIME下拉滚动条最大次数
NOW_URL_COUNT当前采集到第几个链接
LOGIN_URL登录淘宝的链接
SEARCH_LINK采集淘宝链接搜索页面
CONTENT采集链接临时变量
PAGE采集淘宝链接翻页数目
FILTER_SHOP是否过滤相同店铺
TOTAL_URLS_COUNT爬取链接总数‘
'''
from selenium import webdriver

URLS_FILE = 'file/urls.txt'

OUT_FILE = 'file/result.xls'

COUNT_TXT = 'file/count.txt'

DRIVER = webdriver.Chrome()

TIMEOUT = 30

MAX_SCROLL_TIME = 10

TOTAL_URLS_COUNT = 0

NOW_URL_COUNT = 0

LOGIN_URL = 'https://login.taobao.com/member/login.jhtml?spm=a21bo.50862.754894437.1.MVF6jc&f=top&redirectURL=https%3A%2F%2Fwww.taobao.com%2F'

SEARCH_LINK = 'https://www.tmall.com/?spm=a220m.1000858.a2226n0.1.kM59nz'

CONTENT = ''

PAGE = 25

FILTER_SHOP = False

ANONYMOUS_STR = '***'