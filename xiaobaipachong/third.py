# coding:utf8
#山东大学实现无线网络掉线自动重连（其他学校类似）
import urllib.request
import urllib.parse
import socket
import types
import time
import re
import os
import subprocess

class Login:
    #初始化
    def __init__(self):
        #学号密码
        self.username = '201200131012'
        self.password = 'XXXXXX'
        #山大无线STU的IP网段
        self.ip_pre = '211.87'
        #登录时长
        self.overtime = 720
        #检测间隔时间，单位为秒
        self.every = 10

#模拟登陆
    def login(self):
        print (self.getCurrentTime(), u"正在尝试认证QLSC_STU无线网络")
        ip = self.getIP()
        data = {
            "username": self.username,
            "password": self.password,
            "serverType": "",
            "isSavePass": "on",
            "Submit1": "",
            "Language": "Chinese",
            "ClientIP": self.getIP(),
            "timeoutvalue": 45,
            "heartbeat": 240,
            "fastwebornot": False,
            "StartTime": self.getNowTime(),
            #持续时间，超过这个时间自动掉线，可进行设置
            "shkOvertime": self.overtime,
            "strOSName": "",
            "iAdptIndex": "",
            "strAdptName": "",
            "strAdptStdName": "",
            "strFileEncoding": "",
            "PhysAddr": "",
            "bDHCPEnabled": "",
            "strIPAddrArray": "",
            "strMaskArray": "",
            "strMask": "",
            "iDHCPDelayTime": "",
            "iDHCPTryTimes": "",
            "strOldPrivateIP": self.getIP(),
            "strOldPublicIP": self.getIP(),
            "strPrivateIP": self.getIP(),
            "PublicIP": self.getIP(),
            "iIPCONFIG":0,
            "sHttpPrefix": "http://192.168.8.10",
            "title": "CAMS Portal"
        }
        #消息头
        headers = {
            'User-Agent' : 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36',
            'Host': '192.168.8.10',
            'Origin': 'http://192.168.8.10',
            'Referer': 'http://192.168.8.10/portal/index_default.jsp?Language=Chinese'
        }
        post_data = urllib.parse.urlencode(data)
        login_url = "http://192.168.8.10/portal/login.jsp?Flag=0"
        request = urllib.request.Request(login_url, post_data, headers)
        response = urllib.request.urlopen(request)
        result = response.read().decode('gbk')
        
#打印登录结果
    def getLoginResult(self, result):
        if u"用户上线成功" in result:
            print (self.getCurrentTime(),u"用户上线成功,在线时长为",self.overtime/60,"分钟")
        elif u"您已经建立了连接" in result:
            print (self.getCurrentTime(),u"您已经建立了连接,无需重复登陆")
        elif u"用户不存在" in result:
            print (self.getCurrentTime(),u"用户不存在，请检查学号是否正确")
        elif u"用户密码错误" in result:
            pattern = re.compile('<td class="tWhite">.*?2553:(.*?)</b>.*?</td>', re.S)
            res = re.search(pattern, result)
            if res:
                print (self.getCurrentTime(),res.group(1),u"请重新修改密码")
        else:
            print (self.getCurrentTime(),u"未知错误，请检查学号密码是否正确")


#获取当前时间戳，13位
    def getNowTime(self):
        return str(int(time.time()))+"000"


#获取本机无线IP
    def getIP(self):
        local_iP = socket.gethostbyname(socket.gethostname())
        if self.ip_pre in str(local_iP):
            return str(local_iP)
        ip_lists = socket.gethostbyname_ex(socket.gethostname())
        
        for ip_list in ip_lists:
            if isinstance(ip_list, list):
                for i in ip_list:
                    if self.ip_pre in str(i):
                        return str(i)
            elif type(ip_list) is bytes:
                if self.ip_pre in ip_list:
                    return ip_list

#判断当前是否可以联网
    def canConnect(self):
        #：devnull是系统的黑洞，如果ping成功，系统写进去的内容全部被吃掉，result应该为空。如果ping失败了，result就不为空了。所以逻辑是这样的
        fnull = open(os.devnull, 'w')
        result = subprocess.call('ping www.baidu.com', shell = True, stdout = fnull, stderr = fnull)
        fnull.close()
        if result:
            return False
        else:
            return True

#获取当前时间
    def getCurrentTime(self):
        return time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
 
    #主函数
    def main(self):
        print (self.getCurrentTime(), u"您好，欢迎使用模拟登陆系统")
        while True:
            nowIP = self.getIP()
            if not nowIP:
                print (self.getCurrentTime(), u"请检查是否正常连接QLSC_STU无线网络")
            else:
                print (self.getCurrentTime(),u"成功连接了QLSC_STU网络,本机IP为",nowIP)
                self.login()
                while True:
                    can_connect = self.canConnect()
                    if not can_connect:
                        nowIP = self.getIP()
                        if not nowIP:
                            print (self.getCurrentTime(), u"当前已经掉线，请确保连接上了QLSC_STU网络")
                        else:
                            print (self.getCurrentTime(), u"当前已经掉线，正在尝试重新连接")
                            self.login()
                    else:
                        print (self.getCurrentTime(), u"当前网络连接正常")
                    time.sleep(self.every)
            time.sleep(self.every)
 
login = Login()
login.main()