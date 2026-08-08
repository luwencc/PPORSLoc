#!/usr/bin/env python
# encoding: utf-8
from tracemalloc import start
from queue import Queue
from scapy.all import *
from pprint import pprint
import pymysql
import datetime
import time
import threading

data1_lock = threading.Lock()
data = Queue()
data1  = list()
i = 1
filter_rule = r"ether src host 50:fa:84:cd:cf:57 or ether src host 50:fa:84:cb:ec:1d or ether src host 50:fa:84:c8:81:45 or ether src host 50:fa:84:cb:e4:df or ether src host 50:fa:84:d0:ba:02 or ether src host 50:fa:84:c9:f4:e7 or ether src host 50:fa:84:cb:e3:73"

def cdw():
    dpkt = sniff(iface ='wlan1mon', filter=filter_rule,  prn=PackCallBack, store=0)  # 抓包 # prn=lambda x: x.show(),


def PackCallBack(dpkt):
    data.put(dpkt)
    

def filt():
    global i
    while 1:
        ii = data.get()
        mtt = TimeStamp2Time(ii.time) #get timei

        res1 = str(ii.ChannelFrequency)

        res2 = str(ii.dBm_AntSignal)

        res3 = str(ii.addr2)

        res4 = str(ii.addr1)

        with data1_lock:
            data1.append((str(i),mtt,res1,res2,res3,res4))
            i = i+1


def TimeStamp2Time(timeStamp):
    fff = str(timeStamp)[11:21]
    timeTmp = time.localtime(timeStamp)  #time.localtime()格式化时间戳为本地时间
    myTime = time.strftime("%Y-%m-%d %H:%M:%S", timeTmp)  #将本地时间格式化为字符串
    mt = str(myTime)+'.'+fff
    return mt


def SendDataBase():
    which_times = input("which times to collect data:    ")
    connent = pymysql.connect(host='127.0.0.1', user='agxorin', password='admin2023', db='test', port=3306, charset='utf8')

    cursor = connent.cursor()

    cursor.execute("SHOW TABLES LIKE '{}rece_rss'".format(which_times))
    if not cursor.fetchone():
        sql_build = """CREATE TABLE {}rece_rss(id INT(11), time DATETIME(3), channel VARCHAR(25), rss VARCHAR(25), SA VARCHAR(25), DA VARCHAR(25))""".format(which_times)
        cursor.execute(sql_build)
        connent.commit()

    sql="insert into {}rece_rss(id,time,channel,rss,SA,DA) values(%s,%s,%s,%s,%s,%s)".format(which_times)

    sleep_event = threading.Event()
    try:
        while 1:
    
            if data1:

                with data1_lock:
                    cursor.executemany(sql,data1)

                    connent.commit()

                    print(data1[-1]) #用于测试时间匹配

                    #print('\n', data.qsize(),'\n')

                    data1.clear()

            sleep_event.wait(0.5)

    except KeyboardInterrupt:
        print ("interrupt by user!")

    finally:
        print("close mysql")
        connent.close()
