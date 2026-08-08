from ccfs import *
import threading as th
import time

while 1:
    #a = input("input which channel you want:")
    #res = os.system("iwconfig wlan1mon channel "+str(a))
    
    #print("ACTION")
    th1 = th.Thread(target=cdw)
    th2 = th.Thread(target=filt)
    th3 = th.Thread(target=SendDataBase)
    
    th1.start()
    time.sleep(0.1)
    th2.start()
    th3.start()

    th1.join()
    th2.join()
    th3.join()

