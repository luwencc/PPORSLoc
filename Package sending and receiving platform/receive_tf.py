#!/usr/bin/env python
import rospy
import tf
import time
import MySQLdb
from move_base_msgs.msg import *
from issue_multi_position import positions
from datetime import datetime
def callback(data):
    global text
   # rospy.loginfo(data.status.text)
    text=data.status.text
if __name__ == '__main__':
    text=''
    j=0
    z=0
    x=0
    y=0
    c=0
    rospy.init_node('tf')
   # rospy.init_node('/move_base/result')

    i=1
    listener = tf.TransformListener()
    which_times = input("which times to collect data:   ")
    conn = MySQLdb.connect(host='192.168.1.101', user='agxorin', password='admin2023', db='test', port=3306, charset='utf8')
    cursor=conn.cursor()

    cursor.execute("SHOW TABLES LIKE '{}tf_position'".format(which_times))

    if not cursor.fetchone():
        sql_build = """CREATE TABLE {}tf_position (id INT(11),position_x VARCHAR(25), position_y VARCHAR(25),time DATETIME(3),times VARCHAR(25),number VARCHAR(25))""".format(which_times)
        cursor.execute(sql_build)
        conn.commit()

    try: 
        while not rospy.is_shutdown():
             try:
                 (trans,rot) = listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
             except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                 continue
             #t1=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
             #ct=time.time()
             #t2=(ct-int(ct))*1000
             #t="%s.%03d"%(t1,t2)
            # print(t,'position',round(trans[0],2),round(trans[1],2))
            # print('orient',rot)

             cursor.execute("SELECT NOW(3)")
             server_time_str = cursor.fetchone()[0]


             goal_status_sub=rospy.Subscriber('/move_base/result',MoveBaseActionResult,callback,queue_size=2)
             print(i, server_time_str,'position',round(trans[0],2),round(trans[1],2),z,j,x,y)
             c=abs(round(trans[0],2)-x)+abs(round(trans[1],2)-y)
             if text=='Goal reached.':
                 if c>1:
                     j=j+1
                     x=round(trans[0],2)
                     y=round(trans[1],2)
                 if j>=len(positions):
                     j=1
                     z=z+1
                 text=''
            
             #conn = MySQLdb.connect(host='192.168.0.100',user='wang',passwd='123456',db='test',port=3306)
             #conn = MySQLdb.connect(host='192.168.1.101', user='agxorin', password='admin2023', db='test', port=3306, charset='utf8')
             #cursor=conn.cursor()
                 
             sql = 'insert into {}tf_position(id,time,position_x,position_y,times,number) values (%s,%s,%s,%s,%s,%s)'.format(which_times)

             value=(i,server_time_str,round(trans[0],2),round(trans[1],2),z,j)
             i=i+1
             cursor.execute(sql,value)
             conn.commit()

             time.sleep(0.1)

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        print("close mysql")
        conn.close()
