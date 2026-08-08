### **Network and System Setup Instructions**

- **Network Connection:** Upon entering, first check the network settings. You need to connect to your own WiFi, specifically **"connection 1"** (the one starting with **"90"**).

- **Configuration Update:** `nano .bashrc`

  After modifying and saving the `.bashrc` file, you **must** open a new terminal window for the changes to take effect.

  > **Tip:** Alternatively, you can run `source ~/.bashrc` to apply the changes to your current session.

- **Monitor Mode:** `sudo airmon-ng start wlan0`

  **Password:** `xxxxx`

  — This command sets the wireless interface (antenna) to **monitor mode** to allow for packet capturing.

![image-20240417153613543](https://gitee.com/xingmegn/pic_figs/raw/master/figs/202404171536622.png)

### **Execution and Environment Setup**

- **Activate Environment:** `conda activate wheeltec`

  First, activate the specific Conda environment required for the project.

- **Set Monitoring Channel:** `sudo iwconfig wlan0mon channel 3`

  Configure the interface to **Channel 3** for monitoring.

  > **Note:** This step must be performed while the Conda environment is active.

- **Source Directory:** `/wheeltec_robot/src`

  Navigate to this directory to find `caction.py` and `ccfs.py`, which are used for **receiving data**.

- **Running Scripts:** When executing the code, you **must** use `sudo python3 [filename]`.

  *Example:* `sudo python3 caction.py`

<img src="https://gitee.com/xingmegn/pic_figs/raw/master/figs/202404171549410.png" alt="image-20240417154938389" style="zoom: 150%;" />

```
其中：cdw   为抓包函数
filt 数据获取
SendDataBase  将从上步获取的数据存入数据库
```

Navigation and Coordinate Tracking
Start Navigation: First, initiate the navigation system.

Command: python receive_tf.py

Run this script to receive the TF (Transform) coordinates.

```
rospy.init_node("tf")  # 这行代码的作用是初始化一个名为 "tf" 的 ROS 节点。这个节点可以发布消息到话题，订阅话题，提供服务，或者调用其他节点提供的服务。
lestener = tf.TransformListener()  # 创建一个监听器，监听和获取这些坐标变换信息
(trans, rot) = listener.lookupTransform("map", "base_link", rospy.Time(0))  # 获取机器人在地图坐标系中的位置和方向，其中 "map" "base_link" 是两种不同的坐标系
goal_status_sub = rospy.Subscriber("/move_base/result", MoveBaseActionResult, callback, queue_size=1)  # 创建一个订阅者来订阅"/move_base/result"话题。这个话题通常用于发布机器人导航的结果。
```

### **Publishing Position Coordinates**

- **Script:** `issue_position.py`
- **Action:** **Publish Position** Run this script to broadcast or publish the current position/coordinate data to the system.



### Communication between the two boards of the car

```
两块板子连接 ：https://blog.csdn.net/Wild_Ray/article/details/115311103

 
echo 'echo 1 > /proc/sys/net/ipv4/ip_forward' | sudo tee -a /etc/rc.local
echo 'iptables -t nat -A POSTROUTING -o usb0 -j MASQUERADE' | sudo tee -a /etc/rc.local
echo 'iptables -A FORWARD -i eth0 -o usb0 -m state --state RELATED,ESTABLISHED -j ACCEPT' | sudo tee -a /etc/rc.local
echo 'iptables -A FORWARD -i usb0 -o eth0 -j ACCEPT' | sudo tee -a /etc/rc.local
sudo vim /etc/rc.local 
在第一行添加上：#!/bin/sh -e
sudo chmod +x /etc/rc.local
sudo systemctl start rc-local.service
```

### Navigation Logic

```
开启ros导航： roslaunch turn_on_wheeltec_robot navigation.launch 
地图更换在navigation.launch中
打开rviz软件：rviz   ——进行实时监控
打开tf坐标接收：cd  ……     python receive_tf.py
发布坐标：issue_position.py x y a b 其中x, y 为位置坐标；a, b 为方向控制
前 0 1
后 -1 0
左 1 1
后 1 -1
issue_position.py：巡航逻辑是按照点的循环进行
```

### Graph construction logic

```
开启ros建图：roslaunch turn_on_wheeltec_robot mapping.launch 
打开rviz软件
打开键盘控制：roslaunch wheeltec_robot_rc keyboard_teleop.launch
b键可以切换运动逻辑
保存：
打开地图路径：
cd /home/wheeltec/wheeltec_robot/src/turn_on_wheeltec_robot/map
手动保存地图：
rosrun map_server map_saver -f 20220615
```

### Packet receiving logic

```
开启wlan1监听模式： sudo airmon-ng start wlan1
				 sudo iwconfig wlan1mon channel 3
				 sudo python3 M_main.py
```



### Car movement calibration

```
实际前进1m，里程计x是否是+1m，误差±0.02m正常
实际左移1m，里程计y是否是+1m，误差±0.02m正常
实际逆时针旋转1圈，里程计z是否是+6.28rad，误差±0.2rad正常

里程计查看命令：rostopic echo /odom
话题里面的position信息就是里程计

假设实际前进1m，里程计显示x是1.2，那么x_scale参数应该从默认的1改为1/1.2=0.833，y轴、z轴同理
下面的positive是指左转时候的误差，negative是指右转的时候的误差
```

![6978AE04F73743DED4A2A0F117C17B62](https://gitee.com/xingmegn/pic_figs/raw/master/figs/202405281208439.png)

