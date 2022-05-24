# UWB_Datacleaning_3Dpositioning
A project for [2021 Mathematical modeling competition](https://cpipc.acge.org.cn//cw/detail/4/2c9080147c73b890017c7779e57e07d2) Problem E.

![example](https://github.com/wxiangxiaow/UWB_Datacleaning_3Dpositioning/blob/main/images/example.png)

### Data Format
For raw UWB data, the correct data format should be:

**T:144235622:RR:0:0:950:950:118:1910**

**T:144235622:RR:0:1:2630:2630:118:1910**

**T:144235622:RR:0:2:5120:5120:118:1910**

**T:144235622:RR:0:3:5770:5770:118:1910**

Which means

Tag : Time stamp : Range report : Tag ID : Anchor ID :  Anchor ranging value(mm) : Check value of ranging value : Serial number : Object number

### Task 1: Data Cleaning

For data cleaning, there are three options:
1) Only abnormal data (one or more bits are lost) are removed, and the same and similar data are retained. ->task1_NoSimilarNequal.py
2) All of abnormal data , similar data and same data are removed (when euclidean distance <= 15). ->task1_withsimilarNequal.py
3) All of abnormal data , similar data and same data are removed (when euclidean distance <= 18). ->task1_withsimilarNequal.py

### Task 2&3: 3D positioning and model applys on other environment
1) LMS error estimation based on ensemble location model. ->task2_LMSE.py
2) Gauss-Newton method for optimal positioning of three-dimensional position estimation model with residual optimization based on TOF ranging, including measurement accuracy calculation and trajectory prediction. ->task2N3_Gauss-Newton.py

![3d](https://github.com/wxiangxiaow/UWB_Datacleaning_3Dpositioning/blob/main/images/3d.png)

### Task 4: Distinguish interference data
Code reference from [THIS](https://github.com/TerenceChen95/pneumonia-detection-pytorch)

Use A simple (with only Linear, ReLU and dropout layers) Pytorch model to distinguish data. ->task4_train.py

![net](https://github.com/wxiangxiaow/UWB_Datacleaning_3Dpositioning/blob/main/images/net.png)



![a](https://github.com/wxiangxiaow/UWB_Datacleaning_3Dpositioning/blob/main/images/Adagrad+15.png)

A test file is given. ->task4_predict.py
### Requirments
numpy

matplotlib

pandas

torch


### Reference
[1]张旭. 基于超宽带雷达的机器人室内定位方法研究[D].北京建筑大学,2019.
