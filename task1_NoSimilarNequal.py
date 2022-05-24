import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math


correct_dir = './正常数据/'
wrong_dir = './异常数据/'
#len = len(os.listdir(correct_dir))
#print(str(len))


corr_len = len(os.listdir(correct_dir))
wro_len = len(os.listdir(wrong_dir))
dir_list = []
dir_list.append([correct_dir, corr_len, 'normal', '正常'])
dir_list.append([wrong_dir, wro_len, 'error', '异常'])


def test(dir_info):
    txt_dir = dir_info[0]
    txt_len = dir_info[1]
    txt_name = dir_info[2]
    txt_name_ = dir_info[3]
    txt_savedir = './save_txt/'+ txt_name_ +'/'
    txt_savedir_median = './save_txt_mean/'+ txt_name +'/'

    if not os.path.exists(txt_savedir_median):
        os.makedirs(txt_savedir_median)
    '''
    error1：异常数值
    '''
    error1 = []
    error2 = []

    _clear = []
    for i in range(txt_len):
        txt_save_media = txt_savedir_median  + str(i+1)+'.txt'
        txt_path = txt_savedir + str(i+1) + '.txt'
        lines = []
        with open(txt_path, 'r') as file:
            for index in file.readlines():
                index = index.strip('\n').strip('[').strip(']').split(',')
                for j in range(len(index)):
                    index[j] = int(index[j].strip(' '))
                lines.append(index)
        file.close()


        
        ret1 = False
        ret2 = False
        a0_median = 0


        if not ret1:
            a0 = []
            a1 = []
            a2 = []
            a3 = []
            for index in lines:
                a0.append(float(index[1]))
                a1.append(float(index[2]))
                a2.append(float(index[3]))
                a3.append(float(index[4]))
            a0_mean = np.mean(np.array(a0))
            a1_mean = np.mean(np.array(a1))
            a2_mean = np.mean(np.array(a2))
            a3_mean = np.mean(np.array(a3))

            #去除异常数据1（平均数±500）
            #异常范围（可调）
            clear_1 = []
            for j in range(len(a0)):
                da0 = min(abs(a0_mean-a0[j])-500, 0.)
                da1 = min(abs(a1_mean-a1[j])-500, 0.)
                da2 = min(abs(a2_mean-a2[j])-500, 0.)
                da3 = min(abs(a3_mean-a3[j])-500, 0.)
               
                if da0*da1*da2*da3 == 0:
                    error1.append(lines[j])
                    #print('A WRONG DATA')
                    #print(lines[j])
                else:
                    clear_1.append(lines[j])

            #去除相似
            #定义相似：两个向量的欧氏距离小于20
            clear_2 = []
            l1 = len(clear_1)

            for j in range(l1):
                if j == 0:
                    clear_2.append(clear_1[0][1:])
                else:
                    last = clear_1[j][1:]
                    distance = 0
                    l2 = len(clear_2)
                    ret = True
                    for m in range(l2):
                        next = clear_2[m]
                        distance = math.sqrt((int(last[0])-int(next[0]))**2 + (int(last[1])-int(next[1]))**2 + \
                             (int(last[2])-int(next[2]))**2 + (int(last[3])-int(next[3]))**2)
                        if distance < 18:
                            ret = False
                            break
                    if ret:
                        clear_2.append(last)
        

            #做每个txt的均值用
            a0_ = []
            a1_ = []
            a2_ = []
            a3_ = []
            k = 0
            clear_3 = []
            for index in clear_2:
                k = k+1
                a0_.append(float(index[0]))
                a1_.append(float(index[1]))
                a2_.append(float(index[2]))
                a3_.append(float(index[3]))
                indes = []
                indes.append(k)
                indes.append(int(index[0]))
                indes.append(int(index[1]))
                indes.append(int(index[2]))
                indes.append(int(index[3]))
                clear_3.append(indes)
            a0_mean_ = np.mean(np.array(a0))
            a1_mean_ = np.mean(np.array(a1))
            a2_mean_ = np.mean(np.array(a2))
            a3_mean_ = np.mean(np.array(a3))
            mm = []
            mm.append(int(i+1))
            mm.append(float(a0_mean_))
            mm.append(float(a1_mean_))
            mm.append(float(a2_mean_))
            mm.append(float(a3_mean_))

            _clear.append(mm)





            #洗掉圈外的
            _file = open(txt_save_media, 'w')
            for index in clear_3:
                _file.write(str(index))
                _file.write('\n')
            _file.close()

            #print('i='+str(i))

        
    
    return _clear




            
if __name__ == "__main__":
    median_list = []
    for i in range(2):
        median_ = test(dir_list[i])
        for j in range(len(median_)):
            median_list.append(median_[j])
        print(str(len(median_)))
    #print(str(len(median_list)))
    w_file = open('./mean_data.txt', 'w')
    for index in median_list:
        w_file.write(str(index))
        w_file.write('\n')
    w_file.close()
