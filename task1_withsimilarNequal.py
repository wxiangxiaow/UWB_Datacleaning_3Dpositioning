import pandas as pd
import numpy as np
import os


correct_dir = './正常数据/'
wrong_dir = './异常数据/'
#len = len(os.listdir(correct_dir))
#print(str(len))


corr_len = len(os.listdir(correct_dir))
wro_len = len(os.listdir(wrong_dir))
dir_list = []
dir_list.append([correct_dir, corr_len, '正常'])
dir_list.append([wrong_dir, wro_len, '异常'])


def test(dir_info):
    txt_dir = dir_info[0]
    txt_len = dir_info[1]
    txt_name = dir_info[2]
    txt_savedir = './save_txt_sNe/'+ txt_name +'/'
    if not os.path.exists(txt_savedir):
        os.makedirs(txt_savedir)
    '''
    error1：缺少基站数据
    error2：校准数据与测量数据不同
    '''
    error1 = []
    error2 = []

    for i in range(txt_len):
        txt_path = txt_dir  + str(i+1) + '.'+ txt_name + '.txt'
        #txt_path = txt_dir  + str(1) + '.'+ txt_name + '.txt'
        txt_save = txt_savedir + str(i+1) + '.txt'
        lines = []
        with open(txt_path, 'r') as file:
            k = 0
            for index in file.readlines():
                if k == 0:
                    pass
                else:
                    #print(index)
                    index = index.strip('\n').split(':')[4:7]
                    #print(index)
                    for indes in range(len(index)):
                        index[indes] = int(index[indes])
                    #print(index)
                    lines.append(index)
                k = k+1
        file.close()

 
        #print(lines)
        after_line = []
        k = 0
        point = []
        #print(lines)
        for i in range(len(lines)):  
            if int(lines[i][0]) == int(k):
                point.append(lines[i][1])
                k = k + 1
                #print(str(k))
                if k == 4:
                    k = 0
                    after_line.append(point)
                    #print(point)
                    point = []
        
        #print(after_line)

        a0 = []
        a1 = []
        a2 = []
        a3 = []
        for index in after_line:
            #print(index)
            a0.append(float(index[0]))
            a1.append(float(index[1]))
            a2.append(float(index[2]))
            a3.append(float(index[3]))
        a0_mean = np.median(np.array(a0))
        a1_mean = np.median(np.array(a1))
        a2_mean = np.median(np.array(a2))
        a3_mean = np.median(np.array(a3))

        #去除异常数据1（平均数±500）
        #异常范围（可调）
        clear_1 = []
        for j in range(len(a0)):
            da0 = min(abs(a0_mean-a0[j])-1000, 0.)
            da1 = min(abs(a1_mean-a1[j])-1000, 0.)
            da2 = min(abs(a2_mean-a2[j])-1000, 0.)
            da3 = min(abs(a3_mean-a3[j])-1000, 0.)
               
            if da0*da1*da2*da3 == 0.:
                error1.append(lines[j])
                print('A WRONG DATA')
                print(after_line[j])
            else:
                clear_1.append(after_line[j])
        #print(clear_1)



        
        #数error并合并数据
        clear_ = []
        for i in clear_1:
            if len(i) < 9:
                error1.append(i)
            else:
                ret1 = float(i[1])-float(i[2])
                ret2 = float(i[3])-float(i[4])
                ret3 = float(i[5])-float(i[6])
                ret4 = float(i[7])-float(i[8])
                if ret1+ret2+ret3+ret4 != 0:
                    error2.append(i)

        k = 0
        clear_ = []
        for i in clear_1:
            k = k+1
            j = []
            #print(i)
            j.append(int(k))
            j.append(int(i[0]))
            j.append(int(i[1]))
            j.append(int(i[2]))
            j.append(int(i[3]))
            clear_.append(j)
        if len(lines)/4 -float(len(after_line)) > 0:
            print('A DATAMISSING IS HAPPENED on '+ str(txt_path))
            print(error1)
        if len(error2) > 0:
            print('WRONG DATA on '+str(txt_path))
            print(error2)
        #print(len(after_))
        #print(error1)
        #print(error2)

        w_file = open(txt_save, 'w')
        for index in clear_:
            w_file.write(str(index))
            w_file.write('\n')
        w_file.close()

        #print(after_)



            
if __name__ == "__main__":
    for i in range(2):
        test(dir_list[i])

