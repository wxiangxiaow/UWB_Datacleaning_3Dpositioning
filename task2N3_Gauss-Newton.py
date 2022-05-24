import math
import os
import numpy as np


def Jo(x, y, z):
    jocker = []
    x1 = [2*x, 2*y, 2*(z-1300)]
    x2 = [2*(x-5000), 2*y, 2*(z-1700)]
    x3 = [2*x, 2*(y-5000), 2*(z-1700)]
    x4 = [2*(x-5000), 2*(y-5000), 2*(z-1300)]
    jocker.append(x1)
    jocker.append(x2)
    jocker.append(x3)
    jocker.append(x4)
    jocker = np.array(jocker)

    return jocker

def predict(lines):
    
    #init x, y, z
    x = 0.
    y = 0.
    z = 0.

    #input distance
    d1 = lines[0]
    d2 = lines[1]
    d3 = lines[2]
    d4 = lines[3]

    for i in range(2000):
        e04 = x**2 + y**2 + (z-1300)**2 - d1**2
        e14 = (x-5000)**2 + y**2 + (z-1700)**2 - d2**2
        e24 = x**2 + (y-5000)**2 +(z-1700)**2 -d3**2
        e34 = (x-5000)**2 + (y-5000)**2 + (z-1300)**2 - d4**2
        e = np.array([[e04], [e14], [e24], [e34]])

        jo = Jo(x, y, z)
        jT = np.transpose(jo)
        h_1 = np.linalg.inv(np.dot(jT, jo))
        h_ = np.dot(np.dot(h_1, jT), e)
        
        #print(type(h_[0]))

        x = x - float(h_[0])
        y = y - float(h_[1])
        z = z - float(h_[2])

        #print(h_)
    return [x/10, y/10, z/10]


def preprocess():
    txt_path = './data/distance15_median/median_data1.txt'
    lines = []
    with open(txt_path, 'r') as file:
            for index in file.readlines():
                index = index.strip('\n').strip('[').strip(']').split(',')[1:]
                for i in range(len(index)):
                    index[i] = float(index[i])
                lines.append(index)
    file.close()
    #print(lines)
    return lines

def preprocess_tags():
    txt_path = './tags.txt'
    lines = []
    with open(txt_path, 'r', encoding='UTF-8') as file:
        for index in file.readlines():
            index = index.strip('\n').split(' ')[1:]
            line = []
            for i in range(len(index)):
                index[i] = float(index[i].strip(' '))
            lines.append(index)
    file.close()
    return lines


def main():
    x_true = preprocess_tags()
    x_pre_ = preprocess()

    x_pre = []
    for index in x_pre_:
        x_pre.append(predict(index))

    print(len(x_pre))
    print(len(x_true))

    

    w_file = open('./position_error.txt', 'w')
    for index in x_pre:
        indes = str(index[0])+' '+str(index[1]) + ' '+str(index[2])
        w_file.write(str(indes))
        w_file.write('\n')
    w_file.close()

    assert len(x_pre[0])== len(x_true[0])

    x_ = 0.
    y_ = 0.
    z_ = 0.
    for i in range(len(x_true)):
        x_ += abs(float(x_pre[i][0])-float(x_true[i][0]))/float(x_true[i][0])
        y_ += abs(x_pre[i][1]-x_true[i][1])/x_true[i][1]
        z_ += abs(x_pre[i][2]-x_true[i][2])/x_true[i][2]


    print('dx label = '+ str(1-x_/len(x_true)))
    print('dy label = '+ str(1-y_/len(x_true)))
    print('dz label = '+ str(1-z_/len(x_true)))



    
if __name__ == '__main__':
    main()