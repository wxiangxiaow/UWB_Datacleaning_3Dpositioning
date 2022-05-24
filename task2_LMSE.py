import numpy as np

def preprocess():
    txt_path = './mean_data.txt'
    lines = []
    with open(txt_path, 'r') as file:
            for index in file.readlines():
                index = index.strip('\n').strip('[').strip(']').split(',')[1:]
                for i in range(len(index)):
                    index[i] = float(index[i])
                lines.append(index)
    file.close()
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

def LMLE(basic, distance, n):

    #观测矩阵a
    a = []
    #观测结果b
    b = []
    #求解坐标, xd,yd,hd

    #下面根据P(basic),L(distance)生成观测矩阵H
    for i in range(int(n-1)):


        a0 = 2*(basic[i][0]-basic[3][0])
        a1 = 2*(basic[i][1]-basic[3][1])
        a2 = 2*(basic[i][2]-basic[3][2])
        a_ = [a0, a1, a2]
        b_= basic[i][0]**2 - basic[3][0]**2 +basic[i][1]**2 - basic[3][1]**2 + basic[i][2]**2 - basic[3][2]**2 \
                + float(distance[3]**2) - float(distance[i]**2)
        
        a.append(a_)
        b.append([b_])
    
    a = np.array(a)
    b = np.array(b)

    #即((AT*A)-1)*(AT)*b
    aT = np.transpose(a)
    x1 = np.linalg.inv(np.dot(aT, a))
    x = np.dot(np.dot(x1, aT), b)/10
    x_ = []
    for i in range(3):
        x_.append(float(x[i]))
    print(x_)
    return x_



def main():
    #基站数量n=4
    n = 4
    #初始化已知的几个观测位置坐标，x1,y1,h1,x2,y2,h2....
    P = [
        [0., 0., 1300.],
        [5000., 0., 1700.],
        [0., 5000., 1700.],
        [5000., 5000., 1300.]
        ]

    #获得观测的距离，L1-L4
    x_pre = []
    lines = preprocess()
    for L in lines:
        x_pre.append(LMLE(P, L, n))

    #计算精度
    x_true = preprocess_tags()
    #print(len(x_true))
    x_ = 0
    y_ = 0
    z_ = 0
    for i in range(len(x_true)):
        x_ += abs(float(x_pre[i][0])-float(x_true[i][0]))/float(x_true[i][0])
        y_ += abs(x_pre[i][1]-x_true[i][1])/x_true[i][1]
        z_ += abs(x_pre[i][2]-x_true[i][2])/x_true[i][2]
        n1 = np.array(x_true[i])
        n2 = np.array(x_pre[i])

    print('dx label = '+ str(1-x_/len(x_true)))
    print('dy label = '+ str(1-y_/len(x_true)))
    print('dz label = '+ str(1-z_/len(x_true)))



    

if __name__ == '__main__':
    main()