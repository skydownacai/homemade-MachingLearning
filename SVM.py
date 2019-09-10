import numpy as np
import numpy.matlib
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation
def creat_data():
    data_x= []
    data_y= []
    data = {"x":[],'y':[],'label':[]}
    for i in range(100):
        x1 = random.uniform(-10,10)
        x2 = random.uniform(-10,10)
        if x2 >= -2*x1-1:
            y1 = 1
        elif x2 <= -2*x1 - 1.1:
            y1 = -1
        else:
            continue
        data['x'].append(x1)
        data['y'].append(x2)
        data['label'].append(y1)
        data_x.append([x1,x2])
        data_y.append([y1])
    print(data)
    save = pd.DataFrame(data=data)
    save.to_csv('data.csv')
    return (data_x,data_y)
class SVM:
    def __init__(self):
        self.a = []
        #lagrange multipliers
        self.X = []
        self.Y = []
        self.weight = []
        self.e = []
        #the vector whose entries are all 1
        self.m = 0
        #total amount of samples
        self.c = 0
        #penality weight
        self.K = 0
        #the kernel matrix
        self.b = 1
        #the threhold
        self.kernel_mode  = ''
        #the choice of kernel
        self.relax = []
        #the relaxition variables
        self.select= 1
        self.ban = []
        # if value = 1: choose the variable in all sampleset
        # if value = 2: choose the variable in inboundery
        self.w = []
    def kernel(self,x,y):
        '核函数'
        if self.kernel_mode == 'linear':
            return np.dot(x,y)
    def K_martix(self):
        '创建核矩阵'
        self.K = np.array([[0 for i in range(self.m*self.m)]],dtype = np.float64).reshape(self.m,self.m)
        for i in range(self.m):
            for j in range(self.m):
                self.K[i][j] = self.kernel(self.X[i],self.X[j])
    def delta(self,i,j):
        '以i,j进行优化'
        if self.Y[i] == self.Y[j]:
            L = max(0,self.a[i] + self.a[j] - self.c)
            H = min(self.c,self.a[i] + self.a[j])
        else:
            L = max(0,self.a[i] - self.a[j])
            H = min(self.c,self.c + self.a[i] -self.a[j])
            '''确定约束范围，为了使得更新后 0<= ai <=c '''
            #print(i,j)
            #print(i,j)
            #print()
        new = self.a[i][0] + self.Y[i][0]*(self.E(j)-self.E(i))/(self.K[i][i]+self.K[j][j] - 2*self.K[i][j])
        if new < L:
            new = L
        if new > H:
            new = H
            '更新ai'
        return abs(new - self.a[i])
    def SMO(self):
            i,j = self.choose()
            #print('选择{},{}'.format(i,j))
            ai_old = self.a[i][0]
            '如果异号  L = max (0,a1-a2)   H = min(c,c+a1-a2)'
            '如果同号  L = max (0,a1+a2-c) H = min(c,a1+a2)'
            if self.Y[i] == self.Y[j]:
                L = max(0,self.a[i] + self.a[j] - self.c)
                H = min(self.c,self.a[i] + self.a[j])
            else:
                L = max(0,self.a[i] - self.a[j])
                H = min(self.c,self.c + self.a[i] -self.a[j])
            '''确定约束范围，为了使得更新后 0<= ai <=c '''
            #print()
            new = self.a[i][0] + self.Y[i][0]*(self.E(j)-self.E(i))/(self.K[i][i]+self.K[j][j] - 2*self.K[i][j])
            self.a[i] = new
            if self.a[i][0] < L:
                self.a[i] = L
            if self.a[i][0] > H:
                self.a[i] = H
            '更新ai'
            delta_ai = self.a[i][0] - ai_old
            #print('new ai :',self.a[i])
            #print('old ai :',ai_old)
            #print('delat ai:',delta_ai)
            #print(self.a)
            self.a[j] = self.a[j] - self.Y[i][0]*self.Y[j][0]*delta_ai
            '更新aj'
            self.update_bias()
            '更新b'
    def herustic_a(self):
        '硬间隔下的启发式a'
        self.a = numpy.array([0 for i in range(self.m)],dtype = np.float64).reshape(self.m,1)
        #print(np.dot(self.a.T,self.Y))
    def init_SVM(self,X,Y,c,kernel = 'linear'):
        self.X = numpy.array(X)
        self.Y = numpy.array(Y)
        self.m = len(self.Y)
        self.kernel_mode = kernel
        self.c    = c
        self.e = np.array([1]*self.m).reshape(-1,1)
        self.b = 1
        #creat the e-vector
        self.K_martix()
        #creat the kernel matrix
        self.herustic_a()
        #启发式一个拉格朗日乘子向量
    def KKT_verify(self):
        '检验是否满足kkt条件'
        for i in range(self.m):
            if self.a[i] == 0 and self.Y[i][0]*self.g(self.X[i]) < 1:
                return False
            if self.a[i] > 0 and self.a[i] < self.c and self.Y[i][0]*self.g(self.X[i]) != 1:
                return False
            if self.a[i] == self.c and self.Y[i][0]* self.g(self.X[i]) >1:
                return False
            if self.a[i] > self.c:
                return False
        return True
    def get_accurance(self):
        '计算分类正确率'
        acc = 0
        acc1= 0
        for i in range(self.m):
            if self.Y[i][0] == self.sign(i):
                acc += 1
            if self.Y[i][0]*self.g(self.X[i]) >= 1:
                acc1+= 1
            else:
               print(self.X[i],self.Y[i])
        print('软间隔正确率:{}%'.format(acc*100/self.m))
        print('硬间隔正确率:{}%'.format(acc1*100/self.m))
    def fit(self,X,Y,c,kernel,epoch):
        self.init_SVM(X,Y,c,kernel)
        self.check_strain()
        #初始化一些参数和数值
        #print(self.Y )
        #print(self.a)
        #print(self.e)
        #print(self.K)
        #print(self.m)
        count = 1
        last_dual_value = -1000
        #self.check()
        while count <= epoch:
            self.SMO()
            print("{}次训练完成".format(count))
            self.get_accurance()
            print('对偶函数值{}'.format(self.dual_value()))
            self.check_strain()
            count += 1
            self.update_weight()
            print()
        print('训练完成')
        self.plot()
    def predict(self,X,Y,c):
        '''
           X,Y 是数据集
           mode = 1 硬间隔
           mode = 2 软间隔
        '''
        self.Y = Y
        self.X = X
        '初始化参数'
        self.init_SVM(X,Y,c)
        while self.KKT_verify() == False:
              print(self.dual_value())
              '如果没有到最优条件,也就是说如果不满足kkt,那么继续SMO'
              self.SMO()
    def dual_value(self):
        '计算dual 函数的值 :objective is maximium'
        z = self.a*self.Y
        return  (np.dot(self.a.T,self.e) - (1/2)*np.dot(np.dot(z.T,self.K),z))[0][0]
    def g(self,x):
        'g=w.T*x + b'
        ':return float'
        z = self.a*self.Y
        k = numpy.array([self.kernel(x,self.X[i]) for i in range(self.m)])
        return np.dot(z.T,k)[0] + self.b
    def update_bias(self):
        '计算bias :b; 在软间隔线性支持向量机中,最优的w只有一个,但是最优的b可以有多个.考虑到鲁棒性，这时选择多个支持向量的b,求均值'
        sum1 = []
        for i in range(self.m):
            if  self.a[i] > 0 and self.a[i] < self.c:
                '为了使得满足kkt 条件,那么若0< ai < c 必有'
                sum1.append(self.Y[i][0] - self.g(self.X[i]) + self.b)
        self.b = sum(sum1)/len(sum1)
    def choose(self):
        invaild_1 = -1
        invaild_2 = -1
        max_E     = 0
        for i in range(self.m):
            if self.a[i] > 0 and self.a[i] < self.c and 1 - self.Y[i][0]*self.g(self.X[i]) != 0:
               for j in range(self.m):
                   if j != i :
                       d = self.delta(i,j)
                       if  d > max_E:
                          max_E = d
                          invaild_1 = i
                          invaild_2 = j
        #首先找支持向量中违反kkt
        if invaild_1 != -1:
            return (invaild_1,invaild_2)
        for i in range(self.m):
            if self.a[i] == 0 and 1 - self.Y[i][0]*self.g(self.X[i]) > 0:
               for j in range(self.m):
                   if j != i:
                       d = self.delta(i,j)
                       if  d > max_E:
                          max_E = d
                          invaild_1 = i
                          invaild_2 = j
        if invaild_1 != -1:
           return (invaild_1,invaild_2)
        #在ai = 0 中 找违反kkt 条件的
        for i in range(self.m):
            if self.a[i] == self.c and 1 - self.Y[i][0]*self.g(self.X[i]) < 0:
               for j in range(self.m):
                   if j != i:
                       d = self.delta(i,j)
                       if  d > max_E:
                          max_E = d
                          invaild_1 = i
                          invaild_2 = j
        return (invaild_1,invaild_2)
    def sign(self,i):
        '返回第i个样本的符号,SVM结果的符号'
        item = self.g(self.X[i])
        if item > 0 :
            return 1
        if item < 0 :
            return -1
        if item == 0 :
            return 0
    def E(self,i):
        '''第i个样本的误差'''
        ':return float'
        return self.g(self.X[i]) - self.Y[i][0]
    def outcome(self):
        '计算分类结果'
    def check_strain(self):
        print("constrain1 : sigma(ai*yi) = {}".format(numpy.dot(self.a.T,self.Y)[0]))
    def check(self):
        self.a[-1][0] = 0.9
        print(self.a)
    def update_weight(self):
        if self.kernel_mode != 'linear':
            print('求权错误')
        else:
            z = self.a * self.Y
            w = np.dot(self.X.T,z)
            self.w = w
            print("法向量为:",w)
    def plot(self):
        for i in range(self.m):
            if self.Y[i] == 1:
                plt.scatter(self.X[i][0],self.X[i][1],marker='o',c='red')
            else:
                plt.scatter(self.X[i][0],self.X[i][1],marker='x',c='black')
        x = np.array(np.linspace(-10,10,200))
        y = (x*self.w[0]*-1 - self.b)/self.w[1]
        y1= (x*self.w[0]*-1 - self.b+1)/self.w[1]
        y2= (x*self.w[0]*-1 - self.b+-1)/self.w[1]
        y3= (-2*x-1.25)
       # plt.plot(x,y)
       # plt.plot(x,y1)
       # plt.plot(x,y2)
      #  plt.plot(x,y3)
        plt.show()
creat_data()
data = pd.read_csv('data.csv')
X = np.array(data[['x','y']].values)
Y = np.array(data.label.values).reshape(-1,1)
#print(X,Y)
Test = SVM()
Test.fit(X,Y,c=10,kernel='linear',epoch = 1)
#print((np.array(np.linspace(-10,10,200))*2*-1+1)/2)
#print(a)
