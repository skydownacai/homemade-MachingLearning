import math
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import minst_read as m

false_label = 0


def x(x):
    t = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            t.append(x[i][j] / 256)
    return t


cut = 1000


def y(y):
    x = [false_label for i in range(10)]
    x[int(y)] = 1
    return x


def show(x):
    img = []
    height = int(math.sqrt(len(x)))
    for i in range(height):
        this_row = x[i * height:(i + 1) * height]
        img.append(this_row)
    plt.imshow(img, cmap='gray')
    plt.show()


def run():
    train_X = []
    train_Y = []
    Test_X = []
    Test_Y = []
    train_images = m.load_train_images()
    train_labels = m.load_train_labels()
    test_images = m.load_test_images()
    test_labels = m.load_test_labels()
    for i in range(min(len(train_images), cut)):
        train_X.append(x(train_images[i]))
        # show(x(train_images[i]))
        train_Y.append(y(train_labels[i]))
    for i in range(min(len(test_images), cut)):
        Test_X.append(x(test_images[i]))
        Test_Y.append(y(test_labels[i]))
        # plt.imshow(train_images[i], cmap='gray')
        # plt.show()
    print('done')
    return (train_X, train_Y, Test_X, Test_Y)


def equal(x, y):
    for i in range(len(x)):
        if x[i] != y[i]:
            return False
    return True


def find_max(x):
    max_index = 0
    max = -10000
    for i in range(len(x)):
        if x[i] > max:
            max = x[i]
            max_index = i
    return max_index


class NN:
    def __init__(self, n):
        '''用MSE作为cost function'''
        self.cell = []  # 保存每个神经单元的输出值
        self.prepare = []
        self.weight = [[]]  # self.weight[k][j][i]表示第k层第j个神经元与k-1层第i个单元格的链接权值
        self.threshold = [0]
        self.active = [0]  # 第对应层神经单元的激活函数
        self.n = n  # n是学习率
        self.g = [[]]  # 用于存储对应层神经单元MSE对该单元输入的负梯度
        self.MSE = []  # 每一轮的MSE
        self.y = []
        self.x = []
        self.cost_function = 'MSE'
        self.juge = 0

    def sigmoid(self, x):
        '''神经网络单元格都选作sigmoid作为'''
        return 1 / (1 + np.exp(-x))

    def deriv_sig(self, x):
        '''sigmoid的导数'''
        return x * (1 - x)

    def tanh(self, x):
        if x < 0.000000000001:
            x = 0.000000000001
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def deriv_cost_function(self, cost, y, y1, k):
        sum = 0
        if cost == 'MSE':
            # print(y,y1)
            for i in range(len(y)):
                sum += y[i][k] - y1[i][k]
                # print(y[i][k],y1[i][k])
            # print(k,sum)
            return sum

    def MSED(self, y, y1):
        sum = 0
        for i in range(len(y)):
            sum += (y[i] - y1[i]) * (y[i] - y1[i])
        return sum

    def deriv_tanh(self, x):
        return 1 - x ** 2

    def relu(self, x):
        return max(0, x)

    def deriv_relu(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def calculte_cell(self, x, active):
        '''x是单元格的输入,active是激活函数'''
        if active == 'sigmoid':
            return self.sigmoid(x)
        if active == 'tanh':
            return self.tanh(x)
        if active == 'relu':
            return self.relu(x)

    def calculte_cell_gradient(self, x, active):
        '''计算神经元的梯度'''
        if active == 'sigmoid':
            return self.deriv_sig(x)
        if active == 'tanh':
            return self.deriv_tanh(x)
        if active == 'relu':
            return self.deriv_relu(x)

    def Dense(self, dim=0, active='sigmoid'):
        '''创建一层全连接的神经单元，神经单元个数=dim,神经单元激活函数默认为sigmoid'''
        self.prepare.append([dim, active])

    def initial(self, x):
        '初始化网络'
        self.cell.append([])
        for t in range(len(self.prepare)):
            item = self.prepare[t]
            self.cell.append([0 for i in range(item[0])])
            # 增加神经单元
            self.active.append(item[1])
            # 保存对应的激活函数
            self.threshold.append([random.uniform(-0.2, 0.2) for i in range(item[0])])
            self.g.append([0 for i in range(item[0])])
            # 增加阀值
            if t != 0:
                last_item = self.prepare[t - 1][0]
            else:
                last_item = len(x)
            self.weight.append([[random.uniform(-0.2, 0.2) for j in range(last_item)] for i in range(item[0])])
            # 保存权重
        if self.active[-1] == 'sigmoid':
            self.juge = 0.5
        if self.active[-1] == 'tanh':
            self.juge = 0
        if self.active[-1] == 'relu':
            self.juge = 0
        # 初始化神经单元值的保存
        '''初始化网络'''

    def ADD(self, x, y):
        ':return x+y'
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def minor(self, x, y):
        ':return x-y'
        z = []
        for i in range(len(x)):
            z.append(x[i] - y[i])
        return z

    def sub(selfs, x, k):
        z = []
        for i in range(len(x)):
            z.append(x[i] / k)
        return z

    def fit(self, X, Y, epoch=5, cost_function='MSE', batch=20):
        self.initial(X[0])
        for ep in range(epoch):
            print("第{}次训练".format(ep))
            MSES = 0
            account = 0
            total_MSE = [0 for i in range(len(self.cell[-1]))]
            iteration = int(len(X) / batch)
            if iteration != len(X) / batch:
                iteration += 1
            for i in range(iteration):
                batch_output = []
                batch_expect = []
                for j in range(min(batch, len(X) - i * batch)):
                    # print(i,j)
                    self.cell[0] = X[i * batch + j]
                    self.y = Y[i * batch + j]
                    self.update_net_forward()
                    # print('各层神经元值')
                    # for i in range(len(self.cell)):
                    # print(self.cell[i])
                    # print('各层神经元收到的权重')
                    # for j in range(len(self.weight)):
                    # print(self.weight[j])
                    batch_output.append(self.cell[-1][:])
                    batch_expect.append(self.y[:])
                    item = [false_label for i in range(len(self.y))]
                    this_MSE = self.minor(self.cell[-1], self.y)
                    total_MSE = self.ADD(this_MSE, total_MSE)
                    item[find_max(self.cell[-1])] = 1
                    if equal(item, self.y):
                        account += 1
                    else:
                        pass
                    MSES += self.MSED(self.cell[-1], self.y)
                    # print('MSE:',MSES/2)
                # print('期望输出',batch_expect)
                self.update_bp(batch_output, batch_expect)
            print('训练集正确率', account / len(X))
            print('平均误差', MSES / len(X))
            if MSES / len(X) <= 0.0001:
                print(self.cell[-1])
                break
            print()

    def update_net_forward(self):
        '''从输入层开始，从前往后，根据输入层的值更新网络单元格的值'''
        # print(self.weight)
        for k in range(1, len(self.cell)):
            for j in range(len(self.cell[k])):
                # 计算第k层第j个单元格的值
                sum = 0
                for i in range(len(self.cell[k - 1])):
                    # print(k,j,i)
                    sum += self.cell[k - 1][i] * self.weight[k][j][i]
                # print(self.cell[k-1],self.weight[k][j],self.threshold[k][j])
                # print(sum + self.threshold[k][j])
                self.cell[k][j] = self.calculte_cell(sum + self.threshold[k][j], self.active[k])
                # print(self.cell[k][j])

    def update_bp(self, batch_output, batch_expect):
        '''bp更新策略'''
        for k in [i for i in range(1, len(self.active))][::-1]:
            '''首先更新到k层的神经单元的梯度项，然后更新阀值,然后更新第k层的神经单元与k-1层连接的权重'''
            for i in range(len(self.g[k])):
                # 第k层第i个神经元
                if k == len(self.active) - 1:
                    self.g[k][i] = -1 * self.deriv_cost_function(self.cost_function, batch_output, batch_expect,
                                                                 i) * self.calculte_cell_gradient(self.cell[k][i],
                                                                                                  active=self.active[k])
                    # print('{}层{}神经元误差导数为'.format(k,i),self.deriv_cost_function(self.cost_function,batch_output,batch_expect,i))
                    # print('{}层{}神经元求导为{}'.format(k,i,self.calculte_cell_gradient(self.cell[k][i],active=self.active[k])))
                    # print('输出层误差为',self.g[k])
                else:
                    # 计算隐藏层残差
                    sum = 0
                    for j in range(len(self.g[k + 1])):
                        sum += self.g[k + 1][j] * self.weight[k + 1][j][i]
                    # 首先是 负的 k+1层所有神经元的梯度项*与之相连的梯度相乘，最后是与该神经单元的激活函数的导数相乘
                    self.g[k][i] = self.calculte_cell_gradient(self.cell[k][i], active=self.active[k]) * sum
                # print('隐藏层误差为',self.g[k])
        for k in range(1, len(self.active)):
            for i in range(len(self.g[k])):
                # self.threshold[k][i] = self.threshold[k][i] + self.n*self.g[k][i]
                # 更新阀值
                for j in range(len(self.cell[k - 1])):
                    # print("{}层{}神经元与上一层{}神经元连接权重变动值为{}".format(k,i,j,self.n*self.g[k][i]*self.cell[k-1][j]))
                    # print('原先值为{}'.format(self.weight[k][i][j]))
                    self.weight[k][i][j] = self.weight[k][i][j] + self.n * self.g[k][i] * self.cell[k - 1][j]
                    # print('更新为',self.weight[k][i][j])
                # 更新权重:

    def predict(self, X, y):
        Y = []
        for x in X:
            self.cell[0] = x
            self.update_net_forward()
            print('图片数字识别结果为', find_max(self.cell[-1]))
            show(x)
            item = [false_label for i in range(len(self.cell[-1]))]
            item[find_max(self.cell[-1])] = 1
            Y.append(item)
        acc_count = 0
        for i in range(len(Y)):
            if equal(Y[i], y[i]):
                acc_count += 1
        print('预测集正确率', acc_count / len(Y))
        return Y

    def plot(self, x, y):
        figure = plt.figure()
        for i in range(len(x)):
            if y[i][1] == 1:
                plt.scatter(x[i][0], x[i][1], c='red', marker='*')
            else:
                plt.scatter(x[i][0], x[i][1], c='black', marker='x')
        raw_x = np.linspace(-10, 10, 100)
        for i in range(len(self.weight[-1])):
            weight = self.weight[-1][i]
            threshold = self.threshold[-1][i]
            y = []
            for item in raw_x:
                y.append(-1 * threshold / weight[1] - weight[0] * item / weight[1])
            plt.plot(raw_x, y)
        plt.show()


train_X, train_Y, Test_X, Test_Y = run()
Handwritting = NN(0.5)
Handwritting.Dense(40, active='sigmoid')
Handwritting.Dense(10, active='sigmoid')
Handwritting.fit(train_X[:cut], train_Y[:cut], epoch=12, batch=1)
Handwritting.predict(Test_X, Test_Y)
