import pandas as pd
import math
import numpy as np

xigua = pd.read_csv("xigua3.0.csv", encoding='gbk')
y = xigua.好坏
features = xigua.columns[1:-1]
x = xigua.values[:, 1:]


class DecisionTree:
    def __init__(self, data):
        '每个数据向量格式必须为[x,y]'
        self.data = data
        self.Tree = [[data]]
        self.featrues = features
        # 用列表保存每一层的结构

    def entropy(self, D):
        '''计算一个数据集的entropy,数据集D需要为np.array格式'''
        count = {}
        total_count = 0
        sum = 0
        for item in D:
            if item[-1] not in count:
                count[item[-1]] = 1
            else:
                count[item[-1]] += 1
            total_count += 1
        for key in count:
            possiblity = round(count[key] / total_count, 8)
            sum += -1 * possiblity * math.log2(possiblity)
        return sum

    def Gini(self, D):
        '''计算一个数据集的GINI指数'''
        count = {}
        total_count = 0
        sum = 0
        for item in D:
            if item[-1] not in count:
                count[item[-1]] = 1
            else:
                count[item[-1]] += 1
            total_count += 1
        for key in count:
            possibility = round(count[key] / total_count, 8)
            sum += possibility * possibility
        # print(D)
        # print(count)
        return 1 - sum

    def Decision(self, D, cost_function=1):
        '''对于某一个具体的非叶节点，进行划分，返回当前选择的特征，与划分后的数据集'''
        subsets_of_total_discrete_feature = {}
        entropy_of_father = self.entropy(D)
        '''所有离散属性值的划分'''
        indexes_of_continous_feature = []
        for i in range(len(features)):
            try:
                '''表明当前特征是连续特征'''
                D[0][i] = float(D[0][i])
                indexes_of_continous_feature.append(i)
            except:
                subsets_of_total_discrete_feature[i] = self.feature_divide_discrete(D, i)

        if cost_function == 1:
            '''表明使用信息增益来选择最优化划分'''
            '''首先计算离散特征的信息增益'''
            max_gain = 0  # 离散特征的最大信息增益
            max_gain_index = -1  # 离散特征最大信息增益时的特征下标
            max_gain_partition_label = []  # 离散特征最大信息增益时的特征标签
            max_gain_partition = []  # 离散特征最大信息增益时的对应标签划分
            for key in subsets_of_total_discrete_feature:
                '''计算当前离散特征划分下的信息增益，并选择一个最大的增益'''
                # print('计算{}的信息增益'.format(features[key]))
                # print('划分标签为{}'.format(subsets_of_total_discrete_feature[key][0]))
                # print('划分为:')
                entropy_sum = 0
                for subset in subsets_of_total_discrete_feature[key][1]:
                    entropy_sum += len(subset) * self.entropy(subset) / len(D)
                gain = entropy_of_father - entropy_sum
                if max_gain < gain:
                    max_gain = gain
                    max_gain_index = key
                    max_gain_partition_label = subsets_of_total_discrete_feature[key][0]
                    max_gain_partition = subsets_of_total_discrete_feature[key][1]
            '''下面计算'''
            max_gain_contious = 0  # 连续特征的最大信息增益
            max_gain_index_contious = -1  # 连续特征最大信息增益时的特征下标
            max_gain_partition_label_contious = 0  # 连续特征最大信息增益时的特征标签
            max_gain_partition_contious = []  # 连续特征最大信息增益时的对应标签划分
            for key in indexes_of_continous_feature:
                '''key是一个确定的连续特征变量'''
                y = sorted(D[:, key])
                cuts = [(float(y[i]) + float(y[i + 1])) / 2 for i in range(len(y) - 1)]
                for cut in cuts:
                    bigger = []
                    smaller = []
                    for item in D:
                        if float(item[key]) >= cut:
                            bigger.append(item)
                        else:
                            smaller.append(item)
                    '''划分为大于等于与小于两个子集'''
                    this_gain = entropy_of_father - (len(bigger)) / len(D) * self.entropy(bigger) - (
                                len(smaller) / len(D)) * self.entropy(smaller)
                    if this_gain > max_gain_contious:
                        max_gain_contious = this_gain
                        max_gain_index_contious = key
                        max_gain_partition_label_contious = cut
                        max_gain_partition_contious = [bigger, smaller]
            if max(max_gain, max_gain_contious) <= 0:
                return 0
            if max_gain < max_gain_contious:
                return (max_gain_contious, features[max_gain_index_contious], max_gain_partition_label_contious,
                        max_gain_partition_contious)
            return (max_gain, features[max_gain_index], max_gain_partition_label, max_gain_partition)
        if cost_function == 2:
            mini_gini = 1  # 离散特征的最小基尼值
            mini_gini_index = -1  # 离散特征最小基尼值时的特征下标
            mini_gini_partition_label = []  # 离散特征最小基尼值时的特征标签
            mini_gini_partition = []  # 离散特征最小基尼值时的对应标签划分
            gini_father_node = self.Gini(D)
            # print(gini_father_node)
            for key in subsets_of_total_discrete_feature:
                this_gini = 0
                for subset in subsets_of_total_discrete_feature[key][1]:
                    this_sub_set_gini = self.Gini(subset)
                    this_gini += this_sub_set_gini * len(subset) / len(D)
                if this_gini < mini_gini:
                    mini_gini = this_gini
                    mini_gini_index = key
                    mini_gini_partition_label = subsets_of_total_discrete_feature[key][0]
                    mini_gini_partition = subsets_of_total_discrete_feature[key][1]
            mini_gini_contious = 1  # 连续特征的最大最小基尼值增益
            mini_gini_index_contious = -1  # 连续特征最大最小基尼值时的特征下标
            mini_gini_partition_label_contious = 0  # 连续特征最大最小基尼值时的特征标签
            mini_gini_partition_contious = []  # 连续特征最小基尼值时的对应标签划分
            for key in indexes_of_continous_feature:
                '''key是一个确定的连续特征变量'''
                y = sorted(D[:, key])
                cuts = [(float(y[i]) + float(y[i + 1])) / 2 for i in range(len(y) - 1)]
                for cut in cuts:
                    bigger = []
                    smaller = []
                    for item in D:
                        if float(item[key]) >= cut:
                            bigger.append(item)
                        else:
                            smaller.append(item)
                    '''划分为大于等于与小于两个子集'''
                    this_gini = self.Gini(bigger) * len(bigger) / len(D) + self.Gini(smaller) * len(smaller) / len(D)
                    # print(key,cut,this_gini)
                    if this_gini < mini_gini_contious:
                        mini_gini_contious = this_gini
                        mini_gini_index_contious = key
                        mini_gini_partition_label_contious = cut
                        mini_gini_partition_contious = [bigger, smaller]
            if min(mini_gini, mini_gini_contious) >= gini_father_node:
                return 0
            if mini_gini_contious < mini_gini:
                return (mini_gini_contious, features[mini_gini_index_contious], mini_gini_partition_label_contious,
                        mini_gini_partition_contious)
            return (mini_gini, features[mini_gini_index], mini_gini_partition_label, mini_gini_partition)

    def feature_divide_discrete(self, D, feature_index):
        '''给定一个数据集,指定一个离散特征值,将样本划分为多个子数据集,返回划分标签与对应的划分子集'''
        paritition = []
        if len(D) == 0:
            return paritition
        value = []  # 用于保存已经出现过的feature值
        for item in D:
            if item[feature_index] not in value:
                value.append(item[feature_index])
                paritition.append([np.array(item).tolist()])
            else:
                paritition_index = value.index(item[feature_index])
                paritition[paritition_index].append(np.array(item).tolist())
        return (value, paritition)

    def outcome(self, D):
        count = {}
        for item in D:
            if item[-1] not in count:
                count[item[-1]] = 1
            else:
                count[item[-1]] += 1
        return sorted(count, key=lambda e: count[e], reverse=True)[0]

    def fit(self):
        count = 0
        while True:
            next_layer = []
            for i in range(len(self.Tree[count])):
                '对第count+1层节点进行分裂'
                node = self.Tree[count][i]
                Decision = self.Decision(np.array(node), cost_function=2)
                if Decision == 0:
                    print('第{}层第{}节点停止分裂'.format(count + 1, i + 1))
                    # print(node)
                    print('叶子结果为{}'.format(self.outcome(node)))
                    continue
                else:
                    print(
                        '第{}层第{}节点分裂:属性{};分裂标签{},GINI{}'.format(count + 1, i + 1, Decision[1], Decision[2],
                                                                             Decision[0]))
                    for item in Decision[-1]:
                        # print(item)
                        next_layer.append(item)
            count += 1
            if next_layer == []:
                print('决策树生长完毕')
                break
            self.Tree.append(next_layer)
            print('--------------')


xigua = DecisionTree(x).fit()
