import math
import numpy as np
import sys
import scipy.io as sio
from PIL import Image
import os

def sigmod(x):
    return np.array(list(map(lambda i: 1 / (1 + math.exp(-i)), x)))


def get_train_pattern():
    current_dir = "D:/bp1/"
    train = sio.loadmat(current_dir + "mnist_train.mat")["mnist_train"]
    train_label = sio.loadmat(
        current_dir + "mnist_train_labels.mat")["mnist_train_labels"]
    train = np.where(train > 180, 1, 0)  #二值化
    return train, train_label


def get_test_pattern():
    base_url = "D:/bp1/mnist_test/"
    test_img_pattern = []
    for i in range(10):
        img_url = os.listdir(base_url + str(i))
        t = []
        for url in img_url:
            img = Image.open(base_url + str(i) + "/" + url)
            img = img.convert('1')   # 二值化
            img_array = np.asarray(img, 'i')  # 转化为int数组
            img_vector = img_array.reshape(
                img_array.shape[0] * img_array.shape[1]) #展开成一维数组
            t.append(img_vector)
        test_img_pattern.append(t)
    return test_img_pattern

class BPNetwork:
    # 神经网络类
    def __init__(self,in_count, hiden_count, out_count, in_rate, hiden_rate):
        """

        :param in_count: 输入层数
        :param hiden_count: 隐藏层数
        :param out_count: 输出层数
        :param in_rate: 输入层学习率
        :param hiden_rate: 隐藏层学习率
        :return:
        """
        # 各个层的节点数量
        self.in_count = in_count
        self.hiden_count = hiden_count
        self.out_count = out_count

        # 输入层到隐藏层连线的权重随机初始化
        self.w1 = 0.2 * \
            np.random.random((self.in_count, self.hiden_count)) - 0.1
        # 隐藏层到输出层连线的权重随机初始化
        self.w2 = 0.2 * \
            np.random.random((self.hiden_count, self.out_count)) - 0.1
        # 隐藏层偏置向量
        self.hiden_offset = np.zeros(self.hiden_count)
        self.out_offset = np.zeros(self.out_count)

        # 输入层学习率.
        self.in_rate = in_rate
        # 隐藏层学习率
        self.hiden_rate = hiden_rate

    def train(self,train_img_pattern, train_label):
        if self.in_count != len(train_img_pattern[0]):
            sys.exit("输入层维数与样本维数不等")
        for i in range(len(train_img_pattern)):
            if i % 5000 == 0:
                print(i)
            # 生成目标向量
            target = [0] * 10
            target[train_label[i][0]] = 1
            # 前向传播
            # 隐藏层值等于输入层 * w1 + 隐藏层偏置
            hiden_value = np.dot(
                train_img_pattern[i], self.w1) + self.hiden_offset
            hiden_value = sigmod(hiden_value)

            #计算输出层的输出
            out_value = np.dot(hiden_value, self.w2) + self.out_offset
            out_value = sigmod(out_value)

            # 反向更新
            error  = target - out_value
            # 计算输出层误差
            out_error = out_value * (1 - out_value) * error
            # 计算隐藏层误差
            hiden_error = hiden_value * \
                          (1 - hiden_value) * np.dot(self.w2, out_error)

            # 更新w2 ,w2是j行k列的矩阵,存储隐藏层到输出层的权值
            for k in range(self.out_count):
                # 更新w2第k列的值,连接隐藏层所有节点到输出层的第k个节点的边
                # 隐藏层学习率*输入层误差*隐藏层的输出值
                self.w2[:, k] += self.hiden_rate * out_error[k] * hiden_value
            # 更新w1
            for j in range(self.hiden_count):
                self.w1[:, j] += self.in_rate * \
                    hiden_error[j] * train_img_pattern[i]

            # 更新偏置向量
            self.out_offset += self.hiden_rate * out_error
            self.hiden_offset += self.in_rate * hiden_error


    def test(self, test_img_pattern):
        """
        测试神经网络的正确率
        :param test_img_pattern:
        :return:
        """
        right = np.zeros(10)
        test_sum = 0
        for num in range(10):
            num_count = len(test_img_pattern[num])
            test_sum += num_count
            for t in range(num_count): #数字num的第t张图片
                 hiden_value = np.dot(
                 test_img_pattern[num][t],self.w1) + self.hiden_offset
                 hiden_value = sigmod(hiden_value)
                 out_value = np.dot(hiden_value,self.w2) + self.out_offset
                 out_value = sigmod(out_value)

                 if np.argmax(out_value) == num:
                     # 识别正确
                     right[num] += 1
            print("数字%d的识别正确率%f" % (num, right[num] / num_count))
            # 平均识别率
        print("平均识别率为: ", sum(right) / test_sum)


def run():
    # 读入训练集
    train, train_label = get_train_pattern()
    # 读入测试图片
    test_pattern = get_test_pattern()
    # 神经网络配置参数
    in_coount = 28 * 28
    hiden_count = 6
    out_count = 10
    in_rate = 0.1
    hiden_rate = 0.1
    bpnn = BPNetwork(in_coount, hiden_count, out_count, in_rate, hiden_rate)
    bpnn.train(train,train_label)
    bpnn.test(test_pattern)


if __name__ == "__main__":
    run()


