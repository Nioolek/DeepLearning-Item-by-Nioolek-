import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def make_one_hot(data1):
    return (np.arange(10)==data1[:, None]).astype(np.float32)

def handle_and_save_train_data():
    # 读取训练集数据
    train_data = pd.read_csv("./data/train.csv")
    # print(train_data.dtypes)

    label = train_data['label']
    train_data = train_data / 255
    train_data['label'] = label

    train_data.to_csv('./data/train_max1.csv')

def handle_and_save_test_data():
    # 读取测试集数据
    test_data = pd.read_csv("./data/test.csv")
    # print(test_data.shape)
    test_data = np.array(test_data) / 255
    return test_data

def show_data(index):
    """
    利用可视化函数matplotlib.pyplot展示数据图片
    :param index: 图片的索引值
    :return:
    """
    train_data = pd.read_csv("./data/train.csv")
    # print(train_data.loc[0])
    data = np.array(train_data.loc[index])[1:]
    image = data.reshape([28,28])
    print(image)
    # 创建画布
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # cmap表示使用灰度图
    ax1.imshow(image, cmap=plt.cm.gray)
    plt.show()

def get_data():
    """
    用于训练期间获取训练数据
    :return:
    """
    train_data = pd.read_csv("./data/train_max1.csv")

    # 处理训练数据
    del train_data['Unnamed: 0']
    # print(train_data.columns.tolist())
    train_data = np.array(train_data)
    train_label_data = train_data[:,0]
    # print(train_label_data.shape)
    train_image_data = train_data[:,1:]
    # print(train_image_data.shape)
    # print(train_data.shape)
    train_label_data = make_one_hot(train_label_data)

    return train_image_data,train_label_data

def get_test_data():
    test_data = np.array(pd.read_csv("./data/test.csv"))/255
    # print(test_data.shape)
    # print(test_data[0][100:400])
    return test_data

if __name__ == '__main__':
    # handle_and_save_train_data()
    # handle_and_save_test_data()
    # show_data(2)
    # get_data()
    # print(make_one_hot(np.array([1,2,3])))
    get_test_data()