import tensorflow as tf
import numpy as np
import pandas as pd
from handle_data import get_data


INPUT_NODE = 784                # 输入的每张图片shape
OUTPUT_NODE = 10                # 输出的每张图片维度
LAYER1_NODE = 500               # 隐藏层1单元个数
LAYER2_NODE = 100               # 隐藏层2单元个数
MOVING_AVERAGE_DECAY = 0.99     # 平均滑动参数
REGULARIZATION_RATE = 0.0001    # 正则化比率
BATCH_SIZE = 64                 # mini-batch大小
LEARNING_RATE_BASE = 0.001     # 基础的学习率
LEARNING_RATE_DECAY = 0.99      # 学习率衰减参数
TRAINING_STEPS = 5000           # 训练步数

def inference(input_tensor, avg_class, weights1, bias1, weights2, bias2, weights3, bias3, lambd):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1)+bias1)
        # 将正则化项结果加入到collection中
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(weights1))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights2)+bias2)
        # 将正则化项结果加入到collection中
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(weights2))
        layer3 = tf.matmul(layer2, weights3) + bias3
        # 将正则化项结果加入到collection中
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(weights3))
        return layer3
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1)+bias1)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(weights1))
        layer2 = tf.nn.relu(tf.matmul(layer1, avg_class.average(weights2))+avg_class.average(bias2))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(weights2))
        layer3 = tf.matmul(layer2, avg_class.average(weights3)) + avg_class.average(bias3)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(weights3))
        return layer3

def train():
    # 获取训练数据
    train_image_data, train_label_data = get_data()
    # 获取训练集数量
    train_num = train_image_data.shape[0]

    # 建立x和y_占位符
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')

    # 初始化变量W1，b1,W2,b2
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[LAYER2_NODE]))
    weights3 = tf.Variable(tf.truncated_normal([LAYER2_NODE, OUTPUT_NODE], stddev=0.1))
    bias3 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 定义全局步数
    global_step = tf.Variable(0,trainable=False)

    # 制定滑动模型
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    # 把所有的trainable参数加入模型中
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算average_y
    average_y = inference(x, variable_averages, weights1, bias1, weights2, bias2, weights3, bias3, REGULARIZATION_RATE)

    mse_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=average_y,labels=tf.argmax(y_,1)))
    # 将平均损失函数加入collection
    tf.add_to_collection('losses', mse_loss)
    # 取出并求和loss
    # 应用将loss加入collection的方法，在神经网络深度较深时，可以增强代码易读性，可以加速计算，节省资源
    loss = tf.add_n(tf.get_collection('losses'))

    # 设定学习率下降参数
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    # 设定最优化目标
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 一次性完成所有的参数更新
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 进行准确率预测
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        # 初始化所有变量参数
        tf.global_variables_initializer().run()
        # 计算每一轮循环要有多少个batch
        num_batch = int((train_num - 1)/BATCH_SIZE) + 1

        # 初始化模型保存
        saver = tf.train.Saver()

        # 训练TRAINING_STEPS次
        for i in range(TRAINING_STEPS):
            # 随机打乱数据
            arr = np.arange(train_num)
            np.random.shuffle(arr)
            train_image_data_shu = train_image_data[arr]
            train_label_data_shu = train_label_data[arr]

            #
            validate_feed = {
                x: train_image_data,
                y_: train_label_data
            }

            print('进行第%d次训练'%i)

            # 每训练100次计算一次整体数据的准确率,并保存模型
            if i % 100 == 0:
                # 验证整体数据准确率
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("%d 次训练，整体数据正确率：%g" %(i, validate_acc))
                # 保存当前模型
                saver.save(sess, './model/kaggle.ckpt', global_step=global_step)
                print("保存模型变量成功！")

            # 循环所有BATCH
            for k in range(num_batch):
                # 计算每个batch开始和结束的索引值
                start = k*BATCH_SIZE
                end = min(train_num, start+BATCH_SIZE)
                # 获取当前batch下的x,y数据
                xs, ys = train_image_data_shu[start:end], train_label_data_shu[start:end]
                sess.run(train_op, feed_dict={x: xs, y_: ys})
                # 每循环100次，验证第一个batch的准确率
                if i % 10 == 0 and k == 0:
                    batch_acc = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                    print("%d 次训练，整体数据正确率：%g" %(i, batch_acc))


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()