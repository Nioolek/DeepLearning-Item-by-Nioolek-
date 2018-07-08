from handle_data import get_test_data
from train import inference,INPUT_NODE,OUTPUT_NODE,LAYER1_NODE,LAYER2_NODE
import tensorflow as tf
import pandas as pd

def predict(filename):
    # 获取test数据
    test_data = get_test_data()

    # 建立x和y_占位符
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    # y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')

    # 初始化变量W1，b1,W2,b2
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[LAYER2_NODE]))
    weights3 = tf.Variable(tf.truncated_normal([LAYER2_NODE, OUTPUT_NODE], stddev=0.1))
    bias3 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, bias1, weights2, bias2, weights3, bias3, 0.0)
    result = tf.argmax(y,1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess,filename)
        pre = sess.run(result,feed_dict={x: test_data})
        dataframe = pd.DataFrame({'prediction':pre})
        dataframe.to_csv("./data/test_pre.csv", index=False, sep=',')

def main(argv=None):
    filename = './model/kaggle.ckpt-525600'
    predict(filename)

if __name__ == '__main__':
    tf.app.run()