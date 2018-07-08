import tensorflow as tf
# 案例中采用captcha包生成验证码
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# 数字、小写字母、大写字母
number = ['0','1','2','3','4','5','6','7','8','9']
# alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def random_captcha_text(char_set=number,captcha_size=4):
    """
    生成一个列表，列表内包含captcha_size个数的，从char_set中随机挑选的，text列表
    """
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

def gen_captcha_text_and_image():
    """
    随机生成验证码
    :return: 验证码内容，验证码图片（np.arr格式）
    """
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = "".join(captcha_text)

    captcha = image.generate(captcha_text)

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text,captcha_image

def convert2gray(img):
    """
    把图像转换成灰度图像
    因在验证码识别中，图像是灰色和彩色，对结果影响并不大
    """
    if len(img.shape) > 2:
        gray = np.mean(img,-1)
        return gray
    else:
        return img

    # 网上查阅的彩色图像转化成灰色图像有很多方法
    # 上面的是快速转换方法，直接取平均值
    # 一般正规方法为 gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

def text2vec(text):
    """
    文字内容处理,判断文字内容长度是否大于最大长度
    """
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码长度最长为%s个字符' %MAX_CAPTCHA)

    # 生成标签0矩阵
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)

    # 按照text内容给标签0矩阵填充值
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + int(c)
        vector[idx] = 1
    return vector

def vec2text(vec):
    """
    向量转化为文本
    :param vec: 输出向量
    :return: 文本text
    """
    text = []
    # 查找非空的索引值
    char_pos = vec.nonzero()[0]
    for i, c in enumerate(char_pos):
        number = i % 10
        text.append(str(number))

    return "".join(text)

def get_next_batch(batch_size=128):
    """
    生成一个训练batch
    :param batch_size:
    :return:
    """

    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255    # 转成一维，并进行归一化
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y

def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])    # 根据tensorflow的要求，输入CNN的格式要是4维的

    # 以下生成一个3层的CNN网络
    # conv1
    # 使用w_alpha和b_alpha的目的是让生成的w和b尽量小
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,1,32]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding="SAME"),b_c1))
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    conv1 = tf.nn.dropout(conv1,keep_prob)

    # conv2
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3,3,32,64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,w_c2,strides=[1,1,1,1],padding="SAME"),b_c2))
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    conv2 = tf.nn.dropout(conv2,keep_prob)

    # conv3
    w_c3 = tf.Variable(w_alpha*tf.random_normal([3,3,64,64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2,w_c3,strides=[1,1,1,1],padding="SAME"),b_c3))
    conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    conv3 = tf.nn.dropout(conv3,keep_prob)

    # Fully connected layer 全链接层
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv3,[-1,w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense,w_d),b_d))
    dense = tf.nn.dropout(dense,keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense,w_out),b_out)

    return out

def train_crack_captcha_cnn():
    output = crack_captcha_cnn()   # 使用变量接收返回值
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=output))   # 定义损失函数
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)                   # 定义最优化目标
    predict = tf.reshape(output,[-1,MAX_CAPTCHA,CHAR_SET_LEN])                               # reshape计算出的预测值
    max_idx_p = tf.arg_max(predict,2)                                                        # 求出预测的数字结果
    Y_label = tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN])                                 # reshape Y
    max_idx_l = tf.arg_max(Y_label, 2)                                                       # 求出Y的预测结果
    correct_pred = tf.equal(max_idx_l,max_idx_p)                                             # 求出哪些结果是正确预测的，数据为布尔值
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))                              # 求出准确度

    saver = tf.train.Saver()    # 保存计算结果
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        end_count = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _,loss_ = sess.run([optimizer, loss], feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            print(step, loss_)


            if step % 100 == 0:     # 每100步输出一次准确率
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy,feed_dict={X:batch_x_test,Y:batch_y_test,keep_prob:1.})
                print(step, acc)
                if acc > 0.98:
                    end_count += 1
                if end_count > 10 and step >= 5000:   # 当准确率大于0.5的时候，把结果保存，并结束训练
                    saver.save(sess,"./model/crack_capcha%d.model", global_step=step)
                    break

            step += 1

def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,"./model/crack_capcha%d.model-1500")
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        text = text_list[0].tolist()
        return text

if __name__ == '__main__':
    train = 0     # 训练时候，取0。   验证的时候，取1
    if train == 0:
        number = ['0','1','2','3','4','5','6','7','8','9']
        text, image = gen_captcha_text_and_image()
        print("验证码图像channel:",image.shape)    # 打印图像大小(60,160,3)

        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        MAX_CAPTCHA = len(text)
        print("验证码文本最长字符数", MAX_CAPTCHA)
        char_set = number
        CHAR_SET_LEN = len(char_set)

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
        keep_prob = tf.placeholder(tf.float32) # dropout

        train_crack_captcha_cnn()

    if train == 1:
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        char_set = number
        CHAR_SET_LEN = len(char_set)


        text, image = gen_captcha_text_and_image()

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)

        # plt.show()

        MAX_CAPTCHA = len(text)
        image = convert2gray(image)
        image = image.flatten() / 255

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        keep_prob = tf.placeholder(tf.float32)  # dropout

        predict_text = crack_captcha(image)
        print("正确: {}  预测: {}".format(text, predict_text))





