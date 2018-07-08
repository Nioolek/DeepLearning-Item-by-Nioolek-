import tensorflow as tf

# 神经网络相关参数
INPUT_NODE = 784
OUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 卷积层1的size和deep
CONV1_SIZE = 5
CONV1_DEEP = 32

# 卷积层2的size和deep
CONV2_SIZE = 5
CONV2_DEEP = 64

# 全连接层的size
FC_SIZE = 512

# 卷积网络前向传播过程
def inference(input_tensor,train,regularizer):
    # 卷积操作
    with tf.variable_scope('layer1-conv1'):
        # 卷积层1的变量初始化
        conv1_weights = tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))

        # 使用边长为5，深度为32的过滤器，过滤器步长为1，用全0填充
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        # 添加bias，并使用relu
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    # 使用池化层
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

    # 卷积操作
    with tf.variable_scope('layer3-conv2'):
        # 卷积层2的变量初始化
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用边长为5，深度为64的过滤器，过滤器步长为1，用全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights,strides=[1, 1, 1, 1], padding='SAME')

        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 使用池化层
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # 获取pool2的shape
    pool_shape = pool2.get_shape().as_list()
    print(pool_shape)
    # 计算全连接层的维度
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    # 将池化层结果转化为全连接层输入
    reshaped = tf.reshape(pool2, [-1, nodes])

    # 使用全连接层1
    with tf.variable_scope('layer5-fc1'):
        # 初始化全连接层参数
        fc1_weights = tf.get_variable("weight",[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias",[FC_SIZE],initializer=tf.constant_initializer(0.1))
        # 全连接层向前传播
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        # 如果是训练模式，那么使用dropout
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    # 使用全连接层2
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight",[FC_SIZE,NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fc2_weights))
        fc2_biases = tf.get_variable('vias',[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weights)+fc2_biases


    # 返回第六层的输出
    return logit




