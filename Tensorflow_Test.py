# 2018.3.1
# Learn structure of Tensorflow
# episode 1
# 
# 

import tensorflow as tf
import numpy as np

# 1.create data
data_x = np.random.rand(100).astype(np.float32)
data_y = data_x*0.3 + 0.9

# 2.create tensorflow structure
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 随机生成-1.0到1.0的一位权值数列
biases = tf.Variable(tf.zeros([1]))                       # 创建初始值为0的一位偏置

predicted_y = Weights*data_x + biases                     # 预测的y值

loss = tf.reduce_mean(tf.square(predicted_y - data_y))    # 误差
optimizer = tf.train.GradientDescentOptimizer(0.5)        # 定义学习效率（小于1.0）为0.5的优化器
train = optimizer.minimize(loss)                          # 用优化器减小误差

init = tf.global_variables_initializer()                  # 初始化所有变量

# 3.activate tensorflow structure
sess = tf.Session()
sess.run(init)                                            # 激活所有变量

for step in range(100):
    sess.run(train)
    if step % 20 ==0:
        print('step:', step, 'Weights:', sess.run(Weights), 'biases:', sess.run(biases))