from skimage import io, transform
from matplotlib import pyplot
import glob
import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
path = 'C:/Users/36327/Desktop/face'

# 将所有的图片resize成100*100
w = 128
h = 128
c = 3

# 读取图片
def read_img(path):
    i = 0
    cate = [path + '/' + i for i in os.listdir(path) if os.path.isdir(path + '/' + i)]
    imgs = []
    labels = []
    print(cate)
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % im)
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(folder[28:])
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)


#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]



#将所有数据分为训练集和验证集
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]


#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]



#-----------------构建网络----------------------
#占位符
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')


def CNNlayer():
    # 第一个卷积层（128——>64)
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二个卷积层(64->32)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三个卷积层(32->16)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # 第四个卷积层(16->8)
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool4, [-1, 8 * 8 * 128])

    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits = tf.layers.dense(inputs=dense2,
                             units=60,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    return logits


# ---------------------------网络结束---------------------------
logits = CNNlayer()
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据，可将n_epoch设置更大一些
saver = tf.train.Saver(max_to_keep=3)
max_acc = 0
f = open('C:/Users/36327/Desktop/face/val acc.txt', 'w')
g = open('C:/Users/36327/Desktop/face/train acc.txt', 'w')
h = open('C:/Users/36327/Desktop/face/train loss.txt', 'w')
k = open('C:/Users/36327/Desktop/face/val loss.txt', 'w')
n_epoch = 100
batch_size = 64
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
list = []
list1 = []
list2 = []
list3 = []
for epoch in range(n_epoch):
    start_time = time.time()

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err;
        train_acc += ac;
        n_batch += 1;
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))
    list.append(train_loss / n_batch)
    list1.append(train_acc / n_batch)
    print(list)
    print(list1)
    g.write(str(epoch + 1) + ',' + str(train_acc / n_batch) + '\n')
    h.write(str(epoch + 1) + ',' + str(train_loss / n_batch) + '\n')
    # validation
    val_loss, val_acc, n_batch = 0, 0, 0

    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err;
        val_acc += ac;
        n_batch += 1

    print("   validation loss: %f" % (val_loss / n_batch))
    print("   validation acc: %f" % (val_acc / n_batch))
    list2.append(val_loss / n_batch)
    list3.append(val_acc / n_batch)
    print(list2)
    print(list3)
    f.write(str(epoch + 1) + ',' + str(val_acc / n_batch) + '\n')
    k.write(str(epoch + 1) + ',' + str(val_loss / n_batch) + '\n')
    if val_acc > max_acc:
        max_acc = val_acc
        saver.save(sess, 'C:/Users/36327/Desktop/face/faces.ckpt', global_step=epoch + 1)

f.close()
sess.close()

x = range(0,100,1)
a = np.array(x)
y = range(0,4,1)
b = np.array(list)
c = np.array(list2)
plt.figure(figsize=(180,20))
plt.plot(a, b, marker='o', mec='r', mfc='w',label='train loss')
plt.plot(a, c, marker='*', ms=10, label='vail loss')
plt.legend()
plt.xticks(x, x, rotation=1)
plt.margins(0)
plt.subplots_adjust(bottom=0.10)
plt.xlabel('Time') #X轴标签
plt.ylabel("Rate") #Y轴标签
pyplot.yticks(y, rotation=0.5)
plt.title("Loss") #标题
plt.savefig('C:/Users/36327/Desktop/face/f1.png')
plt.close()

d = np.array(list1)
e =np.array(list3)
plt.figure(figsize=(180,20))
plt.plot(a, d, marker='o', mec='r', mfc='w',label='train acc')
plt.plot(a, e, marker='*', ms=10,label='vail acc')
plt.legend()
plt.xticks(x, x, rotation=1)
plt.margins(0)
plt.subplots_adjust(bottom=0.10)
plt.xlabel('Time') #X轴标签
plt.ylabel("Rate") #Y轴标签
pyplot.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.title("Accuracy") #标题
plt.savefig('C:/Users/36327/Desktop/face/f2.png')
plt.close()