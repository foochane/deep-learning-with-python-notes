
# coding: utf-8

# ## 1 加载数据
# IMDB数据集，内置于Keras库中，它包含来自互联网电影数据库（IMDB）的 50 000 条严重两极分
# 化的评论。数据集被分为用于训练的 25 000 条评论与用于测试的 25 000 条评论，训练集和测试
# 集都包含 50% 的正面评论和 50% 的负面评论

# In[1]:

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from keras.datasets import imdb


# In[2]:


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# In[ ]:


train_data.shape


# In[ ]:


train_data


# In[ ]:


train_labels.shape


# In[ ]:


train_labels


# In[ ]:


# 由于num_words = 10000，限定为前10000个最长见的单词
# 所以最大值为9999
max([max(sequence) for sequence in train_data])


# In[ ]:


# 将某条评论迅速解码为英文单词

word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value,key) for (key,value)  in word_index.items()])
decoded_review = ' '.join(
 [reverse_word_index.get(i - 3, '?') for i in train_data[0]]) 


# In[ ]:


word_index


# In[ ]:


len(word_index)


# In[ ]:


decoded_review


# ## 2 准备数据

# In[ ]:


# 将整数序列编码为二进制矩阵 ,one-hot编码
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data) 


# In[ ]:


x_train[0]


# In[ ]:


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[ ]:


y_train


# ## 3 构建网络

# In[ ]:


# 定义模型
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


#  编译模型
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


# 配置优化器
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
   loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


# 定义损失和指标
from keras import losses
from keras import metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
 loss=losses.binary_crossentropy,
 metrics=[metrics.binary_accuracy])


# ## 4 验证

# In[ ]:


# 留出10000个样本作为验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[ ]:


#训练模型
model.compile(optimizer='rmsprop',
 loss='binary_crossentropy',
 metrics=['acc'])
history = model.fit(partial_x_train,
 partial_y_train,
 epochs=20,
 batch_size=512,
 validation_data=(x_val, y_val))


# In[ ]:


history_dict = history.history


# In[ ]:


history_dict.keys()


# In[ ]:


#绘制训练损失和验证损失
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


# 绘制训练精度和验证精度
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

