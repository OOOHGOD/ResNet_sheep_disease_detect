import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 可视化训练过程
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


# 数据预处理（图像增强）
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 像素值归一化到 [0, 1]
    horizontal_flip=True,  # 随机水平翻转
    vertical_flip=True,  # 随机垂直翻转
    rotation_range=20,  # 随机旋转角度范围
    brightness_range=[0.8, 1.2],  # 随机调整亮度
    shear_range=0.2,  # 随机错切变换
    zoom_range=0.2,  # 随机缩放
    validation_split=0.2  # 将部分数据划分为验证集
)

test_datagen = ImageDataGenerator(rescale=1./255)  # 测试集仅进行归一化处理

# 设置数据集路径
train_dir = "sheep\\train\\"  # 训练集目录
test_dir = "sheep\\test\\"  # 测试集目录

# 创建训练集和验证集的数据生成器
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # 调整图像大小
    batch_size=32,  # 批量大小
    class_mode='binary',  # 二分类模式
    subset='training'  # 训练集
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # 验证集
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # 测试集
)

# 基础残差块
def basic_block(input_tensor, filters, strides=1, use_1x1conv=False):
    x = layers.Conv2D(filters, kernel_size=3, padding='same', strides=strides)(input_tensor)
    x = layers.BatchNormalization()(x)  # 批量归一化
    x = layers.ReLU()(x)  # 激活函数
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 是否使用 1x1 卷积调整维度
    if use_1x1conv:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=strides)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    x = layers.Add()([x, shortcut])  # 残差连接
    x = layers.ReLU()(x)
    return x


# 大块残差模块
def big_block(input_tensor, filters, num_blocks, first_bigblock=False):
    for i in range(num_blocks):
        if first_bigblock and i == 0:
            x = basic_block(input_tensor, filters, use_1x1conv=True)  # 第一个大块使用 1x1 卷积
        elif i == 0:
            x = basic_block(input_tensor, filters, strides=2, use_1x1conv=True)  # 降采样
        else:
            x = basic_block(x, filters)
    return x


# 定义 ResNet 模型
def Resnet():
    inputs = layers.Input(shape=(224, 224, 3))  # 输入层
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)  # 初始卷积层
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)  # 最大池化

    x = big_block(x, 64, 2, first_bigblock=True)  # 第一个大块
    x = big_block(x, 128, 2)  # 第二个大块
    x = big_block(x, 256, 2)  # 第三个大块
    x = big_block(x, 512, 2)  # 第四个大块

    x = layers.GlobalAveragePooling2D()(x)  # 全局平均池化
    outputs = layers.Dense(1, activation='sigmoid')(x)  # 输出层，二分类

    model = models.Model(inputs, outputs)
    return model


# 创建模型
model = Resnet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 优化器
    loss='binary_crossentropy',  # 损失函数
    metrics=['accuracy']  # 评估指标
)

# 学习率调度回调
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

# TensorBoard 回调，用于可视化计算图
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# 模型训练
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=1,  # 训练轮数
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[lr_schedule, tensorboard_callback]  # 回调函数
)


# 模型评估
loss, accuracy = model.evaluate(test_generator)
print(f'测试集准确率: {accuracy:.4f}')


# 保存模型
model.save('sheep2_disease_model.h5')  # 保存为 H5 文件格式


# 加载黑体字体文件
font_path = "path/to/simhei.ttf"  # 替换为你的黑体字体文件路径
font_size = 40
font = ImageFont.truetype(font_path, font_size)


plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training set accuracy')
plt.plot(history.history['val_accuracy'], label='Verification set accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and verification accuracy')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training set loss')
plt.plot(history.history['val_loss'], label='Verification set loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and validation loss')
plt.show()


# 获取测试集的真实标签和预测概率
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator).ravel()


# 将预测概率转换为类别标签
y_pred = (y_pred_prob > 0.5).astype(int)


# 计算混淆矩阵
conf_matrix = tf.math.confusion_matrix(y_true, y_pred).numpy()


# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()