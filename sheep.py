import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Preprocessing (Image Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Set paths to your dataset
train_dir = r"G:\Couresware\深度学习\《深度学习》项目大作业\sheep\train"  # 训练集目录
test_dir = r"G:\Couresware\深度学习\《深度学习》项目大作业\sheep\test"  # 测试集目录

# Create datasets and dataloaders
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Custom ResNet Class
def basic_block(input_tensor, filters, strides=1, use_1x1conv=False):
    x = layers.Conv2D(filters, kernel_size=3, padding='same', strides=strides)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if use_1x1conv:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=strides)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def big_block(input_tensor, filters, num_blocks, first_bigblock=False):
    for i in range(num_blocks):
        if first_bigblock and i == 0:
            x = basic_block(input_tensor, filters, use_1x1conv=True)
        elif i == 0:
            x = basic_block(input_tensor, filters, strides=2, use_1x1conv=True)
        else:
            x = basic_block(x, filters)
    return x

def Resnet():
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = big_block(x, 64, 2, first_bigblock=True)
    x = big_block(x, 128, 2)
    x = big_block(x, 256, 2)
    x = big_block(x, 512, 2)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(2, activation='softmax')(x)  # Output 2 classes for binary classification

    model = models.Model(inputs, outputs)
    return model

# Model, Loss Function, Optimizer
model = Resnet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training function
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Evaluate function
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy:.4f}')

# Save the model
model.save('sheep_disease_model.h5')



