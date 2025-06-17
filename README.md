# ResNet_sheep_disease_detect
基于ResNet的病羊图片分类的课程设计

## 基于tensorflow2.6的病羊识别系统

### 项目简介
这个项目是基于tensorflow-gpu2.6版本

使用了CNN卷积神经网络的resnet18模型。

前期我尝试使用了resnet101、50、34、18网络，发现由于我手动获取到的病羊的数据集数量比较小（正常羊356张、不正常羊183张），非常容易使训练出来的模型梯度爆炸和过拟合，尝试后发现resnet18的效果最好，一开始的准确率只能达到0.7-0.85，所以我决定在resnet18的基础上再简化模型。

resnet18包含了八个残差块，分成四个阶段，每个阶段有两个残差块，经过我的简化后，只留下了四个大块，每个大块中有两个基础残差块。这些改动意味着他的表现能力不如原始的resnet18，但是在小数据集上足够了。

### 项目流程
1. **数据准备**
   - 使用 `Image_preprocessing.py` 进行数据预处理。
   - 由于没有公开的病羊识别数据集，我们自制了数据集。通过各大图片网站、期刊、论文、国内外的病羊诊治网站、交流论坛等获取图片数据，然后使用 OpenCV 工具对图片进行裁剪、缩放得到 224x224 大小的数据集。

2. **构建模型、训练模型、评估模型**
   - `sheep.py` 是原始 ResNet18 网络模型，`sheep2.py` 是经过改进后的网络模型。
   - 我们将两个模型保存为：`sheep_disease_model.h5` 和 `sheep2_disease_model.h5`。

3. **使用模型**
   - 使用 `windows.py` 实现可视化操作界面。
   - 通过上传需要识别的羊的照片，进行预测。

4. **其他**
   - `images` 文件夹用于存放 UI 界面的图片。

### 新手教程

#### 环境准备
- 确保安装了 Python 3.x 和 TensorFlow 2.6。
- 安装所需的 Python 包：
  ```bash
  pip install -r requirements.txt
  ```

#### 数据准备
- 将病羊图片数据集放入 `sheep_dataset/train` 和 `sheep_dataset/test` 目录下。
- 运行 `Image_preprocessing.py` 进行数据预处理。

#### 模型训练
- 运行 `sheep_train.py` 进行模型训练。
- 训练完成后，模型将自动保存为 `sheep2_disease_model.h5`。

#### 模型预测
- 运行 `sheep_windows.py` 启动可视化界面。
- 上传羊的照片进行识别，查看识别结果。

#### 注意事项
- 确保数据集的图片格式正确。
- 如果遇到任何问题，请检查日志文件以获取详细信息。
