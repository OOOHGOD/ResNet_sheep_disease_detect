import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil


# 定义主窗口类，继承自 QTabWidget，用于创建多标签界面
class MainWindow(QTabWidget):
    # 初始化方法，设置窗口图标、标题、模型加载等
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/logo.png'))  # 设置窗口图标
        self.setWindowTitle('病羊识别系统')  # 设置窗口标题

        # 模型初始化
        self.model = tf.keras.models.load_model("sheep2_disease_model.h5")  # 加载预训练的Keras模型
        self.to_predict_name = 'images/about.png'  # 设置初始图片路径
        self.class_names = ['正常羊', '高风险羊']  # 定义类别名称

        self.resize(900, 700)  # 设置窗口大小
        self.initUI()  # 初始化界面

    # 界面初始化方法，设置布局和组件
    def initUI(self):
        main_widget = QWidget()  # 创建主页面小部件
        main_layout = QHBoxLayout()  # 使用水平布局
        font = QFont('楷体', 15)  # 设置字体样式

        # 左侧区域：显示图片
        left_widget = QWidget()
        left_layout = QVBoxLayout()  # 使用垂直布局
        img_title = QLabel("传入图片预览")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()

        # 加载初始图片并检查有效性
        img_init = cv2.imread(self.to_predict_name)
        if img_init is None:
            QMessageBox.critical(self, "错误", f"无法加载图片 {self.to_predict_name}. 请检查路径或文件完整性.")
            sys.exit(1)

        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.jpg", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.jpg', img_init)
        self.img_label.setPixmap(QPixmap("images/show.jpg"))

        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        # 右侧区域：按钮和结果显示
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 上传图片 ")
        btn_change.clicked.connect(self.change_img)  # 绑定上传图片按钮点击事件
        btn_change.setFont(font)
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)  # 绑定开始识别按钮点击事件
        label_result = QLabel(' 病羊识别 ')
        self.result = QLabel("等待识别")
        label_result.setFont(QFont('楷体', 16))
        self.result.setFont(QFont('楷体', 24))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)

        # 关于页面：介绍信息
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用病羊识别系统')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_title_1 = QLabel('')
        about_title_1.setFont(QFont('楷体', 18))
        about_title_1.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/show.jpg'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel("")
        label_super.setFont(QFont('楷体', 12))
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addWidget(about_title_1)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        # 添加主页面和关于页面到标签栏
        self.addTab(main_widget, '主页')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/关于.png'))

    # 上传并显示图片方法
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'Image files(*.jpg *.png *jpeg)')
        img_name = openfile_name[0]
        if img_name == '':
            return

        # 将图片复制到目标目录，并更新要预测的图片路径
        target_image_name = "images/show." + img_name.split(".")[-1]
        shutil.copy(img_name, target_image_name)
        self.to_predict_name = target_image_name

        # 加载并处理图片
        img_init = cv2.imread(self.to_predict_name)
        if img_init is None:
            QMessageBox.critical(self, "错误", f"无法加载图片 {self.to_predict_name}. 请检查路径或文件完整性.")
            return

        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.jpg", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.jpg', img_init)
        self.img_label.setPixmap(QPixmap("images/show.jpg"))
        self.result.setText("等待识别")

    # 预测图片方法
    def predict_img(self):
        try:
            tf.keras.backend.clear_session()  # 清理模型状态
            img = Image.open('images/target.jpg').convert('RGB')
            img = img.resize((224, 224))  # 确保尺寸为 224x224
            img = np.asarray(img)  # 转换为 numpy 数组
            img = img / 255.0  # 归一化
            img = np.asarray(img, dtype=np.float32)
            outputs = self.model.predict(img.reshape(1, 224, 224, 3))
            probability = outputs[0][0]
            threshold = 0.7
            result_index = 1 if probability >= threshold else 0
            self.result.setText(self.class_names[result_index])
            print(f"预测概率: {probability}, 类别: {self.class_names[result_index]}")  # 调试用
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测失败: {str(e)}")

    # 界面关闭事件，询问用户是否关闭
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建应用程序对象
    x = MainWindow()  # 实例化主窗口
    x.show()  # 显示主窗口
    sys.exit(app.exec_())  # 进入应用程序的主循环