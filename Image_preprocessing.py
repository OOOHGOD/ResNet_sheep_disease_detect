import os
import cv2

# 定义输入和输出的文件夹路径
train_dir = "sheep/train"
test_dir = "sheep/test"
output_size = (224, 224)

# 定义需要处理的子文件夹
categories = ['Normal sheep', 'Sick sheep']

def resize_and_rename_images_in_folder(folder_path):
    # 遍历文件夹中的所有图片
    for category in categories:
        category_path = os.path.join(folder_path, category)
        if not os.path.exists(category_path):
            print(f"目录 {category_path} 不存在，跳过")
            continue

        # 处理每个类别中的图片
        for idx, image_name in enumerate(os.listdir(category_path)):
            image_path = os.path.join(category_path, image_name)

            # 确保是图片文件
            if image_name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
                try:
                    # 读取图像并调整大小
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"无法读取图片 {image_path}，跳过")
                        continue

                    img_resized = cv2.resize(img, output_size)

                    new_image_name = f"{category.replace(' ', '_')}{idx + 1}.jpg"
                    new_image_path = os.path.join(category_path, new_image_name)

                    # 检查是否已经存在相同名称的文件
                    if os.path.exists(new_image_path):
                        print(f"文件 {new_image_path} 已存在，跳过")
                        continue

                    # 保存调整大小并重新命名后的图像
                    cv2.imwrite(new_image_path, img_resized)
                    print(f"已处理并重命名: {new_image_path}")

                    # 删除原图
                    os.remove(image_path)

                except Exception as e:
                    print(f"处理图片 {image_path} 时出错: {e}")

# 处理训练集和测试集
resize_and_rename_images_in_folder(train_dir)
resize_and_rename_images_in_folder(test_dir)