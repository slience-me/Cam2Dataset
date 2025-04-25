import platform
import cv2
import os
import numpy as np
from datetime import datetime
import random
import time
import tkinter as tk
from tkinter import filedialog
import threading
import logging
from colorlog import ColoredFormatter

# 设置日志格式和颜色
log_format = "%(asctime)s - %(levelname)s - %(message)s"
formatter = ColoredFormatter(
    "%(log_color)s" + log_format,  # 添加颜色
    log_colors={
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

# 创建日志处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)
logging.getLogger().setLevel(logging.INFO)  # 设置日志级别为INFO


class ImageCollector:
    def __init__(self, output_dir='collected_images', camera_index=0, image_size=(640, 640), gui=None):
        """
        初始化图像采集器

        参数:
            output_dir: 输出目录
            camera_index: 摄像头索引 (0通常是默认摄像头)
        """
        self.output_dir = output_dir
        self.camera_index = camera_index
        self.cap = None
        self.gui = gui  # 存储GUI实例
        self.augmented_dir = None
        self.image_size = image_size
        self.original_dir = None
        self.train_dir = None
        self.is_collecting = False  # 添加标志，控制是否正在采集
        self.setup_directories()
        self.show_window = self.check_display_support()

        self.print_system_info()  # 打印系统信息

    def print_system_info(self):
        """打印系统相关信息"""
        logging.info("=" * 50)
        logging.info(f"{'SYSTEM INFORMATION':^50}")
        logging.info("=" * 50)

        # 系统平台信息
        logging.info(f"Platform: {platform.system()} {platform.release()} {platform.version()}")
        logging.info(f"Architecture: {platform.architecture()[0]}")
        logging.info(f"Python Version: {platform.python_version()}")

        # 输出目录信息
        logging.info(f"Output Directory: {self.output_dir}")
        logging.info(f"  - Original Images: {self.original_dir}")
        logging.info(f"  - Augmented Images: {self.augmented_dir}")
        logging.info(f"  - Training Images: {self.train_dir}")

        logging.info("=" * 50 + "\n")

    def setup_directories(self):
        """创建必要的输出目录"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 创建子目录用于存储不同类型的图像
        self.original_dir = os.path.join(self.output_dir, 'original')
        self.augmented_dir = os.path.join(self.output_dir, 'augmented')
        self.train_dir = os.path.join(self.output_dir, os.path.join('images', 'train'))

        for dir_path in [self.original_dir, self.augmented_dir, self.train_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def initialize_camera(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logging.error("无法打开摄像头")
            raise IOError("无法打开摄像头")

        # 摄像头相关信息
        logging.info(f"Camera Index: {self.camera_index}")
        logging.info(f"Supported Camera Resolutions:")

        max_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        max_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logging.info(f"  - Max Resolution: {int(max_width)}x{int(max_height)}")

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logging.info(f"  - Current Resolution: {int(width)}x{int(height)}")

    def generate_filename(self, prefix='img'):
        """生成唯一的文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{prefix}_{timestamp}.jpg"

    def capture_image(self):
        """从摄像头捕获一帧图像"""
        ret, frame = self.cap.read()
        if not ret:
            logging.error("无法从摄像头读取图像")
            raise RuntimeError("无法从摄像头读取图像")
        # 调整图像尺寸
        frame_resized = cv2.resize(frame, self.image_size)
        return frame_resized

    def save_image(self, image, directory, filename):
        """保存图像到指定目录"""
        path = os.path.join(directory, filename)
        cv2.imwrite(path, image)
        return path

    def apply_augmentations(self, image):
        """应用各种数据增强技术"""
        augmented_images = []

        # 原始图像
        augmented_images.append(('original', image))

        # 水平翻转
        flipped_h = cv2.flip(image, 1)
        augmented_images.append(('flipped_h', flipped_h))

        # 垂直翻转
        flipped_v = cv2.flip(image, 0)
        augmented_images.append(('flipped_v', flipped_v))

        # 随机旋转 (0-360度)
        angle = random.uniform(0, 360)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(('rotated', rotated))

        # 随机裁剪 (保留50%-90%的图像)
        h, w = image.shape[:2]
        crop_h = int(h * random.uniform(0.5, 0.9))
        crop_w = int(w * random.uniform(0.5, 0.9))
        start_x = random.randint(0, w - crop_w)
        start_y = random.randint(0, h - crop_h)
        cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
        augmented_images.append(('cropped', cropped))

        # 亮度调整
        brightness = random.uniform(0.7, 1.3)
        brightened = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        augmented_images.append(('brightness', brightened))

        return augmented_images

    def check_display_support(self):
        """检查是否支持显示窗口"""
        try:
            cv2.imshow('test', np.zeros((100, 100, 3), dtype=np.uint8))
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            return True
        except:
            return False

    def save_train_images(self, augmented_images, orignal_name):
        """将增强图像保存到训练目录，并命名为编号"""
        file_mapping = []

        # 获取当前 train 目录下的文件个数，以确定当前编号
        train_dir = self.train_dir
        current_files = os.listdir(train_dir)
        current_count = len([f for f in current_files if f.endswith('.jpg')])  # 只计数以 .jpg 结尾的文件

        for idx, (aug_name, aug_img) in enumerate(augmented_images, 1):
            train_filename = f"{current_count + idx}.jpg"  # 按照个数命名
            self.save_image(aug_img, self.train_dir, train_filename)
            file_mapping.append((train_filename, f"{aug_name}_{orignal_name}"))

        return file_mapping

    def run_collection(self, num_images=10, delay=1):
        """
        运行图像采集过程

        参数:
            num_images: 要采集的图像数量
            delay: 采集间隔(秒)
        """
        self.initialize_camera()

        try:
            logging.info("图像采集开始...按 'q' 键退出")

            count = 0
            real_count = 0
            real_total = num_images * 6
            file_mapping_all = []  # 用于存储所有采集和增强图像的对应关系

            while self.is_collecting and count < num_images:
                # 捕获图像
                frame = self.capture_image()

                # 显示图像（仅当支持时）
                if self.show_window:
                    cv2.imshow('Camera Feed', frame)

                # 生成唯一文件名
                filename = self.generate_filename()

                # 保存原始图像
                self.save_image(frame, self.original_dir, filename)

                # 应用数据增强并保存
                augmented = self.apply_augmentations(frame)
                for aug_name, aug_img in augmented:
                    aug_filename = f"{aug_name}_{filename}"
                    self.save_image(aug_img, self.augmented_dir, aug_filename)
                    aug_path = self.save_image(aug_img, self.augmented_dir, aug_filename)
                    logging.info(f"当前采集图像数量: {real_count+1}/{real_total} | 保存图像 ({aug_name}): {aug_path}")
                    real_count += 1

                # 将增强图像保存到训练目录，并记录文件对应关系
                file_mapping = self.save_train_images(augmented, filename)
                file_mapping_all.extend(file_mapping)

                count += 1

                # 每个图像采集结束后暂停
                if self.show_window:
                    key = cv2.waitKey(delay * 1000)
                    if key == ord('q'):
                        break
                else:
                    time.sleep(delay)  # 使用time.sleep代替cv2.waitKey

            # 保存文件对应关系
            mapping_file = os.path.join(self.output_dir, 'train_file_mapping.txt')
            with open(mapping_file, 'a') as f:
                for train_filename, aug_name in file_mapping_all:
                    f.write(f"{train_filename} -> {aug_name}\n")
            logging.info(f"文件对应关系已保存: {mapping_file}")

            # 通知GUI采集完成
            if self.gui:
                self.gui.update_status_after_collection()

        finally:
            # 释放资源
            self.cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            logging.info("图像采集完成")


class GUI:
    def __init__(self, root):
        self.info_label = None
        self.root = root
        self.root.title("图像采集器")
        self.root.geometry("500x400")

        # 路径选择
        self.output_dir = tk.StringVar(value="collected_images")

        # 输入采集的图片数量(n*6)
        self.num_images_label = tk.Label(self.root, text="输入采集数量: (原图个数)")
        self.num_images_label.pack(pady=5)

        # 使用正则表达式验证输入内容（只允许数字）
        vcmd = (self.root.register(self.validate_input), '%P')  # 注册验证函数
        self.num_images_entry = tk.Entry(self.root, validate="key", validatecommand=vcmd)
        self.num_images_entry.insert(0, "1")  # 默认值为 1
        self.num_images_entry.pack(pady=5)

        # 显示实际采集的图片个数
        self.actual_images_label = tk.Label(self.root, text="实际采集图像数量: (n*6)")
        self.actual_images_label.pack(pady=5)

        # 显示实际采集图像数量的输入框 (disabled 不可编辑)
        self.actual_images_entry = tk.Entry(self.root, state='disabled')  # 设置为不可编辑
        self.actual_images_entry.pack(pady=5)
        self.update_actual_images_display(1)  # 初始值为 1 * 6 = 6
        # 实时更新实际采集数量
        self.num_images_entry.bind('<KeyRelease>', self.on_num_images_change)  # 绑定事件

        # 显示默认路径
        self.default_path_label = tk.Label(self.root, text="默认采集存储路径为：./collected_images")
        self.default_path_label.pack(pady=10)

        # 启动/停止按钮
        self.start_button = tk.Button(self.root, text="启动采集", command=self.toggle_collection)
        self.start_button.pack(pady=10)

        # 路径选择按钮
        self.select_button = tk.Button(self.root, text="选择保存路径", command=self.select_directory)
        self.select_button.pack(pady=10)

        # 状态标签
        self.status_label = tk.Label(self.root, text="状态: 等待启动")
        self.status_label.pack(pady=10)

        # 图像采集器实例(默认路径collected_images)
        self.collector = ImageCollector(output_dir=self.output_dir.get(), gui=self)

    def on_num_images_change(self, event):
        """实时更新实际采集图像数量"""
        try:
            num = int(self.num_images_entry.get())  # 获取输入的数量
            self.update_actual_images_display(num)  # 更新实际图像数量显示
        except ValueError:
            self.update_actual_images_display(0)  # 如果输入不是数字，显示为0

    def update_actual_images_display(self, num):
        """根据输入的数量更新显示的实际采集图像数量"""
        actual_images = num * 6  # 1输入得到6张图像
        self.actual_images_entry.config(state='normal')  # 先设置为可编辑状态，更新内容后再设置为不可编辑
        self.actual_images_entry.delete(0, tk.END)  # 清空当前内容
        self.actual_images_entry.insert(0, str(actual_images))  # 插入新的值
        self.actual_images_entry.config(state='disabled')  # 更新后设置为不可编辑


    def validate_input(self, input_value):
        """验证输入是否为纯数字"""
        if input_value == "" or input_value.isdigit():  # 允许空白或纯数字
            return True
        else:
            return False

    def toggle_collection(self):
        """切换采集状态"""
        if self.collector.is_collecting:
            self.collector.is_collecting = False
            self.status_label.config(text="状态: 停止采集")
            self.start_button.config(text="启动采集")
        else:
            self.collector.is_collecting = True
            self.status_label.config(text="状态: 采集中...")
            self.start_button.config(text="停止采集")

            # 启动采集线程
            collection_thread = threading.Thread(target=self.collector.run_collection,
                                                 args=(int(self.num_images_entry.get()), 1))
            collection_thread.daemon = True  # 让线程随主程序结束
            collection_thread.start()

    def select_directory(self):
        """选择保存目录"""
        selected_dir = filedialog.askdirectory(initialdir=self.output_dir.get())
        if selected_dir:
            self.output_dir.set(selected_dir)
            self.status_label.config(text=f"修改保存路径为: {selected_dir}")
            self.collector.output_dir = selected_dir  # 更新路径
            logging.critical(f"修改保存路径为: {selected_dir}")
            self.collector.setup_directories()

    def update_status_after_collection(self):
        """更新采集完成后的状态"""
        self.status_label.config(text="状态: 采集完成")
        self.start_button.config(text="重新启动采集")


def start_gui():
    """启动GUI"""
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    start_gui()
