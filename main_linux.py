import platform
import cv2
import os
import numpy as np
from datetime import datetime
import random
import time
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
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
            gui: GUI实例
            image_size: 图像尺寸
        """
        self.output_dir = output_dir
        self.camera_index = camera_index
        self.cap = None
        self.gui = gui  # 存储GUI实例
        self.augmented_dir = None
        self.image_size = image_size
        self.original_dir = None
        self.train_dir = None
        self.is_collecting = False  # 控制是否正在采集
        self.preview_window_open = False  # 预览窗口状态
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
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logging.error("无法打开摄像头")
            raise IOError("无法打开摄像头")

        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size[1])

        # 摄像头相关信息
        logging.info(f"Camera Index: {self.camera_index}")
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

        # 更新GUI预览
        if self.gui:
            self.gui.update_preview(frame_resized)

        return frame_resized

    def save_image(self, image, directory, filename):
        """保存图像到指定目录"""
        path = os.path.join(directory, filename)
        cv2.imwrite(path, image)
        return path

    def apply_augmentations(self, image):
        """应用各种数据增强技术并返回带描述的图像列表"""
        augmented_images = []

        # 原始图像
        original_desc = "原始图像"
        augmented_images.append((original_desc, image))

        # 水平翻转
        flipped_h = cv2.flip(image, 1)
        augmented_images.append(("水平翻转", flipped_h))

        # 垂直翻转
        flipped_v = cv2.flip(image, 0)
        augmented_images.append(("垂直翻转", flipped_v))

        # 随机旋转 (0-360度)
        angle = random.uniform(0, 360)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append((f"旋转 {angle:.1f}度", rotated))

        # 随机裁剪 (保留50%-90%的图像)
        h, w = image.shape[:2]
        crop_h = int(h * random.uniform(0.5, 0.9))
        crop_w = int(w * random.uniform(0.5, 0.9))
        start_x = random.randint(0, w - crop_w)
        start_y = random.randint(0, h - crop_h)
        cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
        cropped = cv2.resize(cropped, self.image_size)  # 裁剪后调整回原始尺寸
        augmented_images.append((f"随机裁剪 ({crop_w}x{crop_h})", cropped))

        # 亮度调整
        brightness = random.uniform(0.7, 1.3)
        brightened = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        augmented_images.append((f"亮度调整 {brightness:.1f}x", brightened))

        # 通知GUI显示增强图像
        if self.gui:
            self.gui.show_augmented_images(augmented_images)

        return augmented_images

    def check_display_support(self):
        """检查是否支持显示窗口"""
        try:
            # 创建一个非常小的测试窗口（4x4像素）
            test_window = 'cv2_test_window'
            cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
            # 设置窗口大小为最小，避免在屏幕上显示明显窗口
            cv2.resizeWindow(test_window, 1, 1)
            # 移动到屏幕角落
            cv2.moveWindow(test_window, 0, 0)
            # 显示一个微小的黑色图像
            cv2.imshow(test_window, np.zeros((4, 4, 3), dtype=np.uint8))
            # 短暂等待并检查窗口是否显示
            cv2.waitKey(1)
            # 尝试多次销毁窗口以确保窗口关闭
            for _ in range(3):
                cv2.destroyWindow(test_window)
                cv2.waitKey(1)
            return True
        except Exception as e:
            logging.warning(f"显示支持检查失败: {str(e)}")
            return False
        finally:
            # 确保无论如何都尝试关闭窗口
            try:
                cv2.destroyAllWindows()
            except:
                pass

    def save_train_images(self, augmented_images, original_name):
        """将增强图像保存到训练目录，并命名为编号"""
        file_mapping = []

        # 获取当前 train 目录下的文件个数，以确定当前编号
        train_dir = self.train_dir
        current_files = os.listdir(train_dir)
        current_count = len([f for f in current_files if f.endswith('.jpg')])

        for idx, (aug_name, aug_img) in enumerate(augmented_images, 1):
            train_filename = f"{current_count + idx}.jpg"  # 按照个数命名
            self.save_image(aug_img, self.train_dir, train_filename)
            file_mapping.append((train_filename, f"{aug_name}_{original_name}"))

        return file_mapping

    def show_camera_feed(self):
        """显示摄像头实时画面"""
        if not self.show_window or not self.cap or not self.cap.isOpened():
            return

        self.preview_window_open = True
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

        while self.preview_window_open and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, self.image_size)
                cv2.imshow('Camera Feed', frame)

                # 更新GUI预览
                if self.gui:
                    self.gui.update_preview(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or not self.preview_window_open:
                break

        cv2.destroyWindow('Camera Feed')
        self.preview_window_open = False

    def stop_camera_feed(self):
        """停止摄像头实时画面"""
        self.preview_window_open = False
        if self.show_window:
            cv2.destroyAllWindows()

    def run_collection(self, num_images=10, delay=3, skip_frames=5):
        """
        运行图像采集过程

        参数:
            num_images: 要采集的图像数量
            delay: 采集间隔(秒)
            skip_frames: 跳过的帧数
        """
        self.initialize_camera()

        try:
            logging.info("图像采集开始...按 'q' 键退出")

            count = 0
            real_count = 0
            real_total = num_images * 6
            file_mapping_all = []  # 用于存储所有采集和增强图像的对应关系
            frame_counter = 0  # 帧计数器

            # 启动摄像头预览
            if self.show_window and not self.preview_window_open:
                preview_thread = threading.Thread(target=self.show_camera_feed)
                preview_thread.daemon = True
                preview_thread.start()

            while self.is_collecting and count < num_images:
                # 捕获图像
                frame = self.capture_image()

                frame_counter += 1

                # 跳过指定数量的帧
                if frame_counter % (skip_frames + 1) != 0:
                    continue

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
                    logging.info(f"当前采集图像数量: {real_count + 1}/{real_total} | 保存图像 ({aug_name}): {aug_path}")
                    real_count += 1

                # 将增强图像保存到训练目录，并记录文件对应关系
                file_mapping = self.save_train_images(augmented, filename)
                file_mapping_all.extend(file_mapping)

                count += 1

                # 每个图像采集结束后暂停
                time.sleep(delay)

            # 保存文件对应关系
            mapping_file = os.path.join(self.output_dir, 'train_file_mapping.txt')
            with open(mapping_file, 'a') as f:
                for train_filename, aug_name in file_mapping_all:
                    f.write(f"{train_filename} -> {aug_name}\n")
            logging.info(f"文件对应关系已保存: {mapping_file}")

            # 通知GUI采集完成
            if self.gui:
                self.gui.update_status_after_collection()

        except Exception as e:
            logging.error(f"采集过程中发生错误: {str(e)}")
        finally:
            # 释放资源
            self.stop_camera_feed()
            if self.cap is not None:
                self.cap.release()
            logging.info("图像采集完成")


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像采集器")
        self.root.geometry("1000x900")

        # 主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 控制面板框架
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板")
        self.control_frame.pack(fill=tk.X, pady=5)

        # 摄像头预览框架
        self.preview_frame = ttk.LabelFrame(self.main_frame, text="摄像头实时预览")
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.preview_canvas = tk.Canvas(self.preview_frame, bg='gray')
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_image = None

        # 增强图像展示框架
        self.augmented_frame = ttk.LabelFrame(self.main_frame, text="增强图像展示")
        self.augmented_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 创建一个画布和滚动条用于增强图像展示
        self.augmented_canvas = tk.Canvas(self.augmented_frame, bg='white')
        self.scrollbar = ttk.Scrollbar(self.augmented_frame, orient="vertical", command=self.augmented_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.augmented_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.augmented_canvas.configure(
                scrollregion=self.augmented_canvas.bbox("all")
            )
        )

        self.augmented_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.augmented_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.augmented_canvas.pack(side="left", fill="both", expand=True)

        # 控制面板内容
        self.output_dir = tk.StringVar(value="collected_images")

        # 输入采集数量
        ttk.Label(self.control_frame, text="采集数量(原图):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_images_entry = ttk.Entry(self.control_frame, validate="key",
                                          validatecommand=(self.root.register(self.validate_input), '%P'))
        self.num_images_entry.insert(0, "1")
        self.num_images_entry.grid(row=0, column=1, padx=5, pady=5)

        # 实际采集数量
        ttk.Label(self.control_frame, text="实际采集数量:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.actual_images_entry = ttk.Entry(self.control_frame, state='readonly')
        self.actual_images_entry.grid(row=1, column=1, padx=5, pady=5)
        self.update_actual_images_display(1)

        self.num_images_entry.bind('<KeyRelease>', self.on_num_images_change)

        # 路径显示
        self.path_label = ttk.Label(self.control_frame, text="保存路径: collected_images")
        self.path_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        # 按钮
        self.start_button = ttk.Button(self.control_frame, text="启动采集", command=self.toggle_collection)
        self.start_button.grid(row=3, column=0, padx=5, pady=10, sticky=tk.EW)

        self.select_button = ttk.Button(self.control_frame, text="选择路径", command=self.select_directory)
        self.select_button.grid(row=3, column=1, padx=5, pady=10, sticky=tk.EW)

        # 状态标签
        self.status_label = ttk.Label(self.control_frame, text="状态: 等待启动")
        self.status_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        # 图像采集器实例
        self.collector = ImageCollector(output_dir=self.output_dir.get(), gui=self)

        # 配置网格权重
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(1, weight=1)

    def show_augmented_images(self, augmented_images):
        """显示所有增强后的图像"""
        # 清除之前的图像
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # 设置每行显示的图像数量
        images_per_row = 3
        row = 0
        col = 0

        for desc, img in augmented_images:
            # 转换为RGB格式
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # 调整大小以适应显示
            display_size = (250, 250)
            img_pil.thumbnail(display_size, Image.LANCZOS)

            # 创建Tkinter图像对象
            img_tk = ImageTk.PhotoImage(img_pil)

            # 创建帧用于显示图像和描述
            frame = ttk.Frame(self.scrollable_frame)
            frame.grid(row=row, column=col, padx=5, pady=5)

            # 显示图像
            label_img = ttk.Label(frame, image=img_tk)
            label_img.image = img_tk  # 保持引用
            label_img.pack()

            # 显示描述
            label_desc = ttk.Label(frame, text=desc)
            label_desc.pack()

            # 更新行列位置
            col += 1
            if col >= images_per_row:
                col = 0
                row += 1

        # 更新画布滚动区域
        self.augmented_canvas.configure(scrollregion=self.augmented_canvas.bbox("all"))

    def update_preview(self, frame):
        """更新摄像头预览画面"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        if canvas_width > 0 and canvas_height > 0:
            img.thumbnail((canvas_width, canvas_height), Image.LANCZOS)

            x = (canvas_width - img.width) // 2
            y = (canvas_height - img.height) // 2

            self.preview_image = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(x, y, anchor=tk.NW, image=self.preview_image)

        self.root.update_idletasks()

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
        self.actual_images_entry.config(state='readonly')  # 更新后设置为不可编辑

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
            # 验证输入
            try:
                num_images = int(self.num_images_entry.get())
                if num_images <= 0:
                    raise ValueError

                # 设置跳帧数为5（可根据需要调整）
                skip_frames = 5

                self.collector.is_collecting = True
                self.status_label.config(text=f"状态: 采集中(每{skip_frames+1}帧采集一次)...")
                self.start_button.config(text="停止采集")

                # 启动采集线程
                collection_thread = threading.Thread(
                    target=self.collector.run_collection,
                    args=(num_images, 1, skip_frames)  # 传入skip_frames参数
                )
                collection_thread.daemon = True
                collection_thread.start()

            except ValueError:
                self.status_label.config(text="状态: 请输入有效的采集数量")
                return

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

    def on_close(self):
        """窗口关闭时清理资源"""
        self.collector.stop_camera_feed()
        if self.collector.cap is not None:
            self.collector.cap.release()
        self.root.destroy()


def start_gui():
    """启动GUI"""
    root = tk.Tk()
    app = GUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # 处理窗口关闭事件
    root.mainloop()


if __name__ == "__main__":
    start_gui()
