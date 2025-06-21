import dearpygui.dearpygui as dpg
import numpy as np
from ui.boxes.BaseBox import BaseBox
from ui.components.Canvas2D import Canvas2D
from ui.boxes.SAMMark.utils.cv_utils import *
from ui.boxes.SAMMark.sam2marker.sam2marker import SAM2Image
SIZE = (800, 600)
IMG_BASE_PATH = "mark_image"
DATASETS_NAME = "test"

class MarkBox(BaseBox):
    # only = True 表示只能创建一个实例
    only = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.datasets_path = f"datasets/{DATASETS_NAME}"
        create_datasets_dir(self.datasets_path)
        self.img = None
        self.input = None
        self.data = self.data or np.zeros(3)
        self.file_list_item = None
        self.select_image_name = None
        self.image_list = []
        self.draw_layer = None
        self.image_layer = None
        self.labels = []
        self.sam2 = SAM2Image()
        self.mark_points = []
        self.mark_labels = []

    def load_image(self):
        dpg.delete_item(self.file_list_item, children_only=True)
        self.image_list = read_file(IMG_BASE_PATH)
        for image in self.image_list:
            dpg.add_selectable(
                label=f"{image}",
                tag=f"{image}",
                parent=self.file_list_item,
                callback=self.select_image,
            )

    def rename_image(self):
        rename_img()
        self.load_image()

    def select_image(self, sender, app_data, user_data):
        print(sender)
        if self.select_image_name is not None and self.select_image_name != sender:
            dpg.set_value(self.select_image_name, False)
        self.select_image_name = sender
        self.sam2.load_image(
            read_img(self.select_image_name, SIZE, IMG_BASE_PATH, "RGB")
        )

    def next_image(self, sender, app_data, user_data):
        if not self.image_list:
            return
        # print(dpg.is_item_focused(self.file_list_item))
        if dpg.is_item_hovered(self.file_list_item):
            if app_data < 0:
                current_index = self.image_list.index(self.select_image_name)
                next_index = (current_index + 1) % len(self.image_list)
                next_image = self.image_list[next_index]
                dpg.set_value(next_image, True)
                self.select_image(next_image, None, None)
            else:
                current_index = self.image_list.index(self.select_image_name)
                next_index = (current_index - 1) % len(self.image_list)
                next_image = self.image_list[next_index]
                dpg.set_value(next_image, True)
                self.select_image(next_image, None, None)
            dpg.set_value(next_image, True)
            self.select_image(next_image, None, None)

    def add_positive_points(self, sender, app_data, user_data):
        if dpg.is_item_hovered(self.canvas.drawlist_tag):
            mouse_pos = dpg.get_drawing_mouse_pos()
            mouse_pos_transform = self.canvas.pos_apply_transform(mouse_pos)
            self.mark_points.append(mouse_pos_transform)
            self.mark_labels.append(1)
            dpg.draw_circle(
                mouse_pos_transform,
                radius=5,
                fill=(255, 0, 0, 255),
                parent=self.draw_layer,
            )

    def add_negative_points(self, sender, app_data, user_data):
        if dpg.is_item_hovered(self.canvas.drawlist_tag):
            mouse_pos = dpg.get_drawing_mouse_pos()
            mouse_pos_transform = self.canvas.pos_apply_transform(mouse_pos)
            self.mark_points.append(mouse_pos_transform)
            self.mark_labels.append(0)
            dpg.draw_circle(
                mouse_pos_transform,
                radius=5,
                fill=(0, 255, 0, 255),
                parent=self.draw_layer,
            )
    
    def normalize_rect(self,rect, img_height, img_width):
        """
        将矩形框坐标归一化
        
        Args:
            rect (tuple): 原始矩形框坐标 (x, y, w, h)
            img_height (int): 图像高度
            img_width (int): 图像宽度
        
        Returns:
            tuple: 归一化后的中心坐标和宽高 (x_center, y_center, norm_width, norm_height)
        """
        x, y, w, h = rect
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        return x_center, y_center, norm_width, norm_height

    def save_labels(self,rects, label_name, img_height, img_width, label_file_path, overwrite=False):
        """
        保存标签文件（支持覆盖或追加）
        
        Args:
            rects (list): 矩形框列表
            label_name (str): 标签名称
            img_height (int): 图像高度
            img_width (int): 图像宽度
            label_file_path (str): 标签文件路径
            overwrite (bool, optional): 是否覆盖现有文件. Defaults to False.
        """
        # 选择文件打开模式
        mode = 'w' if overwrite else 'a'
        
        with open(label_file_path, mode) as f:
            for rect in rects:
                # 归一化坐标
                x_center, y_center, norm_width, norm_height = self.normalize_rect(rect, img_height, img_width)
                
                # 写入标签文件 (YOLO格式: class_id x_center y_center width height)
                f.write(f"{label_name} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                
                print(f"Normalized Rectangle: x_center={x_center}, y_center={y_center}, "
                    f"width={norm_width}, height={norm_height}")


    def mark_data(self):
        dpg.delete_item(self.draw_layer, children_only=True)
        if self.select_image_name is None:
            return
        if not self.mark_labels or not self.mark_points:
            print("No points marked.")
            return
        
        # 获取图像尺寸
        img_height, img_width = self.img.shape[:2]
        
        masks, scores, logits = self.sam2.add_point(self.mark_points, self.mark_labels)
        rects, filtered_masks, filtered_scores = self.sam2.process_sam2_prediction(
            masks, scores, logits
        )
        
        label_name = dpg.get_value(self.input)
        print(f"Label Name: {label_name}")
        
        if not rects:
            print("No valid rectangles found.")
            return
        
        # 创建标签文件夹（如果不存在）
        os.makedirs(os.path.join(self.datasets_path, "labels", "all"), exist_ok=True)
        os.makedirs(os.path.join(self.datasets_path, "images", "all"), exist_ok=True)
        
        # 标签文件路径
        label_file_path = os.path.join(self.datasets_path, "labels", "all", 
                                    os.path.splitext(self.select_image_name)[0] + ".txt")
        
        # 保存标签
        self.save_labels(rects, label_name, img_height, img_width, label_file_path)
        
        # 绘制矩形框
        for rect in rects:
            x, y, w, h = rect
            dpg.draw_rectangle(
                (x, y), (x + w, y + h), color=(255, 0, 0, 255), parent=self.draw_layer, thickness=3
            )
        
        # 保存原始图像
        cv2.imwrite(
            os.path.join(self.datasets_path, "images", "all", self.select_image_name),
            cv2.cvtColor(self.img, cv2.COLOR_RGBA2BGR),
        )
        
        # 重置标记点
        self.mark_points = []
        self.mark_labels = []

    def create_handler(self):
        with dpg.handler_registry() as global_hander:
            dpg.add_key_release_handler(dpg.mvKey_Spacebar, callback=self.mark_data)
            dpg.add_mouse_wheel_handler(callback=self.next_image)
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Left,
                callback=self.add_positive_points,
            )
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right,
                callback=self.add_negative_points,
            )

    def create(self):
        with dpg.group(horizontal=True, parent=self.tag) as group:
            self.input = dpg.add_input_text(label="Label", default_value="0", width=120)
            dpg.add_button(label="Load Image", callback=self.load_image)
            dpg.add_button(label="Rename Image", callback=self.rename_image)
            dpg.add_combo(
                items=["Auto", "Manual"],
                width=100,
                label="Mark Mode",
                default_value="Auto",
            )

            dpg.add_checkbox(label="Auto Next Image", default_value=False)

        with dpg.group(horizontal=True, parent=self.tag) as group:
            with dpg.child_window(height=-1, width=135) as self.file_list_item:
                pass
            self.canvas = Canvas2D(group)
            self.image_layer = self.canvas.add_layer()
            self.draw_layer = self.canvas.add_layer()
        self.texture_id = self.canvas.texture_register(SIZE)

        with self.canvas.draw(self.image_layer) as draw_tag:
            dpg.draw_image(self.texture_id, (0, 0), SIZE)
        self.create_handler()

    def update(self):
        if self.select_image_name is None:
            return
        self.img = read_img(self.select_image_name, SIZE, IMG_BASE_PATH)
        self.canvas.texture_update(self.texture_id, self.img)
        # self.data = np.array(dpg.get_value(self.input))


    def destroy(self):
        super().destroy()  # 真正销毁Box
