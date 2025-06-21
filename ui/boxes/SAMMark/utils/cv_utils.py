import cv2
import os

def create_datasets_dir(datasets_path):
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)
        os.makedirs(f"{datasets_path}/images")
        os.makedirs(f"{datasets_path}/images/train")
        os.makedirs(f"{datasets_path}/images/val")
        os.makedirs(f"{datasets_path}/images/all")
        os.makedirs(f"{datasets_path}/labels")
        os.makedirs(f"{datasets_path}/labels/train")
        os.makedirs(f"{datasets_path}/labels/val")
        os.makedirs(f"{datasets_path}/labels/all")


def read_img(filename,size, base_path="mark_image",format="RGBA"):
    base_path = base_path + "/" + filename

    img = cv2.imread(base_path)
    img = cv2.resize(img, size)
    if format == "RGBA":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    elif format == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def read_file(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
            
    images = sorted(os.listdir(directory))
    return images

def rename_img(output_dir="mark_image"):
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        return
    images = sorted(os.listdir(output_dir))
    if not images:
        print("No images found to rename.")
        return
    for i, image in enumerate(images):
        old_path = os.path.join(output_dir, image)
        new_path = os.path.join(output_dir, f"{i:05d}.jpg")
        os.rename(old_path, new_path)