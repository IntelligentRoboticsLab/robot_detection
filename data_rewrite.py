"""
Module that splits COCO Nao/Ball dataset by filtering out
unused images and creates a 80/20 train/val split

TODO: refractor code to not use naive count method for splitting data
TODO: generate json file for annotations instead of txt file to reduce clutter
 """
import json
import os
import shutil

# function to create subfolders for both train and validation splits
def create_subfolders(parent_folder, subfolders):
    for subfolder in subfolders:
        path = os.path.join(parent_folder, subfolder)
        if not os.path.exists(path):
            os.mkdir(path)


# get COCO format annotations
with open("data/coco_nao/bbox.json") as f:
    data = json.load(f)

# image width and height for each image
img_width = data["images"][0]["width"]
img_height = data["images"][0]["height"]

# dict with file names for each image
images = {}
for image in data["images"]:
    images[image["id"]] = image["file_name"]

# Create train folders
create_subfolders("data/coco_nao/train", ["labels", "images"])

# Create validation folders
create_subfolders("data/coco_nao/val", ["labels", "images"])

# get total amount of images that are part of desired category
# accompanied with all the annotations that will be used
TOTAL = 0
category_annotations = []

for annotation in data["annotations"]:
    if annotation['category_id'] == 1:
        TOTAL += 1
        category_annotations.append(annotation)

# use 80/20 split
TRAIN_SPLIT = TOTAL - (int(0.2 * TOTAL))
COUNT = 0

# loop through annotations that will be used
for annotation in category_annotations:
    image_id = annotation["image_id"]
    image_name = images[image_id].split(".")[0]
    image_filename = image_name + ".png"

    # check if train split is already filled based on count
    SPLIT_DIR = "train" if COUNT < TRAIN_SPLIT else "val"

    # open a txt file with the annotation for each used image
    with open(f"data/coco_nao/{SPLIT_DIR}/labels/{image_name}.txt", "w") as f:

        # convert the COCO format annotations into YOLO format
        x_tl, y_tl, w, h = annotation['bbox']

        dw = 1.0 / img_width
        dh = 1.0 / img_height

        x_center = x_tl + w / 2.0
        y_center = y_tl + h / 2.0

        x = x_center * dw
        y = y_center * dh
        w = w * dw
        h = h * dh

        # write coordinates to txt file
        f.write(f"{annotation['category_id']} {x} {y} {w} {h}")

        # copy images from original folder to respective train/val subfolder
        image_filepath = os.path.join("data/coco_nao/images", image_filename)
        if os.path.exists(image_filepath):
            shutil.copy(image_filepath, f"data/coco_nao/{SPLIT_DIR}/images")

        COUNT += 1

