import json
import os
import shutil

with open("data/coco_nao/bbox.json") as f:
    data = json.load(f)

img_width = data["images"][0]["width"]
img_height = data["images"][0]["height"]

images = {}
for image in data["images"]:
    images[image["id"]] = image["file_name"]

if not os.path.exists("data/coco_nao/labels/"):
    os.mkdir("data/coco_nao/labels/")

if not os.path.exists("data/coco_nao/filtered_images"):
    os.mkdir("data/coco_nao/filtered_images")

FILTERED_IMAGES_DIR = "data/coco_nao/filtered_images/"

for annotation in data["annotations"]:
    if annotation['category_id'] == 1 :
        image_id = annotation["image_id"]
        image_name = images[image_id].split(".")[0]
        image_filename = image_name + ".png"


        with open(f"data/coco_nao/labels/{image_name}.txt", "w") as f:
            x_tl, y_tl, w, h = annotation['bbox']

            dw = 1.0 / img_width
            dh = 1.0 / img_height

            x_center = x_tl + w / 2.0
            y_center = y_tl + h / 2.0

            x = x_center * dw
            y = y_center * dh
            w = w * dw
            h = h * dh
            f.write(f"{annotation['category_id']} {x} {y} {w} {h}")

            image_filepath = os.path.join("data/coco_nao/images", image_filename)  # Provide the path to the separate folder
            if os.path.exists(image_filepath):
                shutil.copy(image_filepath, FILTERED_IMAGES_DIR)