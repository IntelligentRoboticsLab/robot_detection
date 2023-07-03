import json
import os
import shutil

with open("data/coco_ball_nao/bbox.json") as f:
    data = json.load(f)

image_width = 1 #data["images"][0]["width"]
image_height = 1 #data["images"][0]["height"]

images = {}
for image in data["images"]:
    images[image["id"]] = image["file_name"]

if not os.path.exists("data/coco_ball_nao/labels/"):
    os.mkdir("data/coco_ball_nao/labels/")

if not os.path.exists("data/coco_ball_nao/filtered_images"):
    os.mkdir("data/coco_ball_nao/filtered_images")

filtered_images_dir = "data/coco_ball_nao/filtered_images/"

for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    image_name = images[image_id].split(".")[0]
    image_filename = image_name + ".png"  # Assuming the images have a .jpg extension

    with open(f"data/coco_ball_nao/labels/{image_name}.txt", "w") as f:
        bbox1 = annotation['bbox'][0] / image_width
        bbox2 = annotation['bbox'][1] / image_height
        bbox3 = annotation['bbox'][2] / image_width
        bbox4 = annotation['bbox'][3] / image_height
        f.write(f"{annotation['category_id']} {bbox1} {bbox2} {bbox3} {bbox4}")

    # image_filepath = os.path.join("data/coco_ball_nao/images", image_filename)  # Provide the path to the separate folder
    # if os.path.exists(image_filepath):
    #     shutil.copy(image_filepath, filtered_images_dir)
