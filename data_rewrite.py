import json
import os

with open("coco_ball_nao/bbox.json") as f:
    data = json.load(f)

image_width = data["images"][0]["width"]
image_height = data["images"][0]["height"]

images = {}
for image in data["images"]:
    images[image["id"]] = image["file_name"]

if not os.path.exists("coco_ball_nao/txtannotations"):
    os.mkdir("coco_ball_nao/txtannotations")

for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    image_name = images[image_id].split(".")[0]
    
    with open(f"coco_ball_nao/txtannotations/{image_name}.txt", "w") as f:
        bbox1 = annotation['bbox'][0] / image_width
        bbox2 = annotation['bbox'][1] / image_height
        bbox3 = annotation['bbox'][2] / image_width
        bbox4 = annotation['bbox'][3] / image_height
        f.write(f"{annotation['category_id']} {bbox1} {bbox2} {bbox3} {bbox4}")
