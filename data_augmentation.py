import albumentations as A
import cv2
import os


def create_augmentation_pipeline():
    # Define the transform for image and bounding box
    transform = A.Compose([
        A.GaussNoise(),  # Adjust the crop size as needed
        A.Flip(p=0.5),
        A.MotionBlur(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo'))

    return transform

label_paths="data/coco_nao/train/labels/"
image_paths="data/coco_nao/train/images/"

transform = create_augmentation_pipeline()
# augmented = transform(image=image, bboxes=bboxes)

if not os.path.exists('data/coco_nao/train/augmented_images'):
    os.mkdir('data/coco_nao/train/augmented_images')

if not os.path.exists('data/coco_nao/train/augmented_labels'):
    os.mkdir('data/coco_nao/train/augmented_labels')

COUNT = 1

# load bounding boxes from label files
label_files = sorted(os.listdir(label_paths))
image_files = sorted(os.listdir(image_paths))

j = 0
# loop over all label files
for label_path, image_path in zip(label_files, image_files):
    # load label file
    label_strings = open(label_paths+label_path).read().splitlines()
    image = cv2.imread(image_paths+image_path)
    label_values = list(map(float, label_strings[0].split(" ")))
    label_values = label_values[1:] + [label_values[0]]
    augmented_data = [transform(image=image, bboxes=[label_values]) for _ in range(3)] + [{'image': image, 'bboxes': [tuple(label_values)]}]

    for i in range(len(augmented_data)):
        cv2.imwrite(os.path.join('data/coco_nao/train/augmented_images', f'augmented_{j+i}.png'), augmented_data[i]['image'])

        with open(f"data/coco_nao/train/augmented_labels/augmented_{j+i}.txt", "w") as f:
            anno = augmented_data[i]['bboxes'][0]
            f.write(f"{int(anno[-1])} {anno[0]} {anno[1]} {anno[2]} {anno[3]}")

    j += 4
