"""
Module that creates additional data using data augmentation
TODO: test various transforms aside from current implementation
"""
import os
import albumentations as A
import cv2


def create_augmentation_pipeline():
    """
    creates transform object using albumentations
    """
    # return the transform for image and bounding box
    return A.Compose([
        A.GaussNoise(),
        A.Flip(p=0.5),
        A.MotionBlur(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo'))

LABEL_PATHS="data/coco_nao/train/labels/"
IMAGE_PATHS="data/coco_nao/train/images/"

# get transform object
transform = create_augmentation_pipeline()

# create temporary directories holding augmented images
# need to add way to handle original data to allow for rerunning script
if not os.path.exists('data/coco_nao/train/augmented_images'):
    os.mkdir('data/coco_nao/train/augmented_images')

if not os.path.exists('data/coco_nao/train/augmented_labels'):
    os.mkdir('data/coco_nao/train/augmented_labels')

COUNT = 1

# load bounding boxes from label files
label_files = sorted(os.listdir(LABEL_PATHS))
image_files = sorted(os.listdir(IMAGE_PATHS))

j = 0
# loop over all label files
for label_path, image_path in zip(label_files, image_files):
    # load label file
    label_strings = open(LABEL_PATHS+label_path, encoding='utf-8').read().splitlines()
    image = cv2.imread(IMAGE_PATHS+image_path)
    label_values = list(map(float, label_strings[0].split(" ")))
    label_values = label_values[1:] + [label_values[0]]
    augmented_data = [transform(image=image, bboxes=[label_values]) for _ in range(3)]
    augmented_data = augmented_data + [{'image': image, 'bboxes': [tuple(label_values)]}]

    for i, data in enumerate(augmented_data):
        cv2.imwrite(os.path.join('data/coco_nao/train/augmented_images',
                                 f'augmented_{j+i}.png'), data['image'])

        with open(f"data/coco_nao/train/augmented_labels/augmented_{j+i}.txt",
                   "w", encoding='utf-8') as f:
            anno = data['bboxes'][0]
            f.write(f"{int(anno[-1])} {anno[0]} {anno[1]} {anno[2]} {anno[3]}")

    j += 4
