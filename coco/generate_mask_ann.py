import os
from pycocotools.coco import COCO
import random
import cv2
import shutil
import numpy as np
import json
from tqdm import tqdm

annotation_file = './annotations/ann.json'
image_folder = './images'
modify_category = True
modify_alpha = 60

if modify_category:
    # read the annotation file
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    target_json_name = annotation_file[:-5]+'_x'+str(modify_alpha)+'.json'
    # scan the categories
    categories = data['categories']
    category_ids = [category['id'] for category in categories]
    # modify the category ids = (ids+1)*60
    for category in categories:
        category['id'] = (category['id'] + 1) * 60
    # modify the category ids in annotations
    annotations = data['annotations']
    for annotation in annotations:
        annotation['category_id'] = (annotation['category_id'] + 1) * 60
    # save the modified annotation file
    with open(target_json_name, 'w') as f:
        json.dump(data, f)
    coco = COCO(target_json_name)
else:
    coco = COCO(annotation_file)


category_ids = coco.getCatIds()
categories = coco.loadCats(category_ids)
category_names = [category['name'] for category in categories]
print(category_names)  # Output: ['top', 'mid', 'bottom']

image_ids = coco.getImgIds()

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
train_folder_name = 'train2017'
val_folder_name = 'val2017'
test_folder_name = 'test2017'
anno_folder_name = 'annotations'
output_folder = './coco_stuff164k'

# Create the train, val, and test folders
os.makedirs(os.path.join(output_folder, 'images',train_folder_name), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'images',val_folder_name), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'images',test_folder_name), exist_ok=True)
os.makedirs(os.path.join(output_folder, anno_folder_name), exist_ok=True)
os.makedirs(os.path.join(output_folder, anno_folder_name, train_folder_name), exist_ok=True)
os.makedirs(os.path.join(output_folder, anno_folder_name, val_folder_name), exist_ok=True)
os.makedirs(os.path.join(output_folder, anno_folder_name, test_folder_name), exist_ok=True)

image_filenames = os.listdir(image_folder)
random.shuffle(image_filenames)

num_images = len(image_filenames)

train_split = int(train_ratio * num_images)
val_split = train_split + int(val_ratio * num_images)

print("generating mask annotations...")


for i, image_id in tqdm(enumerate(image_ids),total=len(image_ids), desc='Processing images'):
    image_info = coco.loadImgs(image_id)[0]
    filename = image_info['file_name']
    src = os.path.join(image_folder, filename)

    if i < train_split:
        dst = os.path.join(output_folder,'images', train_folder_name, filename)
        dst_anno_folder = os.path.join(output_folder, anno_folder_name, train_folder_name)
    elif i < val_split:
        dst = os.path.join(output_folder,'images',val_folder_name, filename)
        dst_anno_folder = os.path.join(output_folder, anno_folder_name, val_folder_name)
    else:
        dst = os.path.join(output_folder,'images', test_folder_name, filename)
        dst_anno_folder = os.path.join(output_folder, anno_folder_name, test_folder_name)

    shutil.copy(src, dst)

    category_ids = coco.getCatIds()
    annotations_ids = coco.getAnnIds(imgIds=image_info['id'], catIds=category_ids, iscrowd=False)
    annotations = coco.loadAnns(annotations_ids)
    anns_img = np.zeros((image_info['height'], image_info['width']))
    for ann in annotations:
        anns_img = np.maximum(anns_img, coco.annToMask(ann) * ann['category_id'])
    
    # save as png
    cv2.imwrite(os.path.join(dst_anno_folder, filename[:-4] + '.png'), anns_img)
