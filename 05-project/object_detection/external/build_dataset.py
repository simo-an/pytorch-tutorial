import os
import os.path as path

# Hyper Parameters
train_ratio = 0.9

classes_num = 6
classes_dict = {
    "crazing": 1,
    "inclusion": 2,
    "patches": 3,
    "pitted_surface": 4,
    "rolled-in_scale": 5,
    "scratches": 6,
}
classes_index = [key for key in classes_dict]

category_num = 300
train_num = int(category_num * train_ratio)
val_num = category_num - train_num

root_path = os.getcwd()

image_path = path.join(root_path, 'NEU-DET', 'Images')
annot_path = path.join(root_path, 'NEU-DET', 'Annotations')
config_path = path.join(root_path, 'NEU-DET', 'Config')

train_image_list = []
val_image_list = []

crazing_image_list = []
inclusion_image_list = []
patches_image_list = []
pitted_surface_image_list = []
rolled_in_scale_image_list = []
scratches_image_list = []

for class_idx in range(classes_num):
    for val_idx in range(1, val_num + 1):
        val_image_list.append(f"{classes_index[class_idx]}_{val_idx}")
    for train_idx in range(val_num + 1, category_num + 1):
        train_image_list.append(f"{classes_index[class_idx]}_{train_idx}")

val_file_handler = open(path.join(config_path, 'val.txt'), mode='w+')
train_file_handler = open(path.join(config_path, 'train.txt'), mode='w+')


def check_item(item):
    img = path.join(image_path, f'{item}.jpg')
    annot = path.join(annot_path, f'{item}.xml')
    assert path.exists(img), f'image {img} not found'
    assert path.exists(annot), f'annotation {annot} not found'

for val_item in val_image_list:
    check_item(val_item)
    val_file_handler.write(f'{val_item}\n')
for train_item in train_image_list:
    check_item(val_item)
    train_file_handler.write(f'{train_item}\n')

val_file_handler.close()
train_file_handler.close()