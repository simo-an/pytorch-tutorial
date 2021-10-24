'''
自定义dataset：PASCAL VOC2007/2012 数据集
文档：https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
'''
import torch
import utils
from torch.utils.data import Dataset
from PIL import Image
from os import path
from lxml import etree


def check_path(path_text):
    assert path.exists(path_text), f'file {path_text} not found'


class VOCDataset(Dataset):
    '''
    voc_root: voc 数据集的根目录
    year: 哪一个年份的数据集
    transforms: 数据预处理
    text_name: train.txt or val.txt
    '''
    def __init__(self, voc_root, year='2012', transforms=None, text_name='train.txt'):
        # 设置数据集、图片、标注的根目录
        self.root = path.join(voc_root, 'VOCdevkit', f'VOC{year}')
        self.image_root = path.join(self.root, 'JPEGImages')
        self.anno_root = path.join(self.root, 'Annotations')
        # 根据 text_name 拿到对应的标注xml文件路径
        text_path = path.join(self.root, 'ImageSets', 'Main', text_name)

        check_path(text_path)

        with open(text_path) as file_reader:
            self.xml_list = [
                path.join(self.anno_root, f'{line.strip()}.xml')
                for line in file_reader.readlines() if len(line.strip()) > 0
            ]

        # 检查一下上面拿到的路径是否存在
        for xml_path in self.xml_list:
            check_path(xml_path)

        # 设置类别
        self.class_dict = utils.class_dict
        # 设置数据预处理器
        self.transforms = transforms

    def get_annotation(self, idx):
        xml_path = self.xml_list[idx]
        check_path(xml_path)

        xml_reader = open(xml_path)
        xml_text = xml_reader.read()
        xml = etree.fromstring(xml_text)
        annotation = utils.parse_xml_to_dict(xml)['annotation']

        return annotation

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):  # 返回 image and target
        # 生成 Image
        image = None
        # 从 xml 文件中读取对于图片的地址 -> 设置图片
        annotation = self.get_annotation(idx)
        image_path = path.join(self.image_root, annotation['filename'])
        image = Image.open(image_path)

        # 生成 target
        target = {
            'boxes': [],
            'labels': [],
            'image_id': [],
            'area': [],
            'iscrowd': [],
        }

        for obj in annotation['object']:
            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin'])
            ymin = float(bndbox['ymin'])
            xmax = float(bndbox['xmax'])
            ymax = float(bndbox['ymax'])
            target['boxes'].append([xmin, ymin, xmax, ymax])
            target['labels'].append(self.class_dict[obj['name']])
            target['area'].append((xmax - xmin) * (ymax - ymin))

            if 'difficult' in obj:
                target['iscrowd'].append(int(obj['difficult']))
            else:
                target['iscrowd'].append(0)

        # Convert to tensor
        target['boxes'] = torch.as_tensor(target['boxes'])
        target['labels'] = torch.as_tensor(target['labels'])
        target['iscrowd'] = torch.as_tensor(target['iscrowd'])
        target['area'] = torch.as_tensor(target['area'])
        target['image_id'] = torch.tensor([idx])

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def get_height_and_width(self, idx):
        annotation = self.get_annotation(idx)

        width = int(annotation['size']['width'])
        height = int(annotation['size']['height'])

        return height, width
