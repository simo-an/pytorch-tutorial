import torch
from torch.utils.data import Dataset
from PIL import Image
from os import path
from lxml import etree
from .utils import class_dict, parse_xml_to_dict


def check_path(path_text):
    assert path.exists(path_text), f'file {path_text} not found'


class DefectDataset(Dataset):
    '''
    voc_root: voc 数据集的根目录
    year: 哪一个年份的数据集
    transforms: 数据预处理
    text_name: train.txt or val.txt
    '''
    def __init__(self, def_root, transforms=None, text_name='train.txt'):
        # 设置数据集、图片、标注的根目录
        self.root = path.join(def_root, 'NEU-DET')
        self.image_root = path.join(self.root, 'Images')
        self.anno_root = path.join(self.root, 'Annotations')
        # 根据 text_name 拿到对应的标注xml文件路径
        text_path = path.join(self.root, 'Config', text_name)

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
        self.class_dict = class_dict
        # 设置数据预处理器
        self.transforms = transforms

    def get_annotation(self, idx):
        xml_path = self.xml_list[idx]
        check_path(xml_path)

        try:
            xml_reader = open(xml_path)
            xml_text = xml_reader.read()
            xml = etree.fromstring(xml_text)
            annotation = parse_xml_to_dict(xml)['annotation']
        except Exception as e:
            print(f"path is: {xml_path}")
            print(e)

        return annotation

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):  # 返回 image and target
        # 生成 Image
        image = None
        # 从 xml 文件中读取对于图片的地址 -> 设置图片
        annotation = self.get_annotation(idx)
        file_name = str(annotation['filename'])
        file_name = file_name if file_name.endswith('.jpg') else f'{file_name}.jpg'

        image_path = path.join(self.image_root, file_name)
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
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        annotation = self.get_annotation(idx)

        width = int(annotation['size']['width'])
        height = int(annotation['size']['height'])

        return height, width

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target
