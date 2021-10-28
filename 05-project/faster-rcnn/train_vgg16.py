'''
使用 VGG16 作为 Backbone
'''
import os
import torch
from torch.utils.data.dataset import T
import torchvision
import transforms as TF

from os import path
from backbone import make_model
from construct import AnchorsGenerator, FasterRCNN
from dataset import TinyDataSet

project_root = path.join(os.getcwd(), '05-project', 'faster-rcnn')
batch_size = 1

def create_model(num_classes):
    # https://download.pytorch.org/models/vgg16-397923af.pth
    
    weights_path = path.join(project_root, 'backbone', 'vgg16_model.pth')
    vgg_feature = make_model(weights_path=weights_path).features # 去掉后面全连接的部分
    backbone = torch.nn.Sequential(
        *list(vgg_feature._modules.values())[:-1] # 去掉最后一个MaxPool
    )
    backbone.out_channels = 512

    anchor_generator = AnchorsGenerator(
        anchor_sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], # 进行 roi pooling 的特征层
        output_size=[7, 7],  # toi pooling 输出特征矩阵尺寸
        sampling_ratio=2     # 采样率
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        roi_heads=roi_pooler
    )

    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device training")

    train_transform = TF.Compose([
        TF.ToTensor(),
        TF.RandomHorizontalFlip(0.5)
    ])

    train_dataset = TinyDataSet(transforms=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        collate_fn=train_dataset.collate_fn
    )

    model = create_model(num_classes=21)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, 
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    model.train()

    for i, [images, targets] in enumerate(train_data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        # losses = sum(loss for loss in loss_dict.values())
        
        # optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()
    


if __name__ == "__main__":
    main()