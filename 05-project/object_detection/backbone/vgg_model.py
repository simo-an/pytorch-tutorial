import torch
import torch.nn as nn

class VGG(nn.Module):
    features: any
    classifier: nn.Sequential

    def __init__(self, 
        features,
        class_num=1000,
        init_weights=False,
        weights_path = None
    ):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, class_num)
        )
        if init_weights and weights_path is None:
            self._init_weights()

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))
    
    def forward(self, image):
        print(image.shape)
        # 224 * 224 * 3
        image = self.features(image)
        # 512 * 7 * 7
        image = torch.flatten(image, start_dim=1)
        
        return self.classifier(image)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def make_features(cfg: list):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=v,
                kernel_size=3,
                padding=1
            )
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    
    return nn.Sequential(*layers)

model_cfg_dict = {
    "vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 412, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_model(model_name="vgg16", weights_path=None):
    model_cfg = model_cfg_dict[model_name]

    assert model_cfg is not None, f"model {model_name} does not exist!"

    return VGG(
        features=make_features(model_cfg),
        weights_path=weights_path
    )