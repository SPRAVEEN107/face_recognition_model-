import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Basic IR Block
# --------------------------
class IRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return out


# --------------------------
# IR-50 Backbone
# --------------------------
class IR50(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        self.layer1 = self._make_layer(64, 64, 3, stride=2)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 14, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [IRBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(IRBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        x = F.normalize(x, p=2, dim=1)  # hypersphere
        return x
