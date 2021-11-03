# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class ResNet18(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True):
        super().__init__()
        print('| A ResNet18 network is instantiated, pre-trained: {}, '
              'number of classes: {}'.format(pretrained, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet18(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self._n_classes)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = self.fc(x)
        return x

class ResNet18_dfc(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True):
        super().__init__()
        print('| A ResNet18_dfc network is instantiated, pre-trained: {}, '
              'number of classes: {}'.format(pretrained, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet18(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_sl = nn.Linear(in_features=512, out_features=self._n_classes)
        self.fc_ce = nn.Linear(in_features=512, out_features=self._n_classes)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc_sl.weight.data)
            # nn.init.constant_(self.fc_sl.weight.data,1)
            if self.fc_sl.bias is not None:
                nn.init.constant_(self.fc_sl.bias.data, val=0)
            nn.init.kaiming_normal_(self.fc_ce.weight.data)
            if self.fc_ce.bias is not None:
                nn.init.constant_(self.fc_ce.bias.data, val=0)

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        out2 = self.fc_ce(x)
        x_clone = x.detach().clone()
        out1 = self.fc_sl(x_clone)
        return out1, out2

class ResNet18_Normalized(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True):
        super().__init__()
        print('| A ResNet18_Normalized network is instantiated, pre-trained: {}, '
              'number of classes: {}'.format(pretrained, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet18(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self._n_classes, bias=False)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))
        x = self.fc(x)
        x = torch.clamp(x, min=-1, max=1)
        x = 30 * x
        return x

class ResNet18_subcenter(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, k=3, pretrained=True, use_two_step=False):
        super().__init__()
        print('| A ResNet18_subcenter network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        self._k = k
        resnet = torchvision.models.resnet18(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(in_features=512, out_features=self._n_classes * self._k, bias=False)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))
        x = self.fc(x)
        feat = x.view(N, self._n_classes, self._k)
        x = self.maxpool(feat)
        x = x.view(N, self._n_classes)
        x = torch.clamp(x, min=-1, max=1)
        x = 30 * x
        return x, feat

class ResNet18_ss(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True, use_two_step=False):
        super().__init__()
        print('| A ResNet18_ss network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet18(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self._n_classes)
        self.rot_pred = nn.Linear(in_features=512, out_features=4)
        self.grey_pred = nn.Linear(in_features=512, out_features=3)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        rot_pred = self.rot_pred(x)
        grey_pred = self.grey_pred(x)
        x = self.fc(x)
        return x, rot_pred

class ResNet18_ss_Normalized(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True, use_two_step=False):
        super().__init__()
        print('| A ResNet18_Normalized network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet18(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self._n_classes,bias=False)
        self.rot_pred = nn.Linear(in_features=512, out_features=4)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        rot_pred = self.rot_pred(x)
        with torch.no_grad():
            self.fc.weight.div(torch.norm(self.fc.weight, dim=1, keepdim=True))
        x = self.fc(x)
        return x, rot_pred

class ResNet18_ss_subcenter(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, k=3, pretrained=True, use_two_step=False):
        super().__init__()
        print('| A ResNet18_subcenter network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        self._k = k
        resnet = torchvision.models.resnet18(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(in_features=512, out_features=self._n_classes * self._k, bias=False)
        self.rot_pred = nn.Linear(in_features=512, out_features=4)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        rot_pred = self.rot_pred(x)
        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))
        x = self.fc(x)
        feat = x.view(N, self._n_classes, self._k)
        x = self.maxpool(feat)
        x = x.view(N, self._n_classes)
        return x, rot_pred, feat


class ResNet34(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),

    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True, use_two_step=True):
        super().__init__()
        print('| A ResNet34 network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet34(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self._n_classes)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = self.fc(x)
        return x

class ResNet50(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True, use_two_step=False):
        super().__init__()
        print('| A ResNet50 network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet50(pretrained=self._pretrained)
        # feature output is (N, 2048)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=self._n_classes)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = self.fc(x)
        return x

class ResNet50_Normalized(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True, use_two_step=False):
        super().__init__()
        print('| A ResNet50_Norm network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet50(pretrained=self._pretrained)
        # feature output is (N, 2048)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=self._n_classes, bias=False)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))
        x = self.fc(x)
        return x, None

class ResNet50_subcenter(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, k=3, pretrained=True, use_two_step=False):
        super().__init__()
        print('| A ResNet50_sub network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._k = k
        self._n_classes = n_classes
        resnet = torchvision.models.resnet50(pretrained=self._pretrained)
        # feature output is (N, 2048)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=self._n_classes * self._k, bias=False)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))
        x = self.fc(x)
        feat = x.view(N, self._n_classes, self._k)
        x = self.maxpool(feat)
        x = x.view(N, self._n_classes)
        return x, feat

if __name__ == '__main__':
    net = ResNet18_subcenter()
    x = torch.rand(64, 3, 448, 448)
    y = net(x)
    print(y.shape)
