from torch import nn


def _fc_block():
    return nn.Sequential(
        nn.Linear(8192, 2048),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(2048, 512),
        nn.ReLU(True),
    )


def _cl_block(num_classes):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(512, num_classes),
    )


class MultiLevelModel(nn.Module):
    def __init__(self, num_classes=19):
        super(MultiLevelModel, self).__init__()
        self.f1 = _fc_block()
        self.cl_block1 = _cl_block(num_classes)
        self.f2 = _fc_block()
        self.cl_block2 = _cl_block(num_classes)
        self.f3 = _fc_block()
        self.cl_block3 = _cl_block(num_classes)
        self.f4 = _fc_block()
        self.cl_block4 = _cl_block(num_classes)
        self.f5 = _fc_block()
        self.cl_block5 = _cl_block(num_classes)
        self.f6 = _fc_block()
        self.cl_block6 = _cl_block(num_classes)
        self.f7 = _fc_block()
        self.cl_block7 = _cl_block(num_classes)

    def forward(self, x):
        features = []
        l1_f = self.f1(x[0, :, :])
        l1_out = self.cl_block1(l1_f)

        l2_f = self.f2(x[1, :, :])
        l2_out = self.cl_block2(l2_f)

        l3_f = self.f3(x[2, :, :])
        l3_out = self.cl_block3(l3_f)

        l4_f = self.f4(x[3, :, :])
        l4_out = self.cl_block4(l4_f)

        l5_f = self.f5(x[4, :, :])
        l5_out = self.cl_block5(l5_f)

        l6_f = self.f6(x[5, :, :])
        l6_out = self.cl_block6(l6_f)

        l7_f = self.f7(x[6, :, :])
        l7_out = self.cl_block7(l7_f)

        out = l1_out + l2_out + l3_out + l4_out + l5_out + l6_out + l7_out
        features.append(l1_f)
        features.append(l2_f)
        features.append(l3_f)
        features.append(l4_f)
        features.append(l5_f)
        features.append(l6_f)
        features.append(l7_f)

        return out, features


class MultiModalNet(nn.Module):
    def __init__(self, num_classes=19):
        super(MultiModalNet, self).__init__()
        self.rgb_net = MultiLevelModel(num_classes)
        self.depth_net = MultiLevelModel(num_classes)
        # self.theta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, rgb, depth):
        rgb_out, rgb_features = self.rgb_net(rgb)
        depth_out, depth_features = self.depth_net(depth)

        return rgb_out, rgb_features, depth_out, depth_features
