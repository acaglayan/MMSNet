import torch.nn as nn


class ResNetExtractor(nn.Module):
    def __init__(self, submodule, extracted_layer):
        super(ResNetExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layer = extracted_layer

    def forward(self, x):
        if self.extracted_layer == 1:                       # end of the first plain max-pool
            modules = list(self.submodule.children())[:4]
        elif self.extracted_layer == 2:                     # end of layer1
            modules = list(self.submodule.children())[:5]
        elif self.extracted_layer == 3:                     # end of layer2
            modules = list(self.submodule.children())[:6]
        elif self.extracted_layer == 4:                     # the inner third module of layer3
            modules = list(self.submodule.children())[:6]
            third_module = list(self.submodule.children())[6]
            third_module_modules = list(third_module.children())[:12]  # take the first three inner modules
            third_module = nn.Sequential(*third_module_modules)
            modules.append(third_module)
        elif self.extracted_layer == 5:                     # end of layer3
            modules = list(self.submodule.children())[:7]
        elif self.extracted_layer == 6:                     # end of layer4
            modules = list(self.submodule.children())[:8]
        else:                                               # end of the last avg-pool
            modules = list(self.submodule.children())[:9]

        self.submodule = nn.Sequential(*modules)
        x = self.submodule(x)
        return x
