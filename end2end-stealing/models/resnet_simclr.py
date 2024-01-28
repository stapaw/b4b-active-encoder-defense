import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, loss=None, include_mlp = True, entropy=False):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet34": models.resnet34(pretrained=False,
                                                        num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        self.loss = loss
        self.entropy = entropy
        dim_mlp = self.backbone.fc.in_features # 512
        if include_mlp:
            # add mlp projection head
            # originally self.backbone.fc = nn.Linear(512 * block.expansion, num_classes)
            # This modifies the fc layer to add another hidden layer inside.
            if self.loss == "symmetrized":
                self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                 nn.BatchNorm1d(dim_mlp),
                                                 nn.ReLU(inplace=True), self.backbone.fc)
            else:
                self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        else:
            #self.backbone.fc = nn.Linear(dim_mlp, dim_mlp, bias=False)
            #self.backbone.fc.weight.data.copy_(torch.eye(dim_mlp))
            #self.backbone.fc.weight.requires_grad = False # last layer does nothing. only there to be compatible with resnet
            self.backbone.fc = nn.Identity() # no head used

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise Exception(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet34 or resnet50")
        else:
            return model

    def forward(self, x):
        # if self.loss == "supcon":
        #     return F.normalize(self.backbone(x), dim=1)
        if self.entropy == "True":
            return F.softmax(self.backbone(x), dim=1)
        return self.backbone(x)


class ResNetSimCLRV2(nn.Module):

    def __init__(self, base_model, out_dim, loss=None, include_mlp=False):
        super(ResNetSimCLRV2, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet34":  models.resnet34(pretrained=False, num_classes=out_dim)
                            ,"resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        self.include_mlp = include_mlp
        self.loss = loss
        dim_mlp = self.backbone.fc.in_features
        if self.loss == "symmetrized":
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                             nn.BatchNorm1d(dim_mlp),
                                             nn.ReLU(inplace=True),
                                             self.backbone.fc)
        else:
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                             nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise Exception(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet34 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        if self.include_mlp:
            x = self.backbone.fc(x)
        return x

class HeadSimCLR(nn.Module):
    """ Takes a representation as input and passes it through the head to get g(z)"""
    def __init__(self, out_dim):
        super(HeadSimCLR, self).__init__()
        self.expansion = 1 # 4 for resnet50+
        self.backbone = nn.Linear(512 * self.expansion, out_dim)
        dim_mlp = 512
        self.backbone = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone)


    def forward(self, x):
        return self.backbone(x)


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=512, out_dim=512):
        """
        dim: feature dimension (default: 512)
        out_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, out_dim, bias=False),
                                        nn.BatchNorm1d(out_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(out_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

class HeadSimSiam(nn.Module):
    """ Takes a representation as input and passes it through the head to get g(z)"""
    def __init__(self, out_dim):
        super(HeadSimSiam, self).__init__()
        self.expansion = 1 # 4 for resnet50+
        self.backbone = nn.Linear(512 * self.expansion, out_dim)
        prev_dim = 512
        self.backbone = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.backbone,
            nn.BatchNorm1d(dim, affine=False))  # output layer
        self.backbone[
            6].bias.requires_grad = False  # hack: not use bias as it is followed by BN


    def forward(self, x):
        return self.backbone(x)

class WatermarkMLP(nn.Module):

    def __init__(self, n_inputs, n_outputs):
        super(WatermarkMLP, self).__init__()
        self.input = nn.Linear(n_inputs, 256)
        # self.hidden1 = nn.Linear(256,256)
        # self.hidden2 = nn.Linear(256, 128)
        self.hidden = nn.Linear(256, 128)
        self.output = nn.Linear(128, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        # x = self.hidden1(x)
        # x = F.relu(x)
        # x = self.hidden2(x)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        return x