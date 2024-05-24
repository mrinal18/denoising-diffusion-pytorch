# meshnet.py
import torch
import torch.nn as nn
import json

def construct_layer(dropout_p, bnorm, gelu, **kwargs):
    layers = []
    conv2d_params = {key: kwargs[key] for key in ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']}
    layers.append(nn.Conv2d(**conv2d_params))

    if bnorm:
        layers.append(nn.BatchNorm2d(conv2d_params['out_channels'], track_running_stats=False))
    if gelu:
        layers.append(nn.GELU())
    if dropout_p > 0:
        layers.append(nn.Dropout2d(dropout_p))
    if conv2d_params['stride'] > 1:
        layers.append(nn.Upsample(scale_factor=conv2d_params['stride'], mode='nearest'))

    return nn.Sequential(*layers)

class MeshNet(nn.Module):
    def __init__(self, in_channels, n_classes, channels, config_file, fat=None):
        super(MeshNet, self).__init__()
        self.channels = in_channels
        self.self_condition = False
        with open(config_file, "r") as f:
            config = json.load(f)

        self.layers = nn.ModuleList([
            construct_layer(
                dropout_p=config["dropout_p"],
                bnorm=config["bnorm"],
                gelu=config["gelu"],
                **layer_config
            ) for layer_config in config["layers"]
        ])
        self.final_layer = nn.Conv2d(channels[-1], in_channels, kernel_size=1)
        self.out_dim = in_channels

    def forward(self, x, t, self_cond=None):
        if self_cond is not None:
            x = torch.cat([x, self_cond], dim=1) 
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x
