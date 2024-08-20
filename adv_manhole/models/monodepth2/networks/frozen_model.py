import torch

from adv_manhole.models.monodepth2.networks import ResnetEncoder, DepthDecoder

class ResnetEncoderFrozen(torch.nn.Module):
    def __init__(self, require_grad=True):
        super(ResnetEncoderFrozen, self).__init__()
        self.model = ResnetEncoder(18, False)
        self.require_grad = require_grad
        self.num_ch_enc = self.model.num_ch_enc
        
        if not self.require_grad:
            for child in self.model.children():
                for param in child.parameters():
                    param.requires_grad = False
                    
    def forward(self, input_img):
        features_map = self.model(input_img)
        return features_map
    
class DepthDecoderFrozen(torch.nn.Module):
    def __init__(self, num_ch, require_grad=True):
        super(DepthDecoderFrozen, self).__init__()
        self.model = DepthDecoder(num_ch)
        self.require_grad = require_grad
        
        if not self.require_grad:
            for child in self.model.children():
                for param in child.parameters():
                    param.requires_grad = False
                    
    def forward(self, input_features):
        disparity = self.model(input_features)
        return disparity[("disp", 0)]
    
class CombinedFrozenModel(torch.nn.Module):
    def __init__(self, require_grad=True):
        super(CombinedFrozenModel, self).__init__()
        self.encoder = ResnetEncoder(18, False)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)
        self.require_grad = require_grad
        
        if not self.require_grad:
            for child in self.encoder.children():
                for param in child.parameters():
                    param.requires_grad = False

            for child in self.decoder.children():
                for param in child.parameters():
                    param.requires_grad = False
                
    def forward(self, x):
        features_map = self.encoder(x)
        disparity = self.decoder(features_map)
        
        return disparity[("disp", 0)]