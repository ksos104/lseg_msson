import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lseg_blocks_zs import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np
import pandas as pd

import os

from .tokencut_clip_model import dino
from .tokencut_clip_model import object_discovery as tokencut
from .tokencut_clip_model import bilateral_solver
from .tokencut_clip_model import metric
from typing import Tuple

from .tokencut_clip_model.dino_decoder import DINODecoder

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out

class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x

# tanh relu
class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class LSeg(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clipRN50x4_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
            "clipRN50x16_vitb32_384": [2, 5, 8, 11],
            "clipRN50x4_vitb32_384": [2, 5, 8, 11],
            "clip_resnet101": [0, 1, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            self.use_pretrained,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # self.scratch.output_conv = head

        self.auxlayer = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        
        # cosine similarity as logits
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

        if backbone in ["clipRN50x16_vitl16_384", "clipRN50x16_vitb32_384"]:
            self.out_c = 768
        elif backbone in ["clipRN50x4_vitl16_384", "clipRN50x4_vitb32_384"]:
            self.out_c = 640
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]

        self.scratch.output_conv = head

        self.texts = []
        # original
        label = ['others', '']
        for class_i in range(len(self.label_list)):
            label[1] = self.label_list[class_i]
            text = clip.tokenize(label)
            self.texts.append(text)

    def forward(self, x, class_info):
        texts = [self.texts[class_i] for class_i in class_info]
        
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        self.logit_scale = self.logit_scale.to(x.device)
        text_features = [self.clip_pretrained.encode_text(text.to(x.device)) for text in texts]

        image_features = self.scratch.head1(path_1)


        imshape = image_features.shape
        image_features = [image_features[i].unsqueeze(0).permute(0,2,3,1).reshape(-1, self.out_c) for i in range(len(image_features))]

        # normalized features
        image_features = [image_feature / image_feature.norm(dim=-1, keepdim=True) for image_feature in image_features]
        text_features = [text_feature / text_feature.norm(dim=-1, keepdim=True) for text_feature in text_features]
        
        logits_per_images = [self.logit_scale * image_feature.half() @ text_feature.t() for image_feature, text_feature in zip(image_features, text_features)]
        outs = [logits_per_image.float().view(1, imshape[2], imshape[3], -1).permute(0,3,1,2) for logits_per_image in logits_per_images]
        out = torch.cat([out for out in outs], dim=0)

        out = self.scratch.output_conv(out)
            
        return out


class LSegNetZS(LSeg):
    """Network for semantic segmentation."""
    def __init__(self, label_list, path=None, scale_factor=0.5, aux=False, use_relabeled=False, use_pretrained=True, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.scale_factor = scale_factor
        self.aux = aux
        self.use_relabeled = use_relabeled
        self.label_list = label_list
        self.use_pretrained = use_pretrained

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

class TokenCut_CLIP(nn.Module):
    def __init__(self,
        arch: str,
        patch_size: int,
        image_size: Tuple[int],
        n_iter: int,
        label_list,
        # head,
        features=256,
        backbone="vitb_rn50_384",
        use_pretrained=False,
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super().__init__()
        
        ## DINO init
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_iter = n_iter
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # build model
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            feat_dim = 384
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            feat_dim = 384
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            feat_dim = 768
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            feat_dim = 768
            
        ## TokenCut Model
        vit_feat = 'k'                ## [k, q, v, kqv]   
        self.tau = 0.2
        self.sigma_spatial = 16
        self.sigma_luma = 16
        self.sigma_chroma = 8
                
        self.model = dino.ViTFeat(url, feat_dim, arch, vit_feat, patch_size)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(self.device)
        
        self.input_dim = feat_dim
        
        # self.head = nn.Linear(self.input_dim, 512)
        self.head = nn.Conv2d(self.input_dim, 512, kernel_size=1)
        # self.mask_layers = nn.Linear(self.input_dim, 1)
        # self.mask_layers = nn.Conv2d(self.input_dim, 1, kernel_size=1)
        
        ## Decoder
        self.dino_decoder = DINODecoder(
            norm = "GN",
        )
        
        ## CLIP init
        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clipRN50x4_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
            "clipRN50x16_vitb32_384": [2, 5, 8, 11],
            "clipRN50x4_vitb32_384": [2, 5, 8, 11],
            "clip_resnet101": [0, 1, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            use_pretrained,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # self.scratch.output_conv = head

        self.auxlayer = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        
        # cosine similarity as logits
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale.requires_grad = False

        if backbone in ["clipRN50x16_vitl16_384", "clipRN50x16_vitb32_384"]:
            self.out_c = 768
        elif backbone in ["clipRN50x4_vitl16_384", "clipRN50x4_vitb32_384"]:
            self.out_c = 640
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]

        self.scratch.output_conv = nn.Sequential(
            Interpolate(scale_factor=self.patch_size, mode="bilinear", align_corners=True),
        )

        self.texts = []
        # original
        label = ['others', '']
        for class_i in range(len(label_list)):
            label[1] = label_list[class_i]
            text = clip.tokenize(label)
            self.texts.append(text)
            
    
    def get_tokencut_binary_map(self, input_images, backbone, patch_size, tau) :
        bs, h, w = input_images.shape[0], input_images.shape[-2], input_images.shape[-1]
        feat_h = h // patch_size
        feat_w = w // patch_size
        outputs, feat = backbone(input_images)
                        
        seed, bipartition, eigvec = tokencut.ncut(feat, [bs, feat_h, feat_w], [patch_size, patch_size], [h,w], tau)
        
        return outputs, bipartition, eigvec
        
    def forward(self, x, class_info):           ## x.shape == [bs, 3, img_w, img_h]
        
        texts = [self.texts[class_i] for class_i in class_info]
        text_features = [self.clip_pretrained.encode_text(text.to(x.device)) for text in texts]
        text_features = [text_feature / text_feature.norm(dim=-1, keepdim=True) for text_feature in text_features]
        
        input_images = x
        bs, w, h = input_images.shape[0], input_images.shape[2] - input_images.shape[2] % self.patch_size, input_images.shape[3] - input_images.shape[3] % self.patch_size
        input_images = input_images[:, :, :w, :h]

        w_featmap = input_images.shape[-2] // self.patch_size
        h_featmap = input_images.shape[-1] // self.patch_size
        
        # features_list = []
        mask_output_list = []
        # attentions_list = []
        # binary_attn_list = []
        
        '''
            Use decoder
            features.shape = [bs, 3600, 384]
        '''
        remains = torch.ones((bs,self.image_size[0],self.image_size[1])).to(self.device)
        for i in range(self.n_iter):       
            features, bipartition, eigvec = self.get_tokencut_binary_map(input_images, self.model, self.patch_size, self.tau)
            
            if i == 0:
                orig_features = features
            else:
                features = orig_features

            bipartition = bipartition * remains

            output_solver, binary_solver = bilateral_solver.bilateral_solver_output(input_images, bipartition, sigma_spatial = self.sigma_spatial, sigma_luma = self.sigma_luma, sigma_chroma = self.sigma_chroma)
            
            if metric.IoU(bipartition, binary_solver) < 0.5:
                binary_solver = bipartition        
                
            if i == self.n_iter - 1:
                binary_solver = remains
            else:
                remains = (remains * (1-binary_solver*1))   
            
            output_solver = F.interpolate(output_solver.unsqueeze(1), size=(h_featmap,w_featmap), mode='bilinear')

            bipartition = F.interpolate(bipartition.unsqueeze(1), size=(h_featmap,w_featmap), mode='nearest')
            if binary_solver.shape[0] == self.image_size[0]:
                binary_solver = F.interpolate((binary_solver*1.).unsqueeze(0).unsqueeze(0), size=(h_featmap,w_featmap), mode='nearest')
            else:
                binary_solver = F.interpolate((binary_solver*1.).unsqueeze(1), size=(h_featmap,w_featmap), mode='nearest')
            
            features = features[:,1:,:].to(torch.float)
            attentions = output_solver.to(torch.float)
            
            '''
                Decoding
            '''
            dec_attentions = binary_solver * attentions
            features, dec_attentions = self.dino_decoder(features, dec_attentions)

            # features_list.append(features)
            # attentions_list.append(attentions)
            # binary_attn_list.append(binary_solver)

            input_images = input_images * remains.to(torch.float).unsqueeze(dim=1)
            
            masked_features = features.permute(0,2,1).reshape(bs, self.input_dim, h_featmap, w_featmap)
            # mask_output = self.head(masked_features).reshape(bs, -1, 3600).permute(0, 2, 1)
            mask_output = self.head(masked_features)
            mask_output_list.append(mask_output*binary_solver)
        
        # binary_attns = torch.concat(binary_attn_list, dim=1).unsqueeze(dim=-1)
        # features_tensor = torch.stack(features_list, dim=1)
        mask_outputs = torch.stack(mask_output_list, dim=1)
        mask_outputs = mask_outputs.sum(dim=1).reshape(bs, -1, h_featmap*w_featmap).permute(0, 2, 1)
        # attentions = torch.concat(attentions_list, dim=1).unsqueeze(dim=-1)
                
        logit_scale = self.logit_scale.exp()
                
        ## mask_classification
        mask_outputs = mask_outputs / mask_outputs.norm(dim=-1, keepdim=True)
        # if self.training:
        #     cls_score = logit_scale * mask_outputs @ self.text_features.clone().detach().t()
        # else:
        #     cls_score = logit_scale * mask_outputs @ self.text_features_test.clone().detach().t()
        text_features = torch.stack(text_features, dim=0)
        # cls_score = logit_scale * mask_outputs @ text_features.clone().detach().t()
        cls_score = logit_scale.half() * mask_outputs.half() @ text_features.clone().detach().transpose(-1,-2)

        # bg_score = logit_scale * mask_outputs @ self.bg_feature.t()
        # outputs_class = torch.cat((cls_score, bg_score), -1)
        outputs_class = cls_score.float()                   ## shape: [bs, 3600, 2]
        # outputs = {"pred_logits": outputs_class}
        
        outputs_class = outputs_class.permute(0, 2, 1).reshape(bs, 2, h_featmap, w_featmap)
        outputs_class = self.scratch.output_conv(outputs_class)
                    
        # mask_inputs = features_tensor.permute(0,1,3,2).reshape(bs*self.n_iter, self.input_dim, 60, 60)
        # mask_inputs = self.mask_layers(mask_inputs).reshape(bs, self.n_iter, 1, 3600).permute(0,1,3,2)
        
        # pred_masks = mask_inputs.contiguous().view(bs, self.n_iter, w_featmap, h_featmap)                   ## shape: [bs, n_iter, 3600, 1]
        # outputs['pred_masks'] = pred_masks
        
        return outputs_class ## shape: [bs, cls, img_h, img_w] == [1, 2, 480, 480]