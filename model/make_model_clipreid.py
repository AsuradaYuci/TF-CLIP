import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from collections import OrderedDict
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from .clip.model import QuickGELU, LayerNorm
# from .TAT import TemporalAttentionTransformer
from .Visual_Prompt import visual_prompt


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    model_path = clip._download(url)  #
    # model_path = '/dataset_cc/Pretrain-models/ViT-B-16.pt'  # 不用下载,用下载好的
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class CrossFramelAttentionBlock(nn.Module):  # 跨帧注意力
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath=0., T=0, ):
        super().__init__()
        self.T = T

        self.message_fc = nn.Linear(d_model, d_model)  # 768 ->768  用一个FC得到信息token
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head, )

        self.attn = nn.MultiheadAttention(d_model, n_head, )
        self.ln_1 = LayerNorm(d_model)

        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))  # mlp
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        l, bt, d = x.size()  # 197, 32, 768
        b = bt // self.T  # 4
        x = x.view(l, b, self.T, d)  # torch.Size([197, 4, 8, 768])
        # x_cls = [4, 8, 768]
        ######## 1.TMC  #####################
        msg_token = self.message_fc(x.mean(0))  # torch.Size([4, 8, 768])
        # msg_token = x.mean(0)  # torch.Size([4, 8, 768])
        msg_token = msg_token.view(b, self.T, 1, d)  # torch.Size([4, 8, 1, 768])

        msg_token = msg_token.permute(1, 2, 0, 3).view(self.T, b, d)  # torch.Size([8, 4, 768])
        msg_token = msg_token + self.drop_path(
            self.message_attn(self.message_ln(msg_token), self.message_ln(msg_token), self.message_ln(msg_token),
                              need_weights=False)[0])
        msg_token = msg_token.view(self.T, 1, b, d).permute(1, 2, 0, 3)  # torch.Size([1, 4, 8, 768])

        x = torch.cat([x, msg_token], dim=0)  # torch.Size([198, 4, 8, 768])
        ########  2.MD ###################
        x = x.view(l + 1, -1, d)  # torch.Size([198, 32, 768])
        x = x + self.drop_path(self.attention(self.ln_1(x)))  # torch.Size([198, 32, 768])
        # x = x[:l, :, :]  # torch.Size([197, 128, 768])
        x = x + self.drop_path(self.mlp(self.ln_2(x)))  # torch.Size([197, 32, 768])
        return x


class Temporal_Memory_Difusion(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None,
                T=8):
        super().__init__()
        # self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)]
        self.width = width
        self.layers = layers

        self.resblocks = nn.Sequential(
            *[CrossFramelAttentionBlock(width, heads, attn_mask, droppath[i], T) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE    # 1

        self.classifier2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        self.classifier_proj_temp = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_proj_temp.apply(weights_init_classifier)
        self.classifier_proj_temp2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_proj_temp2.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.bottleneck_proj_temp = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_temp.bias.requires_grad_(False)
        self.bottleneck_proj_temp.apply(weights_init_kaiming)

        self.bottleneck_proj_temp2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_temp2.bias.requires_grad_(False)
        self.bottleneck_proj_temp2.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.SSP = ImageSpecificPrompt()
        # self.TAT = TemporalAttentionTransformer(T=cfg.INPUT.SEQ_LEN, embed_dim=512, layers=1)
        # width=768  layers=12  heads=12, droppath=None, use_checkpoint, T=8
        # self.SAT = Transformer_SP(width=512, layers=1, heads=8, droppath=None, T=cfg.INPUT.SEQ_LEN)
        # "meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls"
        # self.temppool = visual_prompt(sim_head='meanP', T=cfg.INPUT.SEQ_LEN)
        self.TMD = Temporal_Memory_Difusion(width=768, layers=1, heads=12, droppath=None, T=cfg.INPUT.SEQ_LEN)
        # self.ln_post = LayerNorm(512)


    def forward(self, x = None, get_image = False, cam_label= None, view_label=None, text_features2=None):

        B, T, C, H, W = x.shape  # B=64, T=4.C=3 H=256,W=128

        if get_image == True:
            x = x.view(-1, C, H, W)  # 256,3,256,128

            image_features, image_features_proj = self.image_encoder(x)

            if self.model_name == 'RN50':
                img_feature_proj = image_features_proj[0]
                img_feature_proj = img_feature_proj.view(B, T, -1)
                img_feature_proj = img_feature_proj.mean(1)
                return img_feature_proj

            elif self.model_name == 'ViT-B-16':
                img_feature_proj = image_features_proj[:,0]
                img_feature_proj = img_feature_proj.view(B, T, -1)  # torch.Size([64, 4, 512])
                img_feature_proj = img_feature_proj.mean(1)  # torch.Size([64, 512])
                return img_feature_proj
        
        if self.model_name == 'RN50':
            x = x.view(-1, C, H, W)

            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

            ###################################################
            img_feature_last = img_feature_last.view(B, T, -1)
            img_feature = img_feature.view(B, T, -1)
            img_feature_proj = img_feature_proj.view(B, T, -1)
            
            img_feature_last = img_feature_last.mean(1)
            img_feature = img_feature.mean(1)
            img_feature_proj = img_feature_proj.mean(1)
            ###################################################

        elif self.model_name == 'ViT-B-16':
            x = x.view(-1, C, H, W)  # torch.Size([64, 3, 256, 128])

            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:  # 1
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            # cv_embed = cv_embed.repeat((1, B)).view(B, -1)  # torch.Size([64, 768])
            # cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)  # torch.Size([64, 768])
            cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)  # torch.Size([64, 768])
            # torch.Size([64, 129, 768])  torch.Size([64, 129, 768])  torch.Size([64, 129, 512])
            image_features, image_features_proj_raw = self.image_encoder(x, cv_embed)
            # image_features_SAT: torch.Size([129, 64, 768])

            ###################################################
            img_feature = image_features[:, 0]  # torch.Size([64, 768])
            img_feature_proj = image_features_proj_raw[:, 0]  # torch.Size([64, 512])

            ###################################################
            img_feature = img_feature.view(B, T, -1)  # torch.Size([16, 4, 768])
            img_feature_proj = img_feature_proj.view(B, T, -1)  # # torch.Size([16, 4, 512])
            # f_tp = self.temppool(img_feature_proj)  # b, 512
            img_feature = img_feature.mean(1)  # torch.Size([16, 768])
            img_feature_proj = img_feature_proj.mean(1)  # torch.Size([16, 512])
            ###################################################
            ft_for_another_branch = image_features.detach()
            image_features_SAT = ft_for_another_branch.permute(1, 0, 2)  # BT, 768

            f_sp = self.TMD(image_features_SAT)  # torch.Size([130, 64, 768])
            f_sp2 = f_sp.permute(1, 0, 2)  # torch.Size([64, 130, 768])

            # cls_f_sp = self.ln_post(f_sp2.mean(1))  # torch.Size([64, 512])
            cls_f_sp = f_sp2.mean(1)  # torch.Size([64, 768])
            cls_f_sp_tap = cls_f_sp.view(B, T, -1)
            cls_f_tp = cls_f_sp_tap.mean(1)   #


        feat = self.bottleneck(img_feature)  # torch.Size([16, 768])
        feat_proj = self.bottleneck_proj(img_feature_proj)  # torch.Size([16, 512])
        feat_proj_frame = self.bottleneck_proj_temp(cls_f_sp)
        feat_proj_temp = self.bottleneck_proj_temp2(cls_f_tp)

        if self.training:
            text_features2 = text_features2.unsqueeze(0).expand(B, -1, -1)  # torch.Size([b, 150, 512])
            image_features_proj_raw2 = image_features_proj_raw.view(B, T, -1, image_features_proj_raw.shape[-1])
            video_feature_project = image_features_proj_raw2.mean(1)
            text_features2 = text_features2 + self.SSP(text_features2,
                                                                     video_feature_project)  # torch.Size([8, 625, 512])
            logits = torch.einsum("bd,bkd->bk", img_feature_proj, text_features2)  # torch.Size([4, 101])

            cls_score = self.classifier2(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            cls_score_proj_frame = self.classifier_proj_temp(feat_proj_frame)
            cls_score_proj_temp = self.classifier_proj_temp(feat_proj_temp)
            return [cls_score, cls_score_proj, cls_score_proj_temp, cls_score_proj_frame], [img_feature, img_feature_proj, cls_f_tp], logits

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj, cls_f_tp], dim=1)
                # return f_tp


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PromptGeneratorLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.1,
    ):
        super().__init__()
        self.self_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, visual, visual)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ImageSpecificPrompt(nn.Module):
    def __init__(self, layers=2, embed_dim=512, alpha=0.1, ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)  # 512
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.decoder = nn.ModuleList([PromptGeneratorLayer(embed_dim, embed_dim // 64) for _ in range(layers)])  # 2层
        self.alpha = nn.Parameter(torch.ones(embed_dim) * alpha)  # torch.Size([512])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):  # text: torch.Size([64, 150, 512]) visual: torch.Size([64, 129, 512])
        # B, N, C = visual.shape
        visual = self.memory_proj(visual)  # torch.Size([8, 129, 512])
        text = self.text_proj(text)  # torch.Size([8, 625, 512])
        # visual = self.norm(visual)  # torch.Size([4, 196, 512])  torch.Size([8, 129, 512])
        for layer in self.decoder:
            text = layer(text, visual)  # torch.Size([64, 150, 512])
        text = self.out_proj(text)
        return text