import torch
import torch.nn as nn
import torch.nn.functional as F

from zoedepth.models.zoedepth_rgbt.linear_attention import LocalFeatureTransformer, LocalFeatureCrossAttn

class attension_fusion_block(nn.Module):
    def __init__(self, in_channels, n_layer=2):
        """
        Args:
            in_channels (int): list of input channels
            n_layers (int): layers of transformer attention. Defaults to 2.
        """
        super().__init__()
        self.attentions = nn.ModuleList([
            LocalFeatureTransformer(['self', 'cross'], n_layers=n_layer, d_model=in_channel, attention='full') for in_channel in in_channels
        ])

    def forward(self, xs, ys, return_class_token=False):
        '''
        Args:
            xs list of [torch.Tensor]: [N, L, C]
            ys list of [torch.Tensor]: [N, S, C]
        '''
        assert len(xs) == len(ys) == len(self.attentions)
        xs_out = []
        ys_out = []

        if return_class_token:
            for i, (x, y, attention) in enumerate(zip(xs, ys, self.attentions)):
                xout, yout = attention(x[0], y[0])
                xs_out.append((xout, x[1]))
                ys_out.append((yout, y[1]))
            return tuple(xs_out), tuple(ys_out)
        else:
            for i, (x, y, attention) in enumerate(zip(xs, ys, self.attentions)):
                x, y = attention(x, y)
                xs_out.append(x)
                ys_out.append(y)
            return tuple(xs_out), tuple(ys_out)



class simple_crossattn_fusion_block(nn.Module):
    def __init__(self, in_channels, n_layer=2):
        """
        Args:
            in_channels (int): list of input channels
            n_layers (int): layers of transformer attention. Defaults to 2.
        """
        super().__init__()
        self.in_channels = in_channels
        self.leaners = nn.ModuleList()
        for i in range(len(self.in_channels)):
            hiden_size = self.in_channels[i]
            self.leaners.append(nn.Sequential(nn.Linear(hiden_size, hiden_size), *([nn.Linear(hiden_size, hiden_size)]*(n_layer-1))))

        self.cross_attentions = nn.ModuleList([
            LocalFeatureCrossAttn(['cross'], n_layers=n_layer, d_model=in_channel, attention='full') for in_channel in in_channels
        ])

    def forward(self, xs, ys, return_class_token=False):
        '''
        Args:
            xs list of [torch.Tensor]: [N, L, C]
            ys list of [torch.Tensor]: [N, S, C]
        '''
        assert len(xs) == len(ys) == len(self.cross_attentions)
        xs_out = []
        ys_out = []

        if return_class_token:
            for i, (x, y, attention) in enumerate(zip(xs, ys, self.cross_attentions)):
                x0, y0 = x[0], y[0]
                feat_learner = F.relu(self.leaners[i](x0))
                fused_y0 = attention(y0, feat_learner)
                xs_out.append((x0 + feat_learner, x[1]))
                ys_out.append((fused_y0, y[1]))
            return tuple(xs_out), tuple(ys_out)
        else:
            for i, (x, y, attention) in enumerate(zip(xs, ys, self.cross_attentions)):
                feat_learner = F.relu(self.leaners[i](x))
                fused_y = attention(y, feat_learner) # first is the query, second is the key and value
                xs_out.append(x + feat_learner)
                ys_out.append(fused_y)
            return tuple(xs_out), tuple(ys_out)

# Refer to 2DPASS/network/arch_2dpass.py
class twopass_fusion_block(nn.Module):
    def __init__(self, in_channels, n_layer=1):
        """
        Args:
            in_channels ([int]): list of input channels
            n_layers (int): layers of transformer attention. Defaults to 2.
        """
        super().__init__()
        self.in_channels = in_channels
        self.leaners = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        for i in range(len(self.in_channels)):
            hiden_size = self.in_channels[i]
            self.leaners.append(nn.Sequential(nn.Linear(hiden_size, hiden_size), *([nn.Linear(hiden_size, hiden_size)]*(n_layer-1))))
            self.fcs1.append(nn.Sequential(nn.Linear(hiden_size * 2, hiden_size), *([nn.Linear(hiden_size, hiden_size)]*(n_layer-1))))
            self.fcs2.append(nn.Sequential(nn.Linear(hiden_size, hiden_size), *([nn.Linear(hiden_size, hiden_size)]*(n_layer-1))))

    def forward(self, xs, ys, return_class_token=False):
        '''
        Args:
            xs list of [torch.Tensor]: [N, L, C]; thermal features
            ys list of [torch.Tensor]: [N, S, C]; rgb features
        '''
        assert len(xs) == len(ys) == len(self.leaners)
        xs_out = []
        ys_out = []

        if return_class_token:
            for i, (x, y) in enumerate(zip(xs, ys)):
                x0 = x[0]
                y0 = y[0]
                feat_learner = F.relu(self.leaners[i](x0))
                feat_cat = torch.cat([y0, feat_learner], dim=-1)
                feat_cat = self.fcs1[i](feat_cat)
                feat_weight = torch.sigmoid(self.fcs2[i](feat_cat))
                fuse_feat = F.relu(feat_cat * feat_weight)

                ### My default
                xs_out.append((x0 + feat_learner, x[1]))
                ys_out.append((y0 + fuse_feat, y[1]))

                ### original 2Dpass
                # xs_out.append((x0, x[1]))
                # ys_out.append((fuse_feat, y[1]))
            return tuple(xs_out), tuple(ys_out)
        else:
            for i, (x, y) in enumerate(zip(xs, ys)):
                feat_learner = F.relu(self.leaners[i](x))
                feat_cat = torch.cat([y, feat_learner], dim=-1)
                feat_cat = self.fcs1[i](feat_cat)
                feat_weight = torch.sigmoid(self.fcs2[i](feat_cat))
                fuse_feat = F.relu(feat_cat * feat_weight)

                ### My default
                xs_out.append(x+feat_learner)
                ys_out.append(y+fuse_feat)

                ### original 2Dpass
                # xs_out.append(x)
                # ys_out.append(fuse_feat)

            return tuple(xs_out), tuple(ys_out)


class simple_twopass_fusion_block(nn.Module):
    def __init__(self, in_channels, n_layer=1):
        """
        Args:
            in_channels ([int]): list of input channels
            n_layers (int): layers of transformer attention. Defaults to 2.
        """
        super().__init__()
        self.in_channels = in_channels
        self.leaners = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        for i in range(len(self.in_channels)):
            hiden_size = self.in_channels[i]
            self.leaners.append(nn.Sequential(nn.Linear(hiden_size, hiden_size), *([nn.Linear(hiden_size, hiden_size)]*(n_layer-1))))
            self.fcs1.append(nn.Sequential(nn.Linear(hiden_size * 2, hiden_size), *([nn.Linear(hiden_size, hiden_size)]*(n_layer-1))))
            self.fcs2.append(nn.Sequential(nn.Linear(hiden_size, hiden_size), *([nn.Linear(hiden_size, hiden_size)]*(n_layer-1))))

    def forward(self, xs, ys, return_class_token=False):
        '''
        Args:
            xs list of [torch.Tensor]: [N, L, C]; thermal features
            ys list of [torch.Tensor]: [N, S, C]; rgb features
        '''
        assert len(xs) == len(ys) == len(self.leaners)
        xs_out = []
        ys_out = []

        if return_class_token:
            for i, (x, y) in enumerate(zip(xs, ys)):
                x0 = x[0]
                y0 = y[0]
                feat_learner = F.relu(self.leaners[i](x0))
                feat_cat = torch.cat([y0, feat_learner], dim=-1)
                feat_cat = self.fcs1[i](feat_cat)
                feat_weight = torch.sigmoid(self.fcs2[i](feat_cat))
                fuse_feat = F.relu(feat_learner * feat_weight) # feat_cat -> feat_learner

                ### My default
                xs_out.append((x0 + feat_learner, x[1]))
                ys_out.append((y0 + fuse_feat, y[1]))

                ### original 2Dpass
                # xs_out.append((x0, x[1]))
                # ys_out.append((fuse_feat, y[1]))

            return tuple(xs_out), tuple(ys_out)
        else:
            for i, (x, y) in enumerate(zip(xs, ys)):
                feat_learner = F.relu(self.leaners[i](x))
                feat_cat = torch.cat([y, feat_learner], dim=-1)
                feat_cat = self.fcs1[i](feat_cat)
                feat_weight = torch.sigmoid(self.fcs2[i](feat_cat))
                fuse_feat = F.relu(feat_learner * feat_weight) # feat_cat -> feat_learner

                ### My default
                xs_out.append(x+feat_learner)
                ys_out.append(y+fuse_feat)

                ### original 2Dpass
                # xs_out.append(x)
                # ys_out.append(fuse_feat)

            return tuple(xs_out), tuple(ys_out)
