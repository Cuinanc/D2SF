from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from timm.models.layers import DropPath
from model.modules.graph_frames import Graph
from model.modules.mlp_gcn import Mlp_gcn

from model.modules.attention import Attention
from model.modules.graph import GCN
from model.modules.mlp import MLP
from model.modules.tcn import MultiScaleTCN
from einops import rearrange


class AGFormerBlock(nn.Module):
    """
    Implementation of AGFormer block.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=243):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # 特征混合器：self.mixer 根据 mixer_type 选择特征融合的方法
        if mixer_type == 'attention':
            # 注意力机制：使用 Attention 类来实现空间或时间注意力
            self.mixer = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                   proj_drop=drop, mode=mode)
        elif mixer_type == 'graph':
            # 图卷积网络（GCN）：用于捕捉节点之间的关系，适用于空间或时间特征
            self.mixer = GCN(dim, dim,
                             num_nodes=17 if mode == 'spatial' else n_frames,
                             neighbour_num=neighbour_num,
                             mode=mode,
                             use_temporal_similarity=use_temporal_similarity,
                             temporal_connection_len=temporal_connection_len)
        elif mixer_type == "ms-tcn":
            self.mixer = MultiScaleTCN(in_channels=dim, out_channels=dim)
        else:
            raise NotImplementedError("AGFormer mixer_type is either attention or graph")
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MotionAGFormerBlock(nn.Module):
    # 包含空间和时间分支以及自适应融合机制
    """
    Implementation of MotionAGFormer block. It has two ST and TS branches followed by adaptive fusion.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243):
        super().__init__()
        self.hierarchical = hierarchical
        # 如果 hierarchical 为真，特征维度 dim 将被减半，以便为不同分支处理不同的特征
        dim = dim // 2 if hierarchical else dim

        # ST Attention branch
        self.att_spatial = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                         qk_scale, use_layer_scale, layer_scale_init_value,
                                         mode='spatial', mixer_type="attention",
                                         use_temporal_similarity=use_temporal_similarity,
                                         neighbour_num=neighbour_num,
                                         n_frames=n_frames)
        self.att_temporal = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                          qk_scale, use_layer_scale, layer_scale_init_value,
                                          mode='temporal', mixer_type="attention",
                                          use_temporal_similarity=use_temporal_similarity,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames)

        # ST Graph branch
        if graph_only:
            self.graph_spatial = GCN(dim, dim,
                                     num_nodes=17,
                                     mode='spatial')
            if use_tcn:
                self.graph_temporal = MultiScaleTCN(in_channels=dim, out_channels=dim)
            else:
                self.graph_temporal = GCN(dim, dim,
                                          num_nodes=n_frames,
                                          neighbour_num=neighbour_num,
                                          mode='temporal',
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len)
        else:
            self.graph_spatial = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                               qkv_bias,
                                               qk_scale, use_layer_scale, layer_scale_init_value,
                                               mode='spatial', mixer_type="graph",
                                               use_temporal_similarity=use_temporal_similarity,
                                               temporal_connection_len=temporal_connection_len,
                                               neighbour_num=neighbour_num,
                                               n_frames=n_frames)
            self.graph_temporal = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                                qkv_bias,
                                                qk_scale, use_layer_scale, layer_scale_init_value,
                                                mode='temporal', mixer_type="ms-tcn" if use_tcn else 'graph',
                                                use_temporal_similarity=use_temporal_similarity,
                                                temporal_connection_len=temporal_connection_len,
                                                neighbour_num=neighbour_num,
                                                n_frames=n_frames)

        self.use_adaptive_fusion = use_adaptive_fusion
        # 自适应融合
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.hierarchical:
            B, T, J, C = x.shape
            # 如果使用层次结构，将 x 分为两个部分，前半部分用于注意力机制，后半部分用于图卷积处理
            x_attn, x_graph = x[..., :C // 2], x[..., C // 2:]

            x_attn = self.att_temporal(self.att_spatial(x_attn))
            x_graph = self.graph_temporal(self.graph_spatial(x_graph + x_attn))
        else:
            x_attn = self.att_temporal(self.att_spatial(x))
            x_graph = self.graph_temporal(self.graph_spatial(x))

        if self.hierarchical:
            # 如果使用层次结构，拼接 x_attn 和 x_graph
            x = torch.cat((x_attn, x_graph), dim=-1)
        elif self.use_adaptive_fusion:
            # 如果使用自适应融合，计算两个分支的加权平均，权重通过 self.fusion 得到并通过softmax处理
            alpha = torch.cat((x_attn, x_graph), dim=-1)
            alpha = self.fusion(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2]
        else:
            x = (x_attn + x_graph) * 0.5

        return x


def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                  temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243):
    """
    generates MotionAGFormer layers
    """
    layers = []
    for _ in range(n_layers):
        layers.append(MotionAGFormerBlock(dim=dim,
                                          mlp_ratio=mlp_ratio,
                                          act_layer=act_layer,
                                          attn_drop=attn_drop,
                                          drop=drop_rate,
                                          drop_path=drop_path_rate,
                                          num_heads=num_heads,
                                          use_layer_scale=use_layer_scale,
                                          layer_scale_init_value=layer_scale_init_value,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qkv_scale,
                                          use_adaptive_fusion=use_adaptive_fusion,
                                          hierarchical=hierarchical,
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len,
                                          use_tcn=use_tcn,
                                          graph_only=graph_only,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames))
    layers = nn.Sequential(*layers)

    return layers


class SpatialCGNL(nn.Module):
    # 通过点积核函数来捕捉长距离的空间依赖关系。模块的核心部分是 kernel 函数，它实现了空间的非局部关系。
    """Spatial CGNL block with dot production kernel for image classfication.
    """

    def __init__(self, inplanes, planes, use_scale=False, groups=1):
        # inplanes: 输入通道数，planes: 输出通道数，use_scale: 是否对点积的结果进行缩放，groups: 使用的组数
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNL, self).__init__()
        # conv theta
        self.t = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv1d(planes, inplanes, kernel_size=3, padding=1, stride=1,
                           groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

    def kernel(self, t, p, g, b, c, w):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        # t = t.view(b, 1,c//4, 4  * w)
        # p = p.view(b, 1, c//4, 4  * w)
        # g = g.view(b, c//4, 4 * w, 1)

        # t = t.view(b, c//4, 4  * w)
        # p = p.view(b, c//4, 4  * w)
        # g = g.view(b, 4 * w, c//4)

        t = t.view(b, 1, c * w)
        p = p.view(b, 1, c * w)
        g = g.view(b, c * w, 1)

        att = torch.bmm(p, g)
        # 通过 torch.bmm（批量矩阵乘法）计算点积注意力权重，并根据需要缩放

        if self.use_scale:
            att = att.div((c * w) ** 0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, w)
        return x

    def forward(self, x):
        residual = x
        # t, p, g 是三个通过不同 1D 卷积层生成的特征映射
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


class LPI(nn.Module):
    # 用于在局部窗口内明确地进行 token 之间的交互，从而增强隐式的注意力机制
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0., kernel_size=5, dim=17):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv1d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=1)
        # 1x5 Conv 卷积操作，用于在不同的关节（Joint）之间进行局部信息的交互

        self.act = act_layer()
        # self.bn = nn.SyncBatchNorm(in_features)
        self.gn = nn.GroupNorm(num_groups=out_features, num_channels=in_features)
        self.bn = nn.BatchNorm1d(in_features)
        self.conv2 = torch.nn.Conv1d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=1)

        self.conv3 = torch.nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                     padding=padding, groups=1)
        self.gn1 = nn.GroupNorm(num_groups=out_features, num_channels=out_features)
        # self.gn2 = nn.GroupNorm(num_groups=out_features, num_channels=out_features)
        # self.conv3 = torch.nn.Conv1d(2*in_features, out_features, kernel_size=kernel_size,
        #                              padding=padding, groups=1)

    def forward(self, x):
        # 输入 x 会先进行维度转换，然后经过卷积处理，最后返回与原输入相加的结果
        res = x
        B, N, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.gn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        x += res
        return x


class ATTENTION(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


attns = []


class BLOCK(nn.Module):
    # 融合了注意力机制、全连接层以及局部特征增强模块（如 LPI 和 SpatialCGNL）
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, dim_conv=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ATTENTION(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.causal = TemporalModelOptimized1f(17, dim, 17, 1)
        if dim_conv == 81:
            self.local = SpatialCGNL(dim_conv, int(dim_conv), use_scale=False, groups=3)
        else:
            self.local_mp = LPI(in_features=dim, act_layer=act_layer, dim=dim_conv)

        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(81)

        eta = 1e-5
        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma4 = nn.Parameter(eta * torch.ones(81), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))

        if x.shape[2] == 544:
            # x = x.transpose(-2,-1)
            x = x + self.drop_path(self.gamma3 * self.local(self.norm3(x)))
            # x = x.transpose(-2,-1)
        else:
            x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x)))

        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class Attention1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False,
                 vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        if self.comb == True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb == False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.comb == True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb == False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block1(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention1, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0,
                 depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth > 0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            comb=comb, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.changedim and self.currentdim < self.depth // 2:
            self.reduction = nn.Conv1d(dim, dim // 2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth // 2:
            self.improve = nn.Conv1d(dim, dim * 2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x




class MotionAGFormer(nn.Module):
    """
    MotionAGFormer, the main class of our model.
    """

    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False, graph_only=False,
                 neighbour_num=4, n_frames=243):
        """
        :param n_layers: Number of layers.模型中层的数量
        :param dim_in: Input dimension.输入特征的维度
        :param dim_feat: Feature dimension.特征的维度
        :param dim_rep: Motion representation dimension运动表示的维度，默认为512
        :param dim_out: output dimension. For 3D pose lifting it is set to 3 输出维度
        :param mlp_ratio: MLP ratio.
        :param act_layer: Activation layer.
        :param drop: Dropout rate.
        :param drop_path: Stochastic drop probability.
        :param use_layer_scale: Whether to use layer scaling or not.
        :param layer_scale_init_value: Layer scale init value in case of using layer scaling.
        :param use_adaptive_fusion: Whether to use adaptive fusion or not.
        :param num_heads: Number of attention heads in attention branch
        :param qkv_bias: Whether to include bias in the linear layers that create query, key, and value or not.
        :param qkv_scale: scale factor to multiply after outer product of query and key. If None, it's set to
                          1 / sqrt(dim_feature // num_heads)
        :param hierarchical: Whether to use hierarchical structure or not.
        :param num_joints: Number of joints.
        :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes
        :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
        :param use_tcn: If true, uses MS-TCN for temporal part of the graph branch.
        :param graph_only: Uses GCN instead of GraphFormer in the graph branch.
        :param neighbour_num: Number of neighbors for temporal GCN similarity.
        :param n_frames: Number of frames. Default is 243
        """
        super().__init__()

        # self.joints_embed：一个线性层，将输入特征转换为特征维度
        # self.pos_embed：位置嵌入，用于为关节位置提供额外信息
        # self.norm：层归一化，确保模型的稳定性和收敛速度
        self.joints_embed = nn.Linear(32, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)

        # 通过 create_layers 函数生成多个 MotionAGFormerBlock，实现模型的核心部分
        self.layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical,
                                    use_temporal_similarity=use_temporal_similarity,
                                    temporal_connection_len=temporal_connection_len,
                                    use_tcn=use_tcn,
                                    graph_only=graph_only,
                                    neighbour_num=neighbour_num,
                                    n_frames=n_frames)

        # 表示层：一个包含全连接层和 Tanh 激活的序列，用于生成运动表示
        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, dim_out)

        # 新增
        self.Spatial_patch_to_embedding = nn.Linear(3, 32)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 17, 32))
        self.pos_drop = nn.Dropout(p=0.)
        dpr = [x.item() for x in torch.linspace(0, 0.2, 3)]  # stochastic depth decay rule
        dpr1 = [x.item() for x in torch.linspace(0, 0.2, 4)]  # stochastic depth decay rule
        norm_layer = nn.LayerNorm
        self.Spatial_norm = norm_layer(32)
        self.Temporal_norm = norm_layer(32)

        self.Spatial_blocks = nn.ModuleList([
            BLOCK(
                dim=32, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer, dim_conv=17)
            for i in range(3)])

        self.TTEblocks = nn.ModuleList([
            Block1(
                dim=32, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr1[i + 1], norm_layer=norm_layer, comb=False,
                changedim=False, currentdim=i + 1, depth=4)
            for i in range(3)])

        # # 处理多个帧的2D嵌入映射
        # self.graph = Graph('hm36_gt', 'spatial', pad=1)
        # self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False)
        #
        # self.embedding = nn.Linear(3, 512)
        # self.mlp_gcn = Mlp_gcn(3, 512, 1024, 256, self.A, length=17,
        #                        frames=n_frames)
        # self.head = nn.Linear(512, 3)

    # 空间维度上的特征处理
    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        # 输入形状为[b, c, f, p]
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)

        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        # x = x.view(b,f,17,-1)

        return x

    # 时间维度上的特征处理
    def forward_features1(self, x):
        b, f, cw = x.shape
        x = x.view(b, f, 17, -1)
        assert len(x.shape) == 4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(3):
            x = rearrange(x, 'b f n cw -> (b n) f cw', f=f)
            tteblock = self.TTEblocks[i]

            # x += self.Temporal_pos_embed
            # x = self.pos_drop(x)
            # if i==7:
            #     x = tteblock(x, vis=True)
            #     exit()
            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)

            # x = rearrange(x, 'b f n cw -> (b n) f cw', n=n)
            # x = self.weighted_mean(x)
            # x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
            # x = x.view(b, f, -1)
        return x

    def forward(self, x, return_rep=False):
        """
        :param x: tensor with shape [B, T, J, C] (T=243, J=17, C=3)
        :param return_rep: Returns motion representation feature volume (In case of using this as backbone)
        """
        # 新增(CFI和CJI)
        attns.clear()
        x1 = x
        # 调整维度，把通道维度放在第二位
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape

        x[:, :, 10:16] = 0

        x = self.Spatial_forward_features(x)

        x2 = x
        x = self.forward_features1(x)

        # 新增

        x = self.joints_embed(x)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x

        x = self.head(x)


        return x


def _test():
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 81, 17
    random_x = torch.randn((b, t, j, c)).to('cuda')

    model = MotionAGFormer(n_layers=26, dim_in=3, dim_feat=64, mlp_ratio=4, hierarchical=False,
                           use_tcn=False, graph_only=False, n_frames=t).to('cuda')
    model.eval()

    # 1. 模型参数量
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f"Model parameter #: {model_params:,}")

    # 2. 总 MACs
    total_macs = profile_macs(model, random_x)
    print(f"Model MACs #: {total_macs:,}")

    # 3. 每帧 MACs
    macs_per_frame = total_macs / t
    print(f"MACs per frame: {macs_per_frame:,}")

    # Warm-up 推理
    for _ in range(10):
        _ = model(random_x)

    # 推理速度（FPS）
    import time
    num_iterations = 100
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    average_inference_time = (end_time - start_time) / num_iterations
    fps = 1.0 / average_inference_time
    print(f"FPS: {fps}")

    # 输出验证
    out = model(random_x)
    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"

    # model_params = 0
    # for parameter in model.parameters():
    #     model_params = model_params + parameter.numel()
    # print(f"Model parameter #: {model_params:,}")
    # print(f"Model FLOPS #: {profile_macs(model, random_x):,}")
    #
    # # Warm-up to avoid timing fluctuations
    # for _ in range(10):
    #     _ = model(random_x)
    #
    # import time
    # num_iterations = 100
    # # Measure the inference time for 'num_iterations' iterations
    # start_time = time.time()
    # for _ in range(num_iterations):
    #     with torch.no_grad():
    #         _ = model(random_x)
    # end_time = time.time()
    #
    # # Calculate the average inference time per iteration
    # average_inference_time = (end_time - start_time) / num_iterations
    #
    # # Calculate FPS
    # fps = 1.0 / average_inference_time
    #
    # print(f"FPS: {fps}")
    #
    # out = model(random_x)
    #
    # assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()
