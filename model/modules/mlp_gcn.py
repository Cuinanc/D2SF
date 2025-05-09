from functools import partial
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Gcn(nn.Module):
#图卷积网络，将邻接矩阵adj应用发哦特征上，表示关节之间的关系
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.adj = adj
        self.kernel_size = adj.size(0)
        self.conv = nn.Conv2d(in_channels, out_channels * self.kernel_size, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv, kvw->nctw', (x, self.adj))

        return x.contiguous()


class Mlp(nn.Module):
#标准的多层感知机，用于通道数上的全连接操作
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Mlp_ln(nn.Module):
#MLP的一个扩展版本，除了全连接层，还加入了LN操作
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features)
        )
        
        self.act = act_layer()

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.LayerNorm(out_features)
        )

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
#将GCN和MLP结合在一起，构成模型的基本单元
    def __init__(self, length, frames, dim, tokens_dim, channels_dim, adj, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(length)

        self.gcn_1 = Gcn(dim, dim, adj)
        self.gcn_2 = Gcn(dim, dim, adj)
        self.adj = adj

        if frames == 1:
            self.mlp_1 = Mlp(in_features=length, hidden_features=tokens_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp_1 = Mlp_ln(in_features=length, hidden_features=tokens_dim, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_2 = Mlp(in_features=dim, hidden_features=channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        ## Spatial Graph MLP
        #空间图卷积和MLP，使用GCN处理空间信息，并将特征经过MLP和GCN之后叠加
        x = rearrange(x, f'b j c -> b c j') 
        res = x
        x = self.norm1(x)

        x_gcn_1 = rearrange(x, 'b c j-> b c 1 j') 
        x_gcn_1 = self.gcn_1(x_gcn_1)
        x_gcn_1 = rearrange(x_gcn_1, 'b c 1 j -> b c j') 

        x = res + self.drop_path(self.mlp_1(x) + x_gcn_1)
        
        ## Channel Graph MLP
        #通道图卷积和MLP，处理通道维度上的图卷积，在经过MLP和GCN之后叠加
        x = rearrange(x, f'b c j -> b j c') 
        res = x
        x = self.norm2(x)

        x_gcn_2 = rearrange(x, 'b j c-> b c 1 j') 
        x_gcn_2 = self.gcn_2(x_gcn_2)
        x_gcn_2 = rearrange(x_gcn_2, 'b c 1 j -> b j c') 

        x = res + self.drop_path(self.mlp_2(x) + x_gcn_2)

        return x


class Mlp_gcn(nn.Module):
#创建多个Block，每个block负责处理空间和通道的信息
    def __init__(self, depth, embed_dim, channels_dim, tokens_dim, adj, drop_rate=0.10, length=17, frames=1):
        super().__init__()
        drop_path_rate = 0.2

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(
                length, frames, embed_dim, tokens_dim, channels_dim, adj, 
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

