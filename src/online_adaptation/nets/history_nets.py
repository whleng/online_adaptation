from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from rpad.pyg.nets import dgcnn as pnd
from rpad.pyg.nets import pointnet2 as pnp
from torch_geometric.data import Data

"""
Additional networks required for pnp latent encoding and transformer
"""


class PN2DenseLatentEncoding(nn.Module):
    def __init__(
        self,
        flow_embed_dim,
        in_channels: int = 0,
        out_channels: int = 3,
        p: pnp.PN2DenseParams = pnp.PN2DenseParams(),
    ):
        super().__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels

        # Construct the set aggregation modules. This is the encoder.
        self.sa1 = pnp.SAModule(3 + self.in_ch, p.sa1_outdim, p=p.sa1)
        self.sa2 = pnp.SAModule(3 + p.sa1_outdim, p.sa2_outdim, p=p.sa2)
        self.sa3 = pnp.GlobalSAModule(3 + p.sa2_outdim, p.gsa_outdim, p=p.gsa)

        # The Feature Propagation modules. This is the decoder.
        self.fp3 = pnp.FPModule(
            p.gsa_outdim + flow_embed_dim + p.sa2_outdim, p.sa2_outdim, p.fp3
        )
        self.fp2 = pnp.FPModule(p.sa2_outdim + p.sa1_outdim, p.sa1_outdim, p.fp2)
        self.fp1 = pnp.FPModule(p.sa1_outdim + in_channels, p.fp1_outdim, p.fp1)

        # Final linear layers at the output.
        self.lin1 = torch.nn.Linear(p.fp1_outdim, p.lin1_dim)
        self.lin2 = torch.nn.Linear(p.lin1_dim, p.lin2_dim)
        self.lin3 = torch.nn.Linear(p.lin2_dim, out_channels)
        self.out_act = p.out_act

    def forward(self, data: Data, latents):
        sa0_out = (data.x, data.pos, data.batch)

        # Encode.
        sa1_out = self.sa1(*sa0_out)
        sa2_out = self.sa2(*sa1_out)
        x3, pos3, batch3 = self.sa3(*sa2_out)

        x3 = torch.cat([x3, latents], dim=-1)
        sa3_out = x3, pos3, batch3

        # Decode.
        fp3_out = self.fp3(*sa3_out, *sa2_out)
        fp2_out = self.fp2(*fp3_out, *sa1_out)
        x, _, _ = self.fp1(*fp2_out, *sa0_out)

        # Final layers.
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = self.lin3(x)

        if self.out_act != "none":
            raise ValueError()

        return x


class PN2DenseLatentEncodingEverywhere(nn.Module):
    def __init__(
        self,
        flow_embed_dim,
        in_channels: int = 0,
        out_channels: int = 3,
        p: pnp.PN2DenseParams = pnp.PN2DenseParams(),
    ):
        super().__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        # Construct the set aggregation modules. This is the encoder.
        self.sa1 = pnp.SAModule(3 + self.in_ch, p.sa1_outdim, p=p.sa1)
        self.sa2 = pnp.SAModule(3 + p.sa1_outdim, p.sa2_outdim, p=p.sa2)
        self.sa3 = pnp.GlobalSAModule(3 + p.sa2_outdim, p.gsa_outdim, p=p.gsa)

        # The Feature Propagation modules. This is the decoder.
        self.fp3 = pnp.FPModule(p.gsa_outdim + p.sa2_outdim, p.sa2_outdim, p.fp3)
        self.fp2 = pnp.FPModule(p.sa2_outdim + p.sa1_outdim, p.sa1_outdim, p.fp2)
        self.fp1 = pnp.FPModule(p.sa1_outdim + in_channels, p.fp1_outdim, p.fp1)

        # Linear projection layers to incorporate the flwo embedding.
        # Option: could add relu?
        self.global_linear = torch.nn.Linear(flow_embed_dim, p.gsa_outdim)
        self.fp3_embedding_linear = torch.nn.Linear(flow_embed_dim, p.sa2_outdim)
        self.fp2_embedding_linear = torch.nn.Linear(flow_embed_dim, p.sa1_outdim)
        self.fp1_embedding_linear = torch.nn.Linear(flow_embed_dim, p.fp1_outdim)

        # Final linear layers at the output.
        self.lin1 = torch.nn.Linear(p.fp1_outdim, p.lin1_dim)
        self.lin2 = torch.nn.Linear(p.lin1_dim, p.lin2_dim)
        self.lin3 = torch.nn.Linear(p.lin2_dim, out_channels)
        self.out_act = p.out_act

    def forward(self, data: Data, latents):
        sa0_out = (data.x, data.pos, data.batch)
        # Encode.
        sa1_out = self.sa1(*sa0_out)
        sa2_out = self.sa2(*sa1_out)
        x3, pos3, batch3 = self.sa3(*sa2_out)

        # No concatenation! just hadamard!
        x3 = self.global_linear(latents) * x3
        sa3_out = x3, pos3, batch3

        # Decode.
        x_fp3, pos_fp3, batch_fp3 = self.fp3(*sa3_out, *sa2_out)
        fp3_latents = self.fp3_embedding_linear(latents)
        x_fp3 = fp3_latents.repeat_interleave(torch.bincount(batch_fp3), dim=0) * x_fp3
        fp3_out = x_fp3, pos_fp3, batch_fp3

        x_fp2, pos_fp2, batch_fp2 = self.fp2(*fp3_out, *sa1_out)
        fp2_latents = self.fp2_embedding_linear(latents)
        x_fp2 = fp2_latents.repeat_interleave(torch.bincount(batch_fp2), dim=0) * x_fp2
        fp2_out = x_fp2, pos_fp2, batch_fp2

        x, _, batch_fp1 = self.fp1(*fp2_out, *sa0_out)
        fp1_latents = self.fp1_embedding_linear(latents)
        x = fp1_latents.repeat_interleave(torch.bincount(batch_fp1), dim=0) * x

        # Final layers.
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = self.lin3(x)

        if self.out_act != "none":
            raise ValueError()

        return x


DenseParams = Union[pnp.PN2DenseParams, pnd.DGCNNDenseParams]


class ArtFlowNetHistoryParams:
    net: DenseParams = pnp.PN2DenseParams()
    mask_output_flow: bool = False
    encoder_dim: int = 128


def create_flownet(
    in_channels=0,
    out_channels=3,
    p: DenseParams = pnp.PN2DenseParams(),
    flow_embed_dim=None,
) -> Union[pnp.PN2Dense, pnd.DGCNNDense]:
    if isinstance(p, pnp.PN2DenseParams):
        return PN2DenseLatentEncodingEverywhere(
            flow_embed_dim, in_channels, out_channels, p
        )
    elif isinstance(p, pnd.DGCNNDenseParams):
        return pnd.DGCNNDense(in_channels, out_channels, p)
    else:
        raise ValueError(f"invalid model type: {type(p)}")
