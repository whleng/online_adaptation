from typing import Union

import part_embedding.nets.dgcnn as pnd
import part_embedding.nets.pointnet2 as pnp
import torch
import torch_geometric.data as tgd

from online_adaptation.nets.history_tformer import ArtFlowNetHistoryEncodingModel

DenseParams = Union[pnp.PN2DenseParams, pnd.DGCNNDenseParams]


class ArtFlowNetHistoryParams:
    net: DenseParams = pnp.PN2DenseParams()
    mask_output_flow: bool = False
    encoder_dim: int = 128


def test_transformer():
    num_obs = 2
    num_history = 2

    # Create some random data ofa  point cloud w/ history
    current_pc = []
    history_latents = []
    for _ in range(num_obs):
        current_pc.append(torch.Tensor([[1, 2, 3], [4, 5, 6]]))
        curr_history = []
        for _ in range(num_history):
            curr_history.append(torch.Tensor([[8, 9, 10], [11, 12, 13]]))
        history_latents.append(torch.Tensor(curr_history))

    data_list = [current_pc, history_latents]
    batch = tgd.Batch.from_data_list(data_list)

    # Create your model
    net_params = pnp.PN2DenseParams()
    model = ArtFlowNetHistoryEncodingModel(
        ArtFlowNetHistoryParams(net=net_params, mask_output_flow=False),
    )

    # Pass the data through the model
    preds = model(batch)

    print(preds)

    # MAke sure everything works.


test_transformer()
