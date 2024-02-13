import omegaconf
import torch
import torch_geometric.data as tgd

from online_adaptation.models.history_tformer import (
    FlowHistoryTformerPredictorTrainingModule,
)
from online_adaptation.nets.history_nets import *

DenseParams = Union[pnp.PN2DenseParams, pnd.DGCNNDenseParams]


class ArtFlowNetHistoryParams:
    net: DenseParams = pnp.PN2DenseParams()
    mask_output_flow: bool = False
    encoder_dim: int = 128


def generate_data(B=3):
    data = []

    for i in range(B):
        N = torch.randint(1000, tuple())

        d = tgd.Data(
            x=None,
            curr_pos=torch.rand(N, 3),
        )

        history = []
        flow_history = []
        length = torch.randint(3, 10, tuple())  # num of observations

        for num_obs in range(length):
            pos = torch.rand(N, 3)
            flow = torch.rand(N, 3)
            history = torch.cat((history, pos), axis=0) if num_obs > 0 else pos
            flow_history = (
                torch.cat((flow_history, flow), axis=0) if num_obs > 0 else flow
            )

        d.history = history  # (num_obs x N, 3)
        d.flow_history = flow_history  # (num_obs x N, 3)
        d.K = torch.tensor(length)
        d.lengths = torch.as_tensor([N] * length)  # number of points
        data.append(d)

    batch = tgd.Batch.from_data_list(data)
    return batch


def test_transformer():
    batch = generate_data(B=3).cuda()

    vect_conf = omegaconf.OmegaConf.create(
        {"lr": 1e-4, "batch_size": 3, "mask_input_channel": None, "vectorize": True}
    )
    novect_conf = omegaconf.OmegaConf.create(
        {"lr": 1e-4, "batch_size": 3, "mask_input_channel": None, "vectorize": False}
    )

    v_model = FlowHistoryTformerPredictorTrainingModule(
        None, training_cfg=vect_conf
    ).cuda()
    n_model = FlowHistoryTformerPredictorTrainingModule(
        None, training_cfg=novect_conf
    ).cuda()

    torch.manual_seed(0)
    v_preds = v_model(batch)
    torch.manual_seed(0)
    n_preds = n_model(batch)

    v_flow = v_preds["pred_flow"]
    n_flow = n_preds["pred_flow"]

    v_latent = v_preds["latents"]
    n_latent = n_preds["latents"]

    v_prev = v_preds["results"]
    n_prev = n_preds["results"]

    v_inputs = v_preds["inputs"]
    n_inputs = n_preds["inputs"]
    # breakpoint()

    # assert torch.allclose(v_preds["pred_flow"], n_preds["pred_flow"])
