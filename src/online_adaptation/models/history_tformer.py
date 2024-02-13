# Migrated from discriminative embeddings

from typing import Any, Dict

import lightning as L
import plotly.graph_objects as go
import rpad.visualize_3d.plots as v3p
import torch
import torch_geometric.data as tgd
from flowbot3d.models.artflownet import artflownet_loss, flow_metrics
from plotly.subplots import make_subplots
from torch import optim

from online_adaptation.nets.history_nets import *


def get_history_batch(batch):
    """Extracts a single batch of the history data for encoding,
    because each history element is processed separately."""
    # history_datas = []
    # for data in batch.to_data_list():
    #     for i in range(data.history.shape[0]):
    #         history_datas.append(
    #             tgd.Data(
    #                 x=data.flow_history[i],
    #                 pos=data.history[i],
    #             )
    #         )

    history_datas = []
    for data in batch.to_data_list():
        history_data = []
        # Get start/end positions based on lengths.

        # HACK: remove once the data has been regenerated...
        if len(data.history.shape) == 3:
            data.history = data.history.reshape(-1, 3)
            data.flow_history = data.flow_history.reshape(-1, 3)

        N = data.curr_pos.shape[0]  # num of points
        if hasattr(data, "lengths"):
            ixs = [0] + data.lengths.cumsum(0).tolist()
        else:
            ixs = [(i * N) for i in range(data.K + 1)]
        # breakpoint()
        for i in range(len(ixs) - 1):
            history_data.append(
                tgd.Data(
                    x=data.flow_history[ixs[i] : ixs[i + 1]],
                    pos=data.history[ixs[i] : ixs[i + 1]],
                )
            )
        # print(history_datas)

        history_datas.extend(history_data)

    return tgd.Batch.from_data_list(history_datas)


def history_latents_to_nested_list(batch, history_latents):
    """Converting history latents from stacked form to nested list"""
    datas = batch.to_data_list()
    history_lengths = [0] + [data.K.item() for data in datas]
    ixs = torch.tensor(history_lengths).cumsum(0).tolist()
    post_encoder_latents = []
    for i, data in enumerate(datas):
        post_encoder_latents.append(history_latents[ixs[i] : ixs[i + 1]])

    return post_encoder_latents


# Flow predictor
class FlowHistoryTformerPredictorTrainingModule(L.LightningModule):
    def __init__(self, network, training_cfg) -> None:
        super().__init__()
        # self.network = network

        in_dim = 0
        p = ArtFlowNetHistoryParams()

        # Latent encoding point net
        self.flownet = PN2DenseLatentEncodingEverywhere(
            p.encoder_dim, in_dim, 3, pnp.PN2DenseParams()
        )

        # Create the history flow encoder
        # Indim is 3 because we are going to pass in the history of prev flows
        self.prev_flow_encoder = pnp.PN2Encoder(in_dim=3, out_dim=p.encoder_dim)

        self.transformer = nn.Transformer(d_model=p.encoder_dim)
        # Only take the last index [-1] and pass it as the latent into the
        # existing latent encoding model

        self.p = p

        self.lr = training_cfg.lr
        self.batch_size = training_cfg.batch_size
        self.mask_input_channel = training_cfg.mask_input_channel

        self.vectorize = training_cfg.vectorize

    # Need to unpack the x and pos like how I did in the other file
    def forward(self, batch) -> torch.Tensor:  # type: ignore
        if self.vectorize:
            history_batch = get_history_batch(batch).to(self.device)
            # encoder = pnp.PN2Encoder(in_dim=3, out_dim=256)
            # breakpoint()
            results = self.prev_flow_encoder(history_batch)
            # breakpoint()

            history_nested_list = history_latents_to_nested_list(batch, results)

            # The list of history latents is the input to the transformer.
            # Each element in the list is a variable-lenght sequence of latents, with shape [Ni, 256]
            # The transformer expects the input to have shape [S, N, E], where S is the sequence length, N is the batch size, and E is the embedding size.
            # We need to pad and mask:

            # Pad the sequences to the same length, using torch's pad_sequence function.
            src_padded = nn.utils.rnn.pad_sequence(
                history_nested_list, batch_first=False, padding_value=0
            )

            # Create a mask for the padded sequences.
            src_mask = (src_padded == 0.0).all(dim=-1)

            # This is our query vector. It has shape [S, N, E], where S is the sequence length, N is the batch size, and E is the embedding size.
            tgt = torch.ones(1, batch.num_graphs, self.p.encoder_dim).to(self.device)

            # The transformer also expects the input to be of type float.
            src_padded = src_padded.float()
            tgt = tgt.float()

            # Pass the input through the transformer, with mask and tgt.
            out = self.transformer(
                src_padded, tgt, src_key_padding_mask=src_mask.transpose(1, 0)
            )

            embeddings = out.permute(1, 0, 2).squeeze(1)

            new_batch = batch.to_data_list()
            for data in new_batch:
                data.pos = data.curr_pos
                data.x = None
            new_batch = tgd.Batch.from_data_list(new_batch).to(self.device)
            # breakpoint()
            pred_flow = self.flownet(new_batch, embeddings)

            return pred_flow
            # return {
            #     "pred_flow": pred_flow,
            #     "latents": embeddings,
            #     "results": results,
            #     "inputs": history_batch,
            # }

        else:
            data_list = batch.to_data_list()

            current_data_list = []
            history_latents_list = []
            encoded_prev_pcs = []
            inputs = []

            # Split data into previous and current data.
            for data in data_list:
                current_data_list.append(
                    tgd.Data(
                        x=None,
                        pos=data.curr_pos,
                    )
                )

                # N = data.curr_pos.shape[0] # num of points
                # ixs = [(i*N) for i in range(data.K + 1)]
                ixs = [0] + data.lengths.cumsum(0).tolist()
                curr_history_list = []
                for i in range(len(ixs) - 1):  # data.history is a list
                    prev_pc_data = tgd.Data(
                        x=data.flow_history[ixs[i] : ixs[i + 1]],
                        pos=data.history[ixs[i] : ixs[i + 1]],
                    )
                    curr_history_list.append(prev_pc_data)
                inputs.extend(curr_history_list)
                encoded_prev_pc = self.prev_flow_encoder(
                    tgd.Batch.from_data_list(curr_history_list).to(self.device)
                )
                encoded_prev_pcs.append(encoded_prev_pc)
                encoded_prev_pc = torch.unsqueeze(encoded_prev_pc, 1)
                # encoded prev pc has dims of 2x128 (2 obs each obs has encoding of 128)
                dst = torch.ones((1, 1, self.p.encoder_dim)).to(self.device)
                # breakpoint()
                tformer_representation = self.transformer(
                    encoded_prev_pc, dst
                )  # maybe add .cuda() here somehow
                # input: 16 x 1 x 128
                # output: 1 x 1 x 128
                # breakpoint()

                history_latents_list.append(tformer_representation)

            # Latent everywhere approach
            history_latents = torch.concatenate(
                history_latents_list
            )  # batch size x 128
            history_latents = history_latents.squeeze(1)
            curr_batch = tgd.Batch.from_data_list(current_data_list)
            preds = self.flownet(curr_batch, history_latents)

            return preds
            # return {
            #     "pred_flow": preds,
            #     "latents": history_latents,
            #     "results": torch.cat(encoded_prev_pcs, dim=0),
            #     "inputs": tgd.Batch.from_data_list(inputs)
            # }

    def _step(self, batch: tgd.Batch, mode):
        # Make a prediction.
        f_pred = self(batch)

        # Compute the loss.
        # breakpoint()
        n_nodes = torch.as_tensor([len(d.mask) for d in batch.to_data_list()]).to(self.device)  # type: ignore
        f_ix = batch.mask.bool()
        f_target = batch.flow.float()
        loss = artflownet_loss(f_pred, f_target, n_nodes)

        # Compute some metrics on flow-only regions.
        rmse, cos_dist, mag_error = flow_metrics(f_pred[f_ix], f_target[f_ix])

        self.log(
            f"{mode}/loss",
            loss,
            add_dataloader_idx=False,
            prog_bar=True,
            batch_size=len(batch),
        )
        self.log_dict(
            {
                f"{mode}/rmse": rmse,
                f"{mode}/cosine_similarity": cos_dist,
                f"{mode}/mag_error": mag_error,
            },
            add_dataloader_idx=False,
            batch_size=len(batch),
        )

        return f_pred.reshape(len(batch), -1, 3), loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: tgd.Batch, batch_id):  # type: ignore
        self.train()
        f_pred, loss = self._step(batch, "train")
        return {"loss": loss, "preds": f_pred}

    def validation_step(self, batch: tgd.Batch, batch_id, dataloader_idx=0):  # type: ignore
        self.eval()
        dataloader_names = ["train", "val", "unseen"]
        name = dataloader_names[dataloader_idx]
        f_pred, loss = self._step(batch, name)
        return {"preds": f_pred, "loss": loss}

    @staticmethod
    def make_plots(preds, batch: tgd.Batch) -> Dict[str, go.Figure]:
        obj_id = batch.id
        pos = batch.curr_pos.numpy()
        mask = batch.mask.numpy()
        f_target = batch.flow
        f_pred = preds.squeeze(0)

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene", "colspan": 2}, None],
                [{"type": "scene"}, {"type": "scene"}],
            ],
            subplot_titles=(
                "input data",
                "target flow",
                "pred flow",
            ),
            vertical_spacing=0.05,
        )

        # Parent/child plot.
        labelmap = {0: "unselected", 1: "part"}
        labels = torch.zeros(len(pos)).int()
        labels[mask == 1.0] = 1
        fig.add_traces(v3p._segmentation_traces(pos, labels, labelmap, "scene1"))

        fig.update_layout(
            scene1=v3p._3d_scene(pos),
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=1.0, y=0.75),
        )

        # normalize the flow for visualization.
        n_f_gt = (f_target / f_target.norm(dim=1).max()).numpy()
        n_f_pred = (f_pred / f_target.norm(dim=1).max()).numpy()

        # GT flow.
        fig.add_trace(v3p.pointcloud(pos, 1, scene="scene2", name="pts"), row=2, col=1)
        f_gt_traces = v3p._flow_traces(
            pos, n_f_gt, scene="scene2", name="f_gt", legendgroup="1"
        )
        fig.add_traces(f_gt_traces, rows=2, cols=1)
        fig.update_layout(scene2=v3p._3d_scene(pos))

        # Predicted flow.
        fig.add_trace(v3p.pointcloud(pos, 1, scene="scene3", name="pts"), row=2, col=2)
        f_pred_traces = v3p._flow_traces(
            pos, n_f_pred, scene="scene3", name="f_pred", legendgroup="2"
        )
        fig.add_traces(f_pred_traces, rows=2, cols=2)
        fig.update_layout(scene3=v3p._3d_scene(pos))

        fig.update_layout(title=f"Object {obj_id}")

        return {"artflownet_plot": fig}


class FlowPredictorInferenceModule(L.LightningModule):
    def __init__(self, network, inference_config) -> None:
        super().__init__()
        self.network = network
        self.mask_input_channel = inference_config.mask_input_channel

    # copy over the training module after it works
    def forward(self, batch) -> torch.Tensor:  # type: ignore
        # Maybe add the mask as an input to the network.
        # breakpoint()
        # if self.mask_input_channel:
        #     data.x = data.mask.reshape(len(data.mask), 1)

        # # Run the model.
        # flow = typing.cast(torch.Tensor, self.network(data))

        # return flow

        data_list = batch.to_data_list()

        current_data_list = []
        history_latents_list = []

        # Split data into previous and current data.
        for data in data_list:
            current_data_list.append(
                tgd.Data(
                    x=None,
                    pos=data.curr_pos,
                )
            )

            curr_history_list = []
            for i in range(len(data.history)):  # data.history is a list
                prev_pc_data = tgd.Data(
                    x=data.flow_history[i],  # prev flow
                    pos=data.history[i],  # history pc
                )
                curr_history_list.append(prev_pc_data)
            encoded_prev_pc = self.prev_flow_encoder(
                tgd.Batch.from_data_list(curr_history_list)
            )
            encoded_prev_pc = torch.unsqueeze(encoded_prev_pc, 1)
            # encoded prev pc has dims of 2x128 (2 obs each obs has encoding of 128)
            dst = torch.ones((1, 1, 128)).cuda()
            tformer_representation = self.transformer(
                encoded_prev_pc, dst
            )  # maybe add .cuda() here somehow
            # input: 16 x 1 x 128
            # output: 1 x 1 x 128
            history_latents_list.append(tformer_representation)

        # Latent everywhere approach
        # breakpoint()
        history_latents = torch.concatenate(history_latents_list)  # batch size x 128
        history_latents = history_latents.squeeze(1)
        curr_batch = tgd.Batch.from_data_list(current_data_list)
        preds = self.flownet(curr_batch, history_latents)

        return preds

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:  # type: ignore
        return self.forward(batch)

    # the predict step input is different now, pay attention
    def predict(self, xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict the flow for a single object. The point cloud should
        come straight from the maniskill processed observation function.

        Args:
            xyz (torch.Tensor): Nx3 pointcloud
            mask (torch.Tensor): Nx1 mask of the part that will move.

        Returns:
            torch.Tensor: Nx3 dense flow prediction
        """
        print(xyz, mask)
        assert len(xyz) == len(mask)
        assert len(xyz.shape) == 2
        assert len(mask.shape) == 1

        data = Data(pos=xyz, mask=mask)
        batch = tgd.Batch.from_data_list([data])
        batch = batch.to(self.device)
        self.eval()
        with torch.no_grad():
            flow = self.forward(batch)
        return flow
