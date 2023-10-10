from dataclasses import dataclass
from typing import Dict

import lightning as L
import plotly.graph_objects as go
import rpad.visualize_3d.plots as pvp
import torch
import torch.optim as opt
import torch_geometric.data as tgd
from flowbot3d.models.artflownet import flow_metrics
from plotly.subplots import make_subplots
from torch_geometric.data import Batch


def masked_artflownet_loss(
    f_pred: torch.Tensor,
    f_target: torch.Tensor,
    mask: torch.Tensor,
    n_nodes: torch.Tensor,
    use_mask=False,
) -> torch.Tensor:
    """Maniskill loss.

    Args:
        f_pred (torch.Tensor): Predicted flow.
        f_target (torch.Tensor): Target flow.
        mask (torch.Tensor): only mask
        n_nodes (torch.Tensor): A list describing the number of nodes
        use_mask: Whether or not to compute loss over all points, or just some.

    Returns:
        Loss
    """
    weights = (1 / n_nodes).repeat_interleave(n_nodes)

    if use_mask:
        f_pred = f_pred[mask]
        f_target = f_target[mask]
        weights = weights[mask]

    # Flow loss, per-point.
    raw_se = ((f_pred - f_target) ** 2).sum(dim=1)
    # weight each PC equally in the sum.
    l_se = (raw_se * weights).sum()

    # Full loss.
    loss: torch.Tensor = l_se / len(n_nodes)

    return loss


@dataclass
class ArtFlowNetHistoryParams:
    mask_output_flow: bool = False
    batch_size: int = 64


class ArtFlowNetHistoryModel(L.LightningModule):
    def __init__(self, network, training_cfg: ArtFlowNetHistoryParams):
        super().__init__()
        self.flownet = network
        self.mask_output_flow = training_cfg.mask_output_flow
        self.batch_size = training_cfg.batch_size

    def predict(
        self,
        xyz: torch.Tensor,
        prev_xyz: torch.Tensor,
        prev_flow: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the flow for a single object. The point cloud should
        come straight from the maniskill processed observation function.
        Args:
            xyz (torch.Tensor): Nx3 pointcloud
            mask (torch.Tensor): Nx1 mask of the part that will move.
        Returns:
            torch.Tensor: Nx3 dense flow prediction
        """
        assert len(xyz) == len(mask)
        assert len(xyz.shape) == 2
        assert len(mask.shape) == 1

        # data = Data(pos=xyz, mask=mask)
        # data = Data(pos=xyz, x=mask)
        data = tgd.Data(
            curr_pos=xyz,
            prev_pos=prev_xyz,
            prev_flow=prev_flow,
            x=mask,
        )
        batch = Batch.from_data_list([data])
        batch = batch.to(self.device)
        self.eval()
        with torch.no_grad():
            pred = self.forward(batch)
        return pred["preds"]

    def forward(self, batch: tgd.Batch) -> Dict[str, torch.Tensor]:
        data_list = batch.to_data_list()
        new_data_list = []
        pc_masks = []
        for data in data_list:
            mask = torch.cat(
                [
                    torch.ones(len(data.curr_pos), 1),
                    torch.zeros(len(data.prev_pos), 1),
                ],
                dim=0,
            )
            new_data = tgd.Data(
                x=mask.cuda(),
                pos=torch.cat([data.curr_pos, data.prev_pos], dim=0),
            )
            pc_masks.append(mask)
            new_data_list.append(new_data)
        mask_tensor = torch.cat(pc_masks, dim=0).bool()
        new_batch = Batch.from_data_list(new_data_list)
        new_batch = new_batch.to(self.device)
        preds = self.flownet(new_batch)[mask_tensor.squeeze()]
        return {
            "preds": preds,
        }

    def _step(self, batch: tgd.Batch, mode):
        n_nodes = torch.as_tensor([len(d.mask) for d in batch.to_data_list()]).to(
            self.device
        )

        f_ix = batch.mask.bool()
        # batch.x = batch.mask
        f_target = batch.flow
        N = len(f_target)
        out = self(batch)
        f_pred = out["preds"]
        loss = masked_artflownet_loss(f_pred, f_target, batch.mask, n_nodes)
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

    def configure_optimizers(self):
        return opt.Adam(params=self.parameters(), lr=0.0001)

    @staticmethod
    def make_plots(preds, batch: tgd.Batch) -> Dict[str, go.Figure]:
        f_pred = preds.squeeze(0)
        mask = batch.mask
        pos = batch.curr_pos
        f_target = batch.flow
        obj_id = batch.id

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene"}, {"type": "table"}],
                [{"type": "scene"}, {"type": "scene"}],
            ],
            subplot_titles=(
                "input data",
                "N/A",
                "target flow",
                "pred flow",
            ),
        )

        # Parent/child plot.
        labelmap = {0: "unselected", 1: "part"}
        labels = torch.zeros(len(pos)).int()
        labels[mask == 1.0] = 1
        fig.add_traces(pvp._segmentation_traces(pos, labels, labelmap, "scene1"))

        fig.update_layout(
            scene1=pvp._3d_scene(pos),
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=1.0, y=0.75),
        )

        # Connectedness table.
        fig.append_trace(
            go.Table(
                header=dict(values=["IGNORE", "IGNORE"]),
                cells=dict(values=[[1.0], [1.0]]),
            ),
            row=1,
            col=2,
        )

        # normalize the flow for visualization.
        n_f_target = f_target / f_target.norm(dim=1).max()
        n_f_pred = f_pred / f_target.norm(dim=1).max()

        # GT flow.
        fig.add_trace(pvp.pointcloud(pos, downsample=1, scene="scene2"), row=2, col=1)
        ts = pvp._flow_traces(pos, n_f_target, scene="scene2")
        for t in ts:
            fig.add_trace(t, row=2, col=1)
        fig.update_layout(scene2=pvp._3d_scene(pos))

        # Predicted flow.
        fig.add_trace(pvp.pointcloud(pos, downsample=1, scene="scene3"), row=2, col=2)
        ts = pvp._flow_traces(pos, n_f_pred, scene="scene3")
        for t in ts:
            fig.add_trace(t, row=2, col=2)
        fig.update_layout(scene3=pvp._3d_scene(pos))

        fig.update_layout(title=f"Object {obj_id}")

        return {
            "history_plot": fig,
        }
