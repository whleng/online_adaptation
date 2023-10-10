import math
from typing import List, Optional, Protocol, Union, cast

import flowbot3d.datasets.flow_dataset as f3dd
import numpy as np
import pybullet as p
import rpad.partnet_mobility_utils.dataset as pmd
import torch
import torch_geometric.data as tgd
from rpad.partnet_mobility_utils.articulate import articulate_joint
from rpad.partnet_mobility_utils.render.pybullet import PybulletRenderer
from torch_geometric.data import Data


def get_random_joint(obj_id, client_id, seed=None):
    rng = np.random.default_rng(seed)
    n_joints = p.getNumJoints(obj_id, client_id)
    articulation_ixs = []
    for joint_ix in range(n_joints):
        jinfo = p.getJointInfo(obj_id, joint_ix, client_id)
        if jinfo[2] == p.JOINT_REVOLUTE or jinfo[2] == p.JOINT_PRISMATIC:
            joint_name = jinfo[1].decode("UTF-8")
            joint_type = int(jinfo[2])
            articulation_ixs.append((joint_name, joint_type, joint_ix))
    selected_ix = articulation_ixs[rng.choice(len(articulation_ixs))]
    joint_name, joint_type, joint_ix = selected_ix
    return joint_name, joint_type, joint_ix


class FlowHistoryData(Protocol):
    id: str  # Object ID.

    curr_pos: torch.Tensor  # Points in the point cloud.
    prev_pos: torch.Tensor  # Points in prev point cloud.

    last_action: torch.Tensor  # Last action to get from prev to curr point cloud.
    flow: torch.Tensor  # instantaneous positive 3D flow.
    mask: torch.Tensor  # Mask of the part of interest.

    prev_flow_gt: torch.Tensor  # Ground truth previous flow.
    prev_flow_nn: torch.Tensor  # Nearest neighbor previous flow.

    rev_optical_flow_gt: torch.Tensor  # Ground truth reverse flow.
    rev_optical_flow_nn: torch.Tensor  # Nearest neighbor reverse flow.


def normalize_flow(flow):
    nonzero = (flow != 0.0).any(axis=-1)
    if nonzero.sum() > 0.0:
        nonzero_flow = flow[nonzero]
        largest = np.linalg.norm(nonzero_flow, axis=-1).max()
        flow = flow / largest
    return flow


class FlowHistoryPyGDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        split: Union[pmd.AVAILABLE_DATASET, List[str]],
        randomize_joints: bool = True,
        randomize_camera: bool = True,
        n_points: Optional[int] = 1200,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.seed = seed
        self._dataset = pmd.PCDataset(root=root, split=split, renderer="pybullet")

        self.randomize_joints = randomize_joints
        self.randomize_camera = randomize_camera
        self.n_points = n_points

    def len(self) -> int:
        return len(self._dataset)

    def get(self, index) -> tgd.Data:
        return self.get_data(self._dataset._ids[index])

    @staticmethod
    def get_processed_dir(randomize_joints, randomize_camera):
        joint_chunk = "rj" if randomize_joints else "sj"
        camera_chunk = "rc" if randomize_camera else "sc"
        return f"processed_history_{joint_chunk}_{camera_chunk}"

    def get_data(self, obj_id: str, seed=None) -> FlowHistoryData:
        # Initial randomization parameters.
        joints = "random" if self.randomize_joints else None
        camera_xyz = "random" if self.randomize_camera else None

        rng = np.random.default_rng(seed)
        seed1, seed2, seed3, seed4 = rng.bit_generator._seed_seq.spawn(4)  # type: ignore

        ########################################################################
        # COMPUTE t-1 OBSERVATIONS
        ########################################################################

        # Get the initial render.
        data_t0 = self._dataset.get(
            obj_id=obj_id, joints=joints, camera_xyz=camera_xyz, seed=seed1
        )
        pos_t0 = data_t0["pos"]

        # Compute the flow + mask at that time.
        flow_t0 = f3dd.compute_normalized_flow(
            P_world=pos_t0,
            T_world_base=data_t0["T_world_base"],
            current_jas=data_t0["angles"],
            pc_seg=data_t0["seg"],
            labelmap=data_t0["labelmap"],
            pm_raw_data=self._dataset.pm_objs[obj_id],
            linknames="all",
        )
        mask_t0 = (~(flow_t0 == 0.0).all(axis=-1)).astype(int)

        # Compute the states for the camera and joints at t1.
        # Camera should be the same.
        camera_xyz_t1 = data_t0["T_world_cam"][:3, 3]
        joints_t0 = data_t0["angles"]

        ########################################################################
        # COMPUTE t OBSERVATIONS
        ########################################################################

        # Randomly select a joint to modify by poking through the guts.
        renderer: PybulletRenderer = self._dataset.renderers[obj_id]  # type: ignore
        joint_name, joint_type, joint_ix = get_random_joint(
            obj_id=renderer._render_env.obj_id,
            client_id=renderer._render_env.client_id,
            seed=seed2,
        )

        # Modify the joint.
        if joint_type == p.JOINT_PRISMATIC:
            d_theta = 0.5
        elif joint_type == p.JOINT_REVOLUTE:
            d_theta = math.pi / 4
        else:
            raise ValueError(
                f"Unknown joint type {joint_type}: Options{p.JOINT_PRISMATIC}, {p.JOINT_REVOLUTE}"
            )

        # HACK HACK HACK we need to make sure that the joint is actually in the joint list.
        # This is a bug in the underlying library, annoying.
        joints_t1 = {
            jn: jv
            for jn, jv in joints_t0.items()
            if jn in renderer._render_env.jn_to_ix
        }
        joints_t1[joint_name] += d_theta

        # Describe the action that was taken.
        action = np.zeros(len(joints_t0))
        action[joint_ix] = d_theta

        # Get the second render.
        data_t1 = self._dataset.get(
            obj_id=obj_id, joints=joints_t1, camera_xyz=camera_xyz_t1, seed=seed3
        )
        pos_t1 = data_t1["pos"]

        # Compute the flow + mask at that time.
        flow_t1 = f3dd.compute_normalized_flow(
            P_world=pos_t1,
            T_world_base=data_t1["T_world_base"],
            current_jas=data_t1["angles"],
            pc_seg=data_t1["seg"],
            labelmap=data_t1["labelmap"],
            pm_raw_data=self._dataset.pm_objs[obj_id],
            linknames="all",
        )

        mask_t1 = (~(flow_t1 == 0.0).all(axis=-1)).astype(int)

        ########################################################################
        # ESTIMATE THE PREVIOUS FLOW
        ########################################################################

        # For each point in the previous point cloud, find the nearest neighbor
        # in the current point cloud.
        dist = torch.cdist(torch.from_numpy(pos_t0), torch.from_numpy(pos_t1)).numpy()

        pos_t1_nn_ix = np.argmin(dist, axis=1)
        pos_t1_nn = pos_t1[pos_t1_nn_ix]

        # calculate the previous flow
        flow_nn_t0 = normalize_flow(pos_t1_nn - pos_t0)

        ########################################################################
        # COMPUTE REVERSE OPTICAL FLOWS
        ########################################################################

        link_to_actuate = self._dataset.pm_objs[obj_id].obj.get_joint(joint_name).child

        # Imagine where t1 points would have been.
        pos_t0_rev_gt = articulate_joint(
            obj=self._dataset.pm_objs[obj_id],
            current_jas=data_t1["angles"],
            link_to_actuate=link_to_actuate,
            amount_to_actuate=-d_theta,
            pos=pos_t1,
            seg=data_t1["seg"],
            labelmap=data_t1["labelmap"],
            T_world_base=data_t1["T_world_base"],
        )

        rev_optical_flow_gt = normalize_flow(pos_t0_rev_gt - pos_t1)

        # Now try and estimate what an optical flow method might return, by finding
        # the nearest neighbor in the previous point cloud to the ground-truth reverse
        # point cloud.
        dist = torch.cdist(
            torch.from_numpy(pos_t0_rev_gt), torch.from_numpy(pos_t0)
        ).numpy()

        pos_t0_rev_nn_ix = np.argmin(dist, axis=1)
        pos_t0_rev_nn = pos_t0[pos_t0_rev_nn_ix]

        # calculate the reversed flow
        rev_optical_flow_nn = normalize_flow(pos_t0_rev_nn - pos_t1)

        ########################################################################
        # DOWNSAMPLE
        ########################################################################

        if self.n_points:
            rng = np.random.default_rng(seed4)

            ixs_t0 = rng.permutation(range(len(pos_t0)))[: self.n_points]
            pos_t0 = pos_t0[ixs_t0]
            flow_t0 = flow_t0[ixs_t0]
            mask_t0 = mask_t0[ixs_t0]
            flow_nn_t0 = flow_nn_t0[ixs_t0]

            ixs_t1 = rng.permutation(range(len(pos_t1)))[: self.n_points]
            pos_t1 = pos_t1[ixs_t1]
            flow_t1 = flow_t1[ixs_t1]
            mask_t1 = mask_t1[ixs_t1]
            rev_optical_flow_gt = rev_optical_flow_gt[ixs_t1]
            rev_optical_flow_nn = rev_optical_flow_nn[ixs_t1]

        data = Data(
            id=obj_id,
            curr_pos=torch.from_numpy(pos_t1).float(),
            prev_pos=torch.from_numpy(pos_t0).float(),
            action=torch.from_numpy(action).float(),
            prev_flow_gt=torch.from_numpy(flow_t0).float(),
            prev_flow_nn=torch.from_numpy(flow_nn_t0).float(),
            flow=torch.from_numpy(flow_t1).float(),
            mask=torch.from_numpy(mask_t1).float(),
            prev_mask=torch.from_numpy(mask_t0).float(),
            rev_optical_flow_gt=torch.from_numpy(rev_optical_flow_gt).float(),
            rev_optical_flow_nn=torch.from_numpy(rev_optical_flow_nn).float(),
        )

        return cast(FlowHistoryData, data)
