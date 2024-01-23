import random
from typing import List, Optional, Protocol, Union, cast

import flowbot3d.datasets.flow_dataset as f3dd
import numpy as np
import pybullet as p
import rpad.partnet_mobility_utils.dataset as pmd
import torch
import torch_geometric.data as tgd
from rpad.partnet_mobility_utils.render.pybullet import PybulletRenderer
from torch_geometric.data import Data

"""
Variable length history dataset
- Generated by taking the joint limits of a randomly selected joint
- Rendering small change in movement for 100 steps
- Randomly selecting a variable length (K) from the history
"""
############################################################

# Joints helper functions


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


def get_joint_type(obj_id, client_id, joint_ix):
    jinfo = p.getJointInfo(obj_id, joint_ix, client_id)
    if jinfo[2] == p.JOINT_REVOLUTE:
        return "R"
    elif jinfo[2] == p.JOINT_PRISMATIC:
        return "P"


def get_joints(obj_id, client_id):
    joints = []
    for i in range(p.getNumJoints(obj_id, client_id)):
        jinfo = p.getJointInfo(obj_id, i, client_id)
        joints.append(jinfo)
    return joints


def get_joint_angles(obj_id, client_id):
    angles = {}
    for i in range(p.getNumJoints(obj_id, client_id)):
        jinfo = p.getJointInfo(obj_id, i, client_id)
        jstate = p.getJointState(obj_id, i, client_id)
        angles[jinfo[12].decode("UTF-8")] = jstate[0]
    return angles


############################################################


class FlowDataTformerHistory(Protocol):
    id: str  # Object ID.
    curr_pos: torch.Tensor  # Points in the point cloud.
    flow: torch.Tensor  # instantaneous positive 3D flow.
    mask: torch.Tensor  # Mask of the part of interest.
    K: int  # Size of history window (i.e. number of point clouds in the history)
    history: torch.Tensor  # Array of K point clouds which form the history of the observation


class FlowDatasetTformerHistory(tgd.Dataset):
    def __init__(
        self,
        root: str,
        split: Union[pmd.AVAILABLE_DATASET, List[str]],
        randomize_joints: bool = True,
        randomize_camera: bool = True,
        n_points: Optional[int] = 1200,
        seed: int = 42,
    ):
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

    def get_data(self, obj_id: str, seed=None) -> FlowDataTformerHistory:
        # Initial randomization parameters.
        joints = "random" if self.randomize_joints else None
        camera_xyz = "random" if self.randomize_camera else None

        rng = np.random.default_rng(seed)
        seed1, seed2, seed3, seed4 = rng.bit_generator._seed_seq.spawn(4)  # type: ignore

        data_t0 = self._dataset.get(
            obj_id=obj_id, joints=joints, camera_xyz=camera_xyz, seed=seed1
        )
        raw_data_obj = self._dataset.pm_objs[obj_id].obj

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

        ###################################################################

        # Randomly select a joint to modify by poking through the guts.
        renderer: PybulletRenderer = self._dataset.renderers[obj_id]  # type: ignore
        joint_name, joint_type, joint_ix = get_random_joint(
            obj_id=renderer._render_env.obj_id,
            client_id=renderer._render_env.client_id,
            seed=seed2,
        )

        # Create object from object id
        all_joints = get_joints(
            obj_id=renderer._render_env.obj_id, client_id=renderer._render_env.client_id
        )  # actually link

        joint = raw_data_obj.get_joint(joint_name)

        min_theta, max_theta = joint.limit

        # Decide the "delta" for a specific joint.
        d_theta = (max_theta - min_theta) / 100

        # HACK HACK HACK we need to make sure that the joint is actually in the joint list.
        # This is a bug in the underlying library, annoying.
        joints_t1 = {
            jn: jv
            for jn, jv in joints_t0.items()
            if jn in renderer._render_env.jn_to_ix
        }

        ###################################################################
        # Render. and compute values.
        prev_jas = get_joint_angles(
            obj_id=renderer._render_env.obj_id,
            client_id=renderer._render_env.client_id,
        )

        theta = min_theta + d_theta
        num_obs = 0

        while theta < max_theta:
            joints_t1[joint_name] = theta

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

            # Downsample.
            if self.n_points:
                rng = np.random.default_rng(seed4)

                ixs_t0 = rng.permutation(range(len(pos_t0)))[: self.n_points]
                pos_t0 = pos_t0[ixs_t0]
                flow_t0 = flow_t0[ixs_t0]
                mask_t0 = mask_t0[ixs_t0]

                ixs_t1 = rng.permutation(range(len(pos_t1)))[: self.n_points]
                pos_t1 = pos_t1[ixs_t1]
                flow_t1 = flow_t1[ixs_t1]
                mask_t1 = mask_t1[ixs_t1]

            # Add to series of history
            if num_obs == 0:
                history = np.array([pos_t0])
                flow_history = np.array([flow_t0])
            else:
                history = np.append(history, [pos_t1], axis=0)
                flow_history = np.append(flow_history, [flow_t1], axis=0)

            num_obs += 1
            theta += d_theta

        start_ix = random.randint(0, num_obs - 1)
        end_ix = random.randint(start_ix + 1, num_obs)
        K = end_ix - start_ix

        curr_pos = history[end_ix]
        flow = flow_history[end_ix]

        history = history[start_ix:end_ix]
        flow_history = flow_history[start_ix:end_ix]

        data = Data(
            id=obj_id,
            curr_pos=torch.from_numpy(pos_t1).float(),
            action=torch.from_numpy(action).float(),
            flow=torch.from_numpy(flow_t1).float(),
            mask=torch.from_numpy(mask_t1).float(),
            history=torch.from_numpy(history).float(),  # Snapshot of history
            flow_history=torch.from_numpy(
                flow_history
            ).float(),  # Snapshot of flow history
            link=joint.child,  # child of the joint gives you the link that the joint is connected to
            K=K,  # length of history
        )

        return cast(FlowDataTformerHistory, data)
