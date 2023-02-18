"""Data parser for episodic dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from torchtyping import TensorType

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
#from nerfstudio.utils.images import BasicImages
from nerfstudio.utils.io import load_from_json

CONSOLE = Console()


def get_src_from_pairs(
    ref_idx, all_imgs, pairs_srcs, neighbors_num=None, neighbors_shuffle=False
) -> Dict[str, TensorType]:
    # src_idx[0] is ref img
    src_idx = pairs_srcs[ref_idx]
    # randomly sample neighbors
    if neighbors_num and neighbors_num > -1 and neighbors_num < len(src_idx) - 1:
        if neighbors_shuffle:
            perm_idx = torch.randperm(len(src_idx) - 1) + 1
            src_idx = torch.cat([src_idx[[0]], src_idx[perm_idx[:neighbors_num]]])
        else:
            src_idx = src_idx[: neighbors_num + 1]
    src_idx = src_idx.to(all_imgs.device)
    return {"src_imgs": all_imgs[src_idx], "src_idxs": src_idx}


def get_image(image_filename, alpha_color=None) -> TensorType["image_height", "image_width", "num_channels"]:
    """Returns a 3 channel image.

    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is in correct."
    image = torch.from_numpy(np_image.astype("float32") / 255.0)
    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
    return image


def get_depths_and_normals(image_idx: int, depths, normals):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    depth = depths[image_idx]
    # normal
    normal = normals[image_idx]

    return {"depth": depth, "normal": normal}


def get_sensor_depths(image_idx: int, sensor_depths):
    """function to process additional sensor depths

    Args:
        image_idx: specific image index to work with
        sensor_depths: semantics data
    """

    # sensor depth
    sensor_depth = sensor_depths[image_idx]

    return {"sensor_depth": sensor_depth}


def get_foreground_masks(image_idx: int, fg_masks):
    """function to process additional foreground_masks

    Args:
        image_idx: specific image index to work with
        fg_masks: foreground_masks
    """

    # sensor depth
    fg_mask = fg_masks[image_idx]

    return {"fg_mask": fg_mask}


def get_sparse_sfm_points(image_idx: int, sfm_points):
    """function to process additional sparse sfm points

    Args:
        image_idx: specific image index to work with
        sfm_points: sparse sfm points
    """

    ## sfm points
    #sparse_sfm_points = sfm_points[image_idx]
    #sparse_sfm_points = BasicImages([sparse_sfm_points])
    #return {"sparse_sfm_points": sparse_sfm_points}
    raise NotImplementedError


@dataclass
class EpisodicDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: EpisodicDataParser)
    """target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
    include_mono_prior: bool = False
    """whether or not to load monocular depth and normal """
    include_sensor_depth: bool = False
    """whether or not to load sensor depth"""
    include_foreground_mask: bool = False
    """whether or not to load foreground mask"""
    include_sfm_points: bool = False
    """whether or not to load sfm points"""
    downscale_factor: int = 1
    scene_scale: float = 2.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Friends axis-aligned bbox will be scaled to this value.
    """
    neighbors_num: Optional[int] = None
    neighbors_shuffle: Optional[bool] = False
    """if src image pairs are sorted in ascending order by similarity i.e.
    the last element is the most similar to the first (ref)"""
    skip_every_for_val_split: int = 1
    """sub sampling validation images"""
    auto_orient: bool = False


@dataclass
class EpisodicDataParser(DataParser):
    """Parsing a single or multiple scenes into episodes"""

    config: EpisodicDataParserConfig

    def _load_depth(self, depth_path):
        """ load mono depth """
        depth = np.load(depth_path)
        depth = torch.from_numpy(depth).float()
        return depth

    def _load_normal_in_worldspace(self, normal_path, camtoworld):
        """ load mono normal """
        normal = np.load(normal_path)

        # transform normal to world coordinate system
        normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
        normal = torch.from_numpy(normal).float()

        rot = camtoworld[:3, :3]

        normal_map = normal.reshape(3, -1)
        normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

        normal_map = rot @ normal_map
        normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
        return normal_map

    def _load_fg_mask(self, fg_mask_path):
        """ load foreground mask """
        foreground_mask = np.array(Image.open(fg_mask_path), dtype="uint8")
        foreground_mask = foreground_mask[..., :1]
        foreground_mask = torch.from_numpy(foreground_mask).float() / 255.0
        return foreground_mask

    def _load_sfm(self, sfm_path):
        """ load sparse sfm points """
        sfm_points_view = np.loadtxt(sfm_path)
        sfm_points_view = torch.from_numpy(sfm_points_view).float()
        return sfm_points_view

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        pass

    def _parse_single_scene(self, scene_path: Path):
        # load meta data
        meta = load_from_json(scene_path / "meta_data.json")

        image_filenames = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []

        additional_inputs_dict = {}
        depth_images = []
        normal_images = []
        sensor_depth_images = []
        foreground_mask_images = []
        sfm_points = []

        # read meta data for each frame
        for i, frame in enumerate(meta["frames"]):
            image_filename = (scene_path / frame["rgb_path"])

            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])

            # append image name & camera matrices
            image_filenames.append(image_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)

            if self.config.include_mono_prior:
                assert meta["has_mono_prior"]
                depth_images.append(scene_path / frame["mono_depth_path"])
                normal_images.append(scene_path / frame["mono_normal_path"])

            if self.config.include_sensor_depth:
                assert meta["has_sensor_depth"]
                sensor_depth_images.append(scene_path / frame["sensor_depth_path"])

            if self.config.include_foreground_mask:
                assert meta["has_foreground_mask"]
                foreground_mask_images.append(scene_path / frame["foreground_mask"])

            if self.config.include_sfm_points:
                assert meta["has_sparse_sfm_points"]
                sfm_points.append(scene_path / frame["sfm_sparse_points_view"])

        fx = torch.stack(fx) # B,
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds) # B, 4, 4

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio:
        # +X is right, +Y is up, and +Z is pointing back and away from the camera.
        # -Z is the look-at direction. Other codebases may use the COLMAP/OpenCV convention,
        # where the Y and Z axes are flipped from ours but the +X axis remains the same.
        camera_to_worlds[:, 0:3, 1:3] *= -1

        # optimize camera poses
        if self.config.auto_orient:
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method="up",
                center_poses=False,
            )

        # scene box from meta data
        meta_scene_box = meta["scene_box"]
        aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox(
            aabb=aabb,
            near=meta_scene_box["near"],
            far=meta_scene_box["far"],
            radius=meta_scene_box["radius"],
            collider_type=meta_scene_box["collider_type"],
        )

        height, width = meta["height"], meta["width"]

        # create camera object for each view
        cameras = Cameras( # of shape (B,)
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO supports downsample
        # cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        # load pair information
        pairs_path = self.config.data / "pairs.txt"
        assert pairs_path.exists(), "The list of source views for each view is missing."

        with open(pairs_path, "r") as f:
            pairs = f.readlines()

        pairs_srcs = {}
        for sources_line in pairs:
            sources_array = sources_line.split(" ")
            # invert (flip) the source elements s.t. the most similar source is in index 1 (index 0 is reference)
            pairs_srcs[sources_array[0]] = sources_array[:1:-1]

        additional_inputs_dict["pairs"] = pairs_srcs
        additional_inputs_dict["depth"] = depth_images
        additional_inputs_dict["normal"] = normal_images
        additional_inputs_dict["sensor_depth"] = sensor_depth_images
        additional_inputs_dict["fg_mask"] = foreground_mask_images
        additional_inputs_dict["sfm_points"] = sfm_points

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            depths=depth_images,
            normals=normal_images,
            additional_inputs=additional_inputs_dict,
        )
        return dataparser_outputs


if __name__ == '__main__':
    config = EpisodicDataParserConfig(data=Path('../../../../GeneralizableNerf/data/sdfstudio-demo-data/dtu-scan65'))
    D = EpisodicDataParser(config)
    outputs = D.get_dataparser_outputs('test')
