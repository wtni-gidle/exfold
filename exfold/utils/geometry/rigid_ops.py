"""Rigid3Array Transformations represented by a Matrix and a Vector."""

from __future__ import annotations
import dataclasses
from typing import Union, List, Optional

import torch

from exfold.utils.geometry.rotation_ops import Rot3Array
from exfold.utils.geometry.vector_ops import Vec3Array


Float = Union[float, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class Rigid3Array:
    """Rigid Transformation, i.e. element of special euclidean group."""

    rotation: Rot3Array
    translation: Vec3Array
    
    def __getitem__(self, index) -> Rigid3Array:
        return Rigid3Array(
            self.rotation[index],
            self.translation[index],
        )
    
    def __mul__(self, other: torch.Tensor) -> Rigid3Array:
        return Rigid3Array(
            self.rotation * other,
            self.translation * other,
        )
    
    def __matmul__(self, other: Rigid3Array) -> Rigid3Array:
        """i.e. compose"""
        new_rotation = self.rotation @ other.rotation # __matmul__
        new_translation = self.apply_to_point(other.translation)
        return Rigid3Array(new_rotation, new_translation)
    
    def map_tensor_fn(self, fn) -> Rigid3Array:
        return Rigid3Array(
            self.rotation.map_tensor_fn(fn),
            self.translation.map_tensor_fn(fn),
        )
    
    def inverse(self) -> Rigid3Array:
        """Return Rigid3Array corresponding to inverse transform."""
        inv_rotation = self.rotation.inverse()
        inv_translation = inv_rotation.apply_to_point(-self.translation)
        return Rigid3Array(inv_rotation, inv_translation)
    
    def apply_to_point(self, point: Vec3Array) -> Vec3Array:
        """Apply Rigid3Array transform to point."""
        return self.rotation.apply_to_point(point) + self.translation
    
    def apply(self, point: torch.Tensor) -> torch.Tensor:
        """Apply Rigid3Array transform to Tensor."""
        return self.apply_to_point(Vec3Array.from_array(point)).to_tensor()
    
    def apply_inverse_to_point(self, point: Vec3Array) -> Vec3Array:
        """Apply inverse Rigid3Array transform to point."""
        new_point = point - self.translation
        return self.rotation.apply_inverse_to_point(new_point)
    
    def invert_apply(self, point: torch.Tensor) -> torch.Tensor:
        return self.apply_inverse_to_point(Vec3Array.from_array(point)).to_tensor()
    
    def compose_rotation(self, other_rotation: Rot3Array) -> Rigid3Array:
        rot = self.rotation @ other_rotation
        return Rigid3Array(rot, self.translation.clone())
    
    def compose(self, other_rigid: Rigid3Array) -> Rigid3Array:
        return self @ other_rigid
    
    def unsqueeze(self, dim: int) -> Rigid3Array:
        return Rigid3Array(
            self.rotation.unsqueeze(dim),
            self.translation.unsqueeze(dim),
        )
    
    @property
    def shape(self) -> torch.Size:
        return self.rotation.xx.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.rotation.xx.dtype

    @property
    def device(self) -> torch.device:
        return self.rotation.xx.device
    
    @classmethod
    def identity(
        cls, 
        shape: torch.Size, 
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> Rigid3Array:
        """Return identity Rigid3Array of given shape."""
        return cls(
            Rot3Array.identity(shape, dtype=dtype, device=device),
            Vec3Array.zeros(shape, dtype=dtype, device=device)
        )

    @classmethod
    def cat(cls, rigids: List[Rigid3Array], dim: int) -> Rigid3Array:
        return cls(
            Rot3Array.cat(
                [r.rotation for r in rigids], dim=dim
            ),
            Vec3Array.cat(
                [r.translation for r in rigids], dim=dim
            ),
        )
    
    def scale_translation(self, factor: Float) -> Rigid3Array:
        """Scale translation in Rigid3Array by 'factor'."""
        return Rigid3Array(self.rotation, self.translation * factor)
    
    def to_tensor4x4(self) -> torch.Tensor:
        """Convert Rigid3Array to 4x4 tensor."""
        rot_array = self.rotation.to_tensor()
        vec_array = self.translation.to_tensor()
        array = torch.zeros(
            rot_array.shape[:-2] + (4, 4), 
            device=rot_array.device, 
            dtype=rot_array.dtype
        )
        array[..., :3, :3] = rot_array
        array[..., :3, 3] = vec_array
        array[..., 3, 3] = 1.
        return array
    
    def reshape(self, new_shape) -> Rigid3Array:
        rots = self.rotation.reshape(new_shape)
        trans = self.translation.reshape(new_shape)
        return Rigid3Array(rots, trans)
    
    def stop_rot_gradient(self) -> Rigid3Array:
        return Rigid3Array(
            self.rotation.stop_gradient(),
            self.translation,
        )
    
    @classmethod
    def from_tensor4x4(cls, array: torch.Tensor) -> Rigid3Array:
        """Convert Rigid3Array from 4x4 tensor."""
        rot = Rot3Array.from_array(
            array[..., :3, :3],
        )
        vec = Vec3Array.from_array(array[..., :3, 3])
        return cls(rot, vec)
    
    @classmethod
    def from_3_points_svd(
        cls, 
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ) -> Rigid3Array:
        """
        svd decomposition to find rotation and translation
        Args:
            point_1: [*, 3] coordinates
            point_2: [*, 3] coordinates
            point_3: [*, 3] coordinates
        Returns:
            Rigid3Array
        """
        # [*, 3, 3]
        points = torch.stack([p1, p2, p3], dim=-1).to(dtype=torch.float32)
        # [*, 3, 1]
        center = torch.mean(points, dim=-1, keepdim=True)
        # [*, 3, 3]
        points = points - center
        U, _, Vh = torch.linalg.svd(points, full_matrices=False)
        # [*, ]
        det = torch.det(torch.matmul(U, Vh))
        # [3, 3]
        diag = points.new_zeros(points.shape)
        diag[..., 0, 0] = 1.0
        diag[..., 1, 1] = 1.0
        diag[..., 2, 2] = det
        rot = U @ diag @ Vh
        rot = rot.to(dtype=p1.dtype)
        center = center.squeeze(-1).to(dtype=p1.dtype)

        rot = Rot3Array.from_array(rot)
        vec = Vec3Array.from_array(center)

        return cls(rot, vec)
        
    def cuda(self) -> Rigid3Array:
        return Rigid3Array.from_tensor4x4(self.to_tensor4x4().cuda())
