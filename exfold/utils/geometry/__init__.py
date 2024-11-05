# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Geometry Module."""

from exfold.utils.geometry import rigid_ops
from exfold.utils.geometry import rotation_ops
from exfold.utils.geometry import vector_ops

Rot3Array = rotation_ops.Rot3Array
Rigid3Array = rigid_ops.Rigid3Array
Vec3Array = vector_ops.Vec3Array

square_euclidean_distance = vector_ops.square_euclidean_distance
euclidean_distance = vector_ops.euclidean_distance
dihedral_angle = vector_ops.dihedral_angle
dot = vector_ops.dot
cross = vector_ops.cross
