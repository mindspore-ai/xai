# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Deep neural network explainers."""

from .backprop.gradient import Gradient
from .backprop.gradcam import GradCAM
from .backprop.modified_relu import Deconvolution, GuidedBackprop
from .shap import SHAPGradient, SHAPKernel
from .perturb.occlusion import Occlusion
from .perturb.rise import RISE
from .perturb.riseplus import RISEPlus
from .perturb.lime import LIMETabular

__all__ = [
    'Gradient',
    'Deconvolution',
    'GuidedBackprop',
    'GradCAM',
    "SHAPGradient",
    "SHAPKernel",
    'Occlusion',
    'RISE',
    'RISEPlus',
    'LIMETabular'
]
