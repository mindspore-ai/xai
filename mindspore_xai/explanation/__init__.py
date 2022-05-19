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
"""Explainers."""
from mindspore import log

from ..explainer import Gradient, GradCAM, Deconvolution, GuidedBackprop, Occlusion, RISE, RISEPlus
from ..tool.cv import OoDNet

log.warning("'mindspore_xai.explanation' and 'mindspore_xai.explanation.OoDNet' are deprecated from version 1.8.0 "
            "and will be removed in a future version, use 'mindspore_xai.explainer' and "
            "'mindspore_xai.tool.cv.OoDNet' instead.")

__all__ = [
    'Gradient',
    'Deconvolution',
    'GuidedBackprop',
    'GradCAM',
    'Occlusion',
    'RISE',
    'RISEPlus',
    'OoDNet'
]
