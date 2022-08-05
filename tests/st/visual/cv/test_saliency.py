# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Tests of saliency tool."""
import pytest
import PIL
import numpy as np
import mindspore as ms

from mindspore_xai.visual.cv import normalize_saliency, saliency_to_rgba, saliency_to_image


H = W = 5


class TestVisualization:
    """Unit test for Saliency Visualization."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_normalize_saliency_input_tensor(self):
        """Test for normalize saliency function with a tensor input."""
        # Saliency with Tensor type
        saliency_t = ms.Tensor(np.random.rand(H, W), ms.float32)
        normalized_t = normalize_saliency(saliency_t)
        assert isinstance(normalized_t, np.ndarray)
        assert normalized_t.shape == (H, W)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_saliency_to_rgba_input_tensor(self):
        """Test for saliency to rgba function with a tensor input."""
        # Saliency with Tensor type
        saliency_t = ms.Tensor(np.random.rand(H, W), ms.float32)
        rgb_map_t = saliency_to_rgba(saliency_t)
        assert isinstance(rgb_map_t, np.ndarray)
        assert rgb_map_t.shape == (H, W, 4)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_saliency_to_image_input_tensor(self):
        """Test for saliency to image function with a tensor input."""
        # Saliency with Tensor type
        saliency_t = ms.Tensor(np.random.rand(H, W), ms.float32)
        image_t = saliency_to_image(saliency_t)
        assert isinstance(image_t, PIL.Image.Image)
        assert image_t.size == (H, W)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_normalize_saliency_input_np(self):
        """Test for normalize saliency function with a np.ndarray input."""
        # Saliency with np.array type
        saliency_np = np.random.rand(H, W)
        normalized_np = normalize_saliency(saliency_np)
        assert isinstance(normalized_np, np.ndarray)
        assert normalized_np.shape == (H, W)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_saliency_to_rgba_input_np(self):
        """Test for saliency to rgba function with a np.ndarray input."""
        # Saliency with np.array type
        saliency_np = np.random.rand(H, W)
        rgb_map_np = saliency_to_rgba(saliency_np)
        assert isinstance(rgb_map_np, np.ndarray)
        assert rgb_map_np.shape == (H, W, 4)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_saliency_to_image_input_np(self):
        """Test for saliency to image function with a np.ndarray input."""
        # Saliency with np.array type
        saliency_np = np.random.rand(H, W)
        image_np = saliency_to_image(saliency_np)
        assert isinstance(image_np, PIL.Image.Image)
        assert image_np.size == (H, W)
