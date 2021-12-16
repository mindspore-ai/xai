# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Saliency visualization."""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import mindspore as ms


def _unify_saliency(saliency):
    """Unify the saliency input."""
    if isinstance(saliency, ms.Tensor):
        saliency = saliency.asnumpy()
    saliency = np.squeeze(saliency)
    if len(saliency.shape) != 2:
        raise ValueError(f"The squeezed saliency shape({saliency.shape}) length is not 2.")
    return saliency


def np_normalize_saliency(saliency):
    """Normalize the saliency numpy array."""
    rng_min = saliency.min()
    rng_max = saliency.max()
    if not np.isfinite(rng_min) or not np.isfinite(rng_max) or rng_max <= rng_min:
        return saliency
    return (saliency - rng_min) / (rng_max - rng_min)


def np_saliency_to_rgba(saliency, cm=None, alpha_factor=1.2, as_uint8=True, normalize=True):
    """Convert the saliency numpy array to RGBA numpy array."""
    if normalize:
        saliency = np_normalize_saliency(saliency)

    if cm is None:
        cm = plt.cm.viridis
    pixels = cm(saliency)

    # saliency intensity as opacity
    pixels[:, :, -1] = saliency * alpha_factor
    if as_uint8:
        return np.clip(pixels * 255, 0, 255).astype(dtype=np.uint8)
    return np.clip(pixels, 0, 1)


def np_saliency_to_image(saliency, original=None, cm=None, normalize=True, with_alpha=False):
    """Convert the saliency numpy array to PIL.Image.Image."""
    pixels = np_saliency_to_rgba(saliency, cm=cm, as_uint8=True, normalize=normalize)
    saliency_img = Image.fromarray(pixels, mode="RGBA")
    if isinstance(original, Image.Image):
        if original.size != saliency_img.size:
            original = original.resize(saliency_img.size)
        if original.mode != 'RGBA':
            original = original.convert('RGBA')
        saliency_img = Image.alpha_composite(original, saliency_img)
    return saliency_img if with_alpha else saliency_img.convert('RGB')


def normalize_saliency(saliency):
    """Normalize the saliency map."""
    saliency = _unify_saliency(saliency)
    return np_normalize_saliency(saliency)


def saliency_to_rgba(saliency, cm=None, alpha_factor=1.2, as_uint8=True, normalize=True):
    """Convert the saliency map to a RGBA numpy array."""
    saliency = _unify_saliency(saliency)
    return np_saliency_to_rgba(saliency, cm, alpha_factor, as_uint8, normalize)


def saliency_to_image(saliency, original=None, cm=None, normalize=True, with_alpha=False):
    """Convert the saliency map to a PIL.Image.Image object."""
    saliency = _unify_saliency(saliency)
    return np_saliency_to_image(saliency, original, cm, normalize, with_alpha)
