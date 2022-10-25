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
from mindspore.train._utils import check_value_type


def _unify_saliency(saliency):
    """Unify the saliency input."""
    check_value_type("saliency", saliency, [ms.Tensor, np.ndarray])
    if isinstance(saliency, ms.Tensor):
        if not ((saliency.dtype == ms.float32) or (saliency.dtype == ms.float64)):
            raise ValueError("saliency should have dtype ms.float32 or ms.float64.")
        saliency = saliency.asnumpy()
    else:
        if not ((saliency.dtype == np.float32) or (saliency.dtype == np.float64)):
            raise ValueError("saliency should have dtype np.float32 or np.float64.")
    saliency = np.squeeze(saliency)
    if len(saliency.shape) != 2:
        raise ValueError(f"The squeezed saliency shape({saliency.shape}) length is not 2.")
    return saliency


def np_normalize_saliency(saliency):
    """
    Normalize the saliency numpy array by value range.

    Args:
        saliency(np.ndarray): numpy array of saliency map.

    Returns:
        np.ndarray, normalized saliency map in shape of :math:`(H, W)` .
    """
    rng_min = saliency.min()
    rng_max = saliency.max()
    if not np.isfinite(rng_min) or not np.isfinite(rng_max) or rng_max <= rng_min:
        return saliency
    return (saliency - rng_min) / (rng_max - rng_min)


def np_saliency_to_rgba(saliency, cm=None, alpha_factor=1.2, as_uint8=True, normalize=True):
    """
    Convert the saliency numpy array to RGBA numpy array.

    Args:
        saliency(np.ndarray): numpy array of saliency map in shape of :math:`(H, W)`.
        cm(Callable, optional): Color map, viridis of matplotlib will be used if `None` is provided. Default: `None`.
        alpha_factor(float): Alpha channel multiplier. Default: 1.2.
        as_uint8(bool): Return as with UINT8 data type. Default: `True`.
        normalize(bool): Normalize the input saliency map. Default: `True`.

    Returns:
        np.ndarray, RGBA numpy array in shape of :math:`(H, W, 4)` if `cm` was set to `None`.
    """
    if not ((cm is None) or callable(cm)):
        raise ValueError("cm should be function or Nonetype.")
    check_value_type("alpha_factor", alpha_factor, float)
    check_value_type("as_uint8", as_uint8, bool)
    check_value_type("normalize", normalize, bool)

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
    """
    Convert the saliency numpy array to PIL.Image.Image.

    Args:
        saliency(np.ndarray): numpy array of saliency map in shape of :math:`(H, W)`.
        original(PIL.Image.Image, optional): `The original image
            <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image>`_ . Default: `None`.
        cm(Callable, optional): Color map, viridis of matplotlib will be used if None is provided. Default: `None`.
        normalize(bool): Normalize the input saliency map. Default: `True`.
        with_alpha(bool): Add alpha channel to the returned image. Default: `False`.

    Returns:
        PIL.Image.Image, the converted image object in size of :math:`(H, W)` with RGB or RGBA (if `with_alpha` is
        `True`) channels.
    """
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
    """
    Normalize the saliency map.

    Args:
        saliency(Tensor, np.ndarray): Saliency map in shape of :math:`(H, W)`.

    Returns:
        np.ndarray, the normalized saliency map in shape of :math:`(H, W)`.

    Examples:
        >>> import numpy as np
        >>> from mindspore_xai.visual.cv import normalize_saliency
        >>>
        >>> # prepare the saliency map
        >>> saliency_np = np.array([[0.4, 0.3, 0.1], [0.5, 0.9, 0.1]])
        >>> output_img = normalize_saliency(saliency_np)
        >>> print(output_img.shape)
        (2, 3)
    """
    saliency = _unify_saliency(saliency)
    return np_normalize_saliency(saliency)


def saliency_to_rgba(saliency, cm=None, alpha_factor=1.2, as_uint8=True, normalize=True):
    """
    Convert the saliency map to a RGBA numpy array.

    Args:
        saliency(Tensor, np.ndarray): Saliency map in shape of :math:`(H, W)`.
        cm(Callable, optional): Color map, viridis of matplotlib will be used if `None` is provided. Default: `None`.
        alpha_factor(float, optional): Alpha channel multiplier. Default: 1.2.
        as_uint8(bool, optional): Return as with UINT8 data type. Default: `True`.
        normalize(bool, optional): Normalize the input saliency map. Default: `True`.

    Returns:
        np.ndarray, the converted RGBA map in shape of :math:`(H, W, 4)` if `cm` was set to `None`.

    Examples:
        >>> import numpy as np
        >>> from mindspore_xai.visual.cv import saliency_to_rgba
        >>>
        >>> # prepare the saliency map
        >>> saliency_np = np.array([[0.4, 0.3, 0.1], [0.5, 0.9, 0.1]])
        >>> output_img = saliency_to_rgba(saliency_np)
        >>> print(output_img.shape)
        (2, 3, 4)
    """
    saliency = _unify_saliency(saliency)
    return np_saliency_to_rgba(saliency, cm, alpha_factor, as_uint8, normalize)


def saliency_to_image(saliency, original=None, cm=None, normalize=True, with_alpha=False):
    """
    Convert the saliency map to a PIL.Image.Image object.

    Args:
        saliency(Tensor, np.ndarray): Saliency map in shape of :math:`(H, W)`.
        original(PIL.Image.Image, optional): `The original image
            <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image>`_ . Default: `None`.
        cm(Callable, optional): Color map, viridis of matplotlib will be used if `None` is provided. Default: `None`.
        normalize(bool, optional): Normalize the input saliency map. Default: `True`.
        with_alpha(bool, optional): Add alpha channel to the returned image. Default: `False`.

    Returns:
        PIL.Image.Image, the converted image object in size of :math:`(H, W)` with RGB or RGBA (if `with_alpha` is
        `True`) channels.

    Examples:
        >>> import numpy as np
        >>> from PIL import Image
        >>> from mindspore_xai.visual.cv import saliency_to_image
        >>>
        >>> # prepare the original image
        >>> img_array = np.random.randint(255, size=(400, 400), dtype=np.uint8)
        >>> orig_img = Image.fromarray(img_array)
        >>> # prepare the saliency map
        >>> saliency_np = np.random.rand(400, 400)
        >>> output_img = saliency_to_image(saliency_np, orig_img)
        >>> print(output_img.size)
        (400, 400)
    """
    check_value_type("original", original, [Image.Image, type(None)])
    check_value_type("with_alpha", with_alpha, bool)
    saliency = _unify_saliency(saliency)
    return np_saliency_to_image(saliency, original, cm, normalize, with_alpha)
