mindspore_xai.visual
=================================

CV类可视化。

.. py:function:: mindspore_xai.visual.cv.normalize_saliency(saliency)

    归一化热力图。

    参数：
        - **saliency** (Tensor, np.ndarray) - shape为 :math:`(H, W)` 的热力图。

    返回：
        np.ndarray，shape为 :math:`(H, W)` 的归一化热力图。

.. py:function:: mindspore_xai.visual.cv.saliency_to_rgba(saliency, cm=None, alpha_factor=1.2, as_uint8=True, normalize=True)

    将热力图转换成RGBA numpy数组。

    参数：
        - **saliency** (Tensor, np.ndarray) - shape为 :math:`(H, W)` 的热力图。
        - **cm** (Callable, 可选) - 颜色图，如果为 ``None`` ，将使用matplotlib默认的viridis色带。默认值： ``None`` 。
        - **alpha_factor** (float, 可选) - Alpha通道乘子。默认值：1.2。
        - **as_uint8** (bool, 可选) - 返回UINT8数据类型。默认值： ``True`` 。
        - **normalize** (bool, 可选) - 归一化输入的热力图。默认值： ``True`` 。

    返回：
        np.ndarray，如果 `cm` 为 ``None`` ，返回shape 为 :math:`(H, W, 4)` 的RGBA图。

.. py:function:: mindspore_xai.visual.cv.saliency_to_image(saliency, original=None, cm=None, normalize=True, with_alpha=False)

    将热力图转换成PIL.Image.Image对象。

    参数：
        - **saliency** (Tensor, np.ndarray) - shape为 :math:`(H, W)` 的热力图。
        - **original** (PIL.Image.Image, 可选) - `原图 <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image>`_ 。默认值： ``None`` 。
        - **cm** (Callable, 可选) - 颜色图，如果为 ``None`` ，使用matplotlib默认的viridis色带。默认值： ``None`` 。
        - **normalize** (bool, 可选) - 归一化输入的热力图。默认值： ``True`` 。
        - **with_alpha** (bool, 可选) - 在返回的图像中加入alpha通道。默认值： ``False`` 。

    返回：
        PIL.Image.Image，size为 :math:`(H, W)` 的RGB通道图像。如果 `with_alpha` 为 ``True`` ，返回RGBA通道图像。