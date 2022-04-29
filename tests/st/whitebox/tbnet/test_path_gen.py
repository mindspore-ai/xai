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
"""Initialization of tests of PathGen."""
import tempfile
import pytest

from mindspore_xai.whitebox.tbnet import PathGen


class TestPathGen:
    """Test PathGen."""

    def setup_method(self):
        """Setup the test case."""
        self.path_gen = PathGen(per_item_paths=5)

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_path_gen(self):
        """Test PathGen."""
        src_csv = """user,item,rating,a0,a1
u0,i0,p,a00,a10
u0,i2,p,a02,a10
u0,i3,c,a01,a12
u0,i6,p,a03,a11
u0,i7,x,a00,a15
u0,i8,x,a00,a15
u1,i5,p,a00,a10
u1,i1,x,a02,a12
u1,i3,c,a01,a12
u1,i6,p,a03,a11
u1,i7,x,a00,a15
u1,i8,c,a00,a15
"""

        src_fp = tempfile.TemporaryFile(mode='r+')
        src_fp.write(src_csv)
        src_fp.flush()
        src_fp.seek(0)

        path_fp = tempfile.TemporaryFile(mode='w')

        rows = self.path_gen.generate(src_fp, path_fp)
        assert rows == 7
