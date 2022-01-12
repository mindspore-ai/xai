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
"""Create patch."""
import argparse
import shutil
import tempfile
from pathlib import Path

from utils import get_patch_file, get_package_dir, load_config, git_create_patch, get_repo_from_url, PACKAGES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create patch (difference between open source codes and local codes)')
    parser.add_argument('package', type=str, choices=PACKAGES,
                        help='package name')

    args = parser.parse_args()
    package = args.package

    url, files = load_config(package)
    # The package location inside the xai, e.g. xai/mindspore_xai/third_party/lime
    package_dir = get_package_dir(package)
    patch_file = get_patch_file(package)

    for f in files:
        dst = package_dir / f['dst']
        if not dst.is_file():
            raise FileNotFoundError('{} not exist in the package directory!'.format(dst))

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir) / package
        get_repo_from_url(url, repo_dir)

        # copy the modified codes from local directory to repo directory
        for f in files:
            src = repo_dir / f['src']
            dst = package_dir / f['dst']
            if not src.is_file():
                raise FileNotFoundError('{} not exist in the repo directory!'.format(src))
            shutil.copyfile(str(dst), str(src))

        git_create_patch(str(repo_dir), patch_file)

    print('patch saved to {}'.format(patch_file))
