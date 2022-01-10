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
import argparse
import shutil
import tempfile
import filecmp
from pathlib import Path

from utils import get_patch_file, get_package_dir, load_config, git_apply_patch, git_clone, PACKAGES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='clone open source codes and apply patch')
    parser.add_argument('package', type=str, choices=PACKAGES,
                        help='package name')

    args = parser.parse_args()
    package = args.package

    git_url, tag, files = load_config(package)
    # The package location inside the xai, e.g. xai/mindspore_xai/third_party/lime
    package_dir = get_package_dir(package)
    patch_file = get_patch_file(package)

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir) / package
        git_clone(git_url, repo_dir, files, tag)

        git_apply_patch(repo_dir, patch_file)

        # copy the interested files from repo directory to local directory
        for f in files:
            src = repo_dir / f['src']
            dst = package_dir / f['dst']
            if dst.is_file():
                if not filecmp.cmp(str(src), str(dst)):
                    raise FileExistsError('Your local changes to the file {} would be overwritten by the patch, '
                                          'please create a new patch from it or delete the file.'.format(dst))
            dst.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(str(src), str(dst))

    print('patch applied to {}'.format(package_dir))
