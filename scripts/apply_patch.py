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
"""Apply patch."""
import argparse
import shutil
import filecmp
from pathlib import Path

from utils import get_patch_file, get_package_local_dir, load_config, git_apply_patch, get_source_code_from_url, \
    cache_dir, list_third_party_src_pkg


def safe_copy(src_path, dst_path):
    """
    safe copy file

    Args:
        src_path (Path): Source.
        dst_path (Path): Destination.
    """
    if dst_path.is_file():
        if not filecmp.cmp(str(src_path), str(dst_path)):
            raise FileExistsError('Your local changes to the file {} would be overwritten by the patch, '
                                  'please create a new patch from it or delete the file.'.format(dst_path))

    dst_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(str(src_path), str(dst_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='clone open source codes and apply patch')
    parser.add_argument('package', type=str, help='package name, or "all" if want to apply patch for all '
                                                  'third party packages')

    args = parser.parse_args()

    if args.package == "all":
        packages = list_third_party_src_pkg()
    else:
        packages = [args.package]

    for package in packages:
        print("applying patch for {}".format(package))
        url, files = load_config(package)
        # The package location inside the xai, e.g. xai/mindspore_xai/third_party/lime
        local_dir = get_package_local_dir(package)
        patch_file = get_patch_file(package)

        source_code_dir = cache_dir / package

        get_source_code_from_url(url, package)
        git_apply_patch(str(source_code_dir), patch_file)

        # copy the interested files from source code directory to package local directory
        for f in files:
            source_code_p = source_code_dir / f
            local_p = local_dir / f

            if source_code_p.is_file():
                safe_copy(source_code_p, local_p)
            elif source_code_p.is_dir():
                for src in source_code_p.rglob("*.py"):
                    dst = str(src).replace(str(source_code_p), str(local_p))
                    safe_copy(src, Path(dst))

        # remove the source codes
        shutil.rmtree(str(source_code_dir))

        # create __init__.py
        (local_dir / "__init__.py").touch()

        print('patch applied to {}'.format(local_dir))
