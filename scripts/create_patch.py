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

from utils import get_patch_file, get_package_local_dir, load_config, git_create_patch, get_source_code_from_url, \
    cache_dir, rm_dir, list_third_party_src_pkg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create patch for a package'
                                                 '(difference between source codes and local codes)')
    parser.add_argument('package', type=str, help='package name, or "all" if want to create patch for all '
                                                  'third party packages')

    args = parser.parse_args()
    if args.package == "all":
        packages = list_third_party_src_pkg()
    else:
        packages = [args.package]

    for package in packages:
        print("creating patch for {}".format(package))
        url, files = load_config(package)
        # The package location inside the xai, e.g. xai/mindspore_xai/third_party/lime
        local_dir = get_package_local_dir(package)
        patch_file = get_patch_file(package)

        for f in files:
            local_p = local_dir / f
            if not local_p.exists():
                raise FileNotFoundError('{} not exist in local directory!'.format(local_p))

        source_code_dir = cache_dir / package
        get_source_code_from_url(url, package)

        # copy the modified codes from local directory to source code directory
        for f in files:
            source_code_p = source_code_dir / f
            local_p = local_dir / f

            # path could be a file or a directory
            if local_p.is_file():
                shutil.copyfile(str(local_p), str(source_code_p))
            elif local_p.is_dir():
                for src in local_p.rglob("*.py"):
                    dst = str(src).replace(str(local_p), str(source_code_p))
                    shutil.copyfile(str(src), str(dst))

        git_create_patch(str(source_code_dir), patch_file)

        # remove the source codes
        rm_dir(source_code_dir)

        print('patch saved to {}'.format(patch_file))
