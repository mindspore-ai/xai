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
import os
import subprocess
import json
from pathlib import Path

PACKAGES = ["lime"]
# i.e. xai/
root_dir = Path(__file__).resolve().parents[1]


def get_git_version():
    out = subprocess.check_output(['git', '--version']).strip().decode('utf-8')
    version = out.split(' ')[2]
    # e.g. [2, 30, 1]
    return [int(x) for x in version.split('.')[:3]]


def run_cmd(cmd, directory=None):
    cwd = Path.cwd()
    # temporary switch working directory
    if directory is not None:
        os.chdir(str(directory))

    if subprocess.call(cmd, shell=True) != 0:
        raise RuntimeError("failed to run command: `{}`".format(cmd))

    if directory is not None:
        os.chdir(str(cwd))


def git_clone(git_url, repo_dir, files, tag='master'):
    git_version = get_git_version()

    # sparse-checkout was introduced in Git version 2.25
    if git_version[0] >= 2 and git_version[1] >= 25:
        # Clone minimum files, ref: https://stackoverflow.com/a/63786181
        files = [f['src'] for f in files]

        clone_cmd = 'git clone -c advice.detachedHead=false -q  --filter=blob:none --no-checkout --sparse -b {} ' \
                    '--depth 1 {} {}'.format(tag, git_url, repo_dir)
        checkout_cmd = "git sparse-checkout init --cone && git sparse-checkout add {} && git checkout -q".format(
            " ".join(files)
        )

        run_cmd(clone_cmd)
        run_cmd(checkout_cmd, repo_dir)
    else:
        clone_cmd = 'git clone -c advice.detachedHead=false -q -b {} --depth 1 {} {}'.format(tag, git_url, repo_dir)
        run_cmd(clone_cmd)


def git_create_patch(repo_dir, patch_file):
    cmd = 'git diff > {}'.format(patch_file)
    run_cmd(cmd, repo_dir)


def git_apply_patch(repo_dir, patch_file):
    cmd = 'git apply {}'.format(patch_file)
    run_cmd(cmd, repo_dir)


def load_config(package):
    config_file = (root_dir / 'third_party' / package / package).with_suffix('.json')

    with config_file.open() as f:
        config = json.load(f)

    return config['git_url'], config['tag'], config['files']


def get_package_dir(package):
    return root_dir / 'mindspore_xai' / 'third_party' / package


def get_patch_file(package):
    return (root_dir / 'third_party' / package / package).with_suffix('.patch')
