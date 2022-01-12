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
"""Utilities."""
import os
import subprocess
import json
import urllib.request
import zipfile
import tempfile
import shutil
from pathlib import Path

PACKAGES = ["lime"]
# i.e. xai/
root_dir = Path(__file__).resolve().parents[1]


def get_repo_from_url(url, repo_dir):
    """
    Download zipped repo from url and unzip to local directory

    Args:
        url (str): Url.
        repo_dir (Path): Local directory.
    """
    package = repo_dir.name
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_repo_dir = Path(tmp_dir) / package

        # download zip file
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp_file:
            urllib.request.urlretrieve(url, tmp_file.name)

            # unzip
            with zipfile.ZipFile(tmp_file.name, "r") as zip_ref:
                zip_ref.extractall(str(tmp_repo_dir))

        shutil.move(str(next(tmp_repo_dir.glob("{}*".format(package)))), str(repo_dir))

    # init git
    cmd = "git init -q && git add ."
    run_cmd(cmd, str(repo_dir))


def get_git_version():
    """
    Get Git version.

    Returns:
        list, git version.
    """
    out = subprocess.check_output(['git', '--version']).strip().decode('utf-8')
    version = out.split(' ')[2]
    # e.g. [2, 30, 1]
    return [int(x) for x in version.split('.')[:3]]


def run_cmd(cmd, directory=None):
    """
    Run command.

    Args:
        cmd (str): command.
        directory (str): working directory.
    """
    cwd = Path.cwd()
    # temporary switch working directory
    if directory is not None:
        os.chdir(str(directory))

    if subprocess.call(cmd, shell=True) != 0:
        raise RuntimeError("failed to run command: `{}`".format(cmd))

    if directory is not None:
        os.chdir(str(cwd))


def git_clone(git_url, repo_dir, files, tag='master'):
    """
    Clones a repository into a local directory

    Args:
        git_url (str): Git url.
        repo_dir (str): Local directory.
        files (list): list of src/dst mapping.
        tag (str): Target Git repo tag.
    """
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
    """
    Create Git patch.

    Args:
        repo_dir (str): Local repo directory.
        patch_file (str): Patch file path.
    """
    cmd = 'git diff > {}'.format(patch_file)
    run_cmd(cmd, repo_dir)


def git_apply_patch(repo_dir, patch_file):
    """
    Apply Git patch.

    Args:
        repo_dir (str): Local repo directory.
        patch_file (str): Patch file path.
    """
    cmd = 'git apply {}'.format(patch_file)
    run_cmd(cmd, repo_dir)


def load_config(package):
    """
    Get Git version.

    Args:
        package (str): Target package.

    Returns:
        str, package url.
        list, list of src/dst mapping.
    """
    config_file = (root_dir / 'third_party' / package / package).with_suffix('.json')

    with config_file.open() as f:
        config = json.load(f)

    return config['url'], config['files']


def get_package_dir(package):
    """
    Get Git package directory.

    Args:
        package (str): Target package.

    Returns:
        str, Git package directory.
    """

    return root_dir / 'mindspore_xai' / 'third_party' / package


def get_patch_file(package):
    """
    Get patch file path.

    Args:
        package (str): Target package.

    Returns:
        str, Patch file path.
    """

    return (root_dir / 'third_party' / package / package).with_suffix('.patch')
