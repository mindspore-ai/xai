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
import shutil
from pathlib import Path

PACKAGES = ["lime", "shap"]
# i.e. xai/
root_dir = Path(__file__).resolve().parents[1]
cache_dir = root_dir / ".cache" / "patch"
cache_dir.mkdir(exist_ok=True, parents=True)


def get_source_code_from_url(url, package):
    """
    Download zipped source codes from url and unzip to cache directory

    Args:
        url (str): Url.
        package (str): package name.
    """
    package_file = cache_dir / "{}.zip".format(package)
    if not zipfile.is_zipfile(str(package_file)):
        urllib.request.urlretrieve(url, str(package_file))

    source_code_dir = cache_dir / package
    if source_code_dir.is_dir():
        shutil.rmtree(str(source_code_dir))

    # unzip to a tmp folder first, then move to source_code_dir.
    tmp_source_code_dir = source_code_dir.with_suffix(".tmp")
    if source_code_dir.is_dir():
        shutil.rmtree(str(tmp_source_code_dir))
    with zipfile.ZipFile(str(package_file), "r") as zip_ref:
        zip_ref.extractall(str(tmp_source_code_dir))

    license_file = next(tmp_source_code_dir.rglob("LICENSE"))
    license_file.parent.rename(source_code_dir)

    if tmp_source_code_dir.is_dir():
        shutil.rmtree(str(tmp_source_code_dir))

    # init git
    cmd = "git init -q && git add ."
    run_cmd(cmd, str(source_code_dir))


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


def git_create_patch(source_code_dir, patch_file):
    """
    Create Git patch.

    Args:
        source_code_dir (str): Local source code directory.
        patch_file (str): Patch file path.
    """
    # add '-N' to use diff on untracked files.
    cmd = 'git diff > {}'.format(patch_file)
    run_cmd(cmd, source_code_dir)


def git_apply_patch(source_code_dir, patch_file):
    """
    Apply Git patch.

    Args:
        source_code_dir (str): Local source code directory.
        patch_file (str): Patch file path.
    """
    # add --ignore-space-change and --ignore-whitespace to avoid error in Windows
    cmd = 'git apply --ignore-space-change --ignore-whitespace {}'.format(patch_file)
    run_cmd(cmd, source_code_dir)


def load_config(package):
    """
    Get Git version.

    Args:
        package (str): Target package.

    Returns:
        str, package url.
        list, list of string.
    """
    config_file = (root_dir / 'third_party' / package / package).with_suffix('.json')

    with config_file.open() as f:
        config = json.load(f)

    return config['url'], config['files']


def get_package_local_dir(package):
    """
    Get Git package local directory.

    Args:
        package (str): Target package.

    Returns:
        str, Git package local directory.
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
