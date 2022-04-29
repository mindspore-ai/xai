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
import ssl
import filecmp

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

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
    print("downloading {}".format(url))
    package_file = cache_dir / "{}.zip".format(package)
    if not zipfile.is_zipfile(str(package_file)):
        with urllib.request.urlopen(url, context=ctx) as u, \
                open(package_file, 'wb') as f:
            f.write(u.read())

    source_code_dir = cache_dir / package
    if source_code_dir.is_dir():
        rm_dir(source_code_dir)

    # unzip to a tmp folder first, then move to source_code_dir.
    tmp_source_code_dir = source_code_dir.with_suffix(".tmp")
    if source_code_dir.is_dir():
        rm_dir(tmp_source_code_dir)
    with zipfile.ZipFile(str(package_file), "r") as zip_ref:
        zip_ref.extractall(str(tmp_source_code_dir))

    license_file = next(tmp_source_code_dir.rglob("LICENSE"))
    license_file.parent.rename(source_code_dir)

    if tmp_source_code_dir.is_dir():
        rm_dir(tmp_source_code_dir)

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


def rm_dir(directory):
    """Remove a directory."""
    if os.name == 'nt':
        subprocess.call(f'RMDIR /S /Q {str(directory)}', shell=True)
    else:
        shutil.rmtree(str(directory))


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

    if not config_file.is_file():
        raise ValueError("Invalid package '{}'".format(package))

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


def list_third_party_src_pkg():
    """
    List third party source packages

    Returns:
        list, list of string.
    """
    dirs = (root_dir / 'third_party').glob("*")
    return [d.stem for d in dirs if d.is_dir()]


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


def apply_patch(package):
    """
    Apply patch

    Args:
        package (str): package name, or "all" if you want to apply patch for all third party packages
    """
    if package == "all":
        packages = list_third_party_src_pkg()
    else:
        packages = [package]

    for pkg in packages:
        print("applying patch for {}".format(pkg))
        url, files = load_config(pkg)
        # The package location inside the xai, e.g. xai/mindspore_xai/third_party/lime
        local_dir = get_package_local_dir(pkg)
        patch_file = get_patch_file(pkg)

        source_code_dir = cache_dir / pkg

        get_source_code_from_url(url, pkg)
        git_apply_patch(str(source_code_dir), str(patch_file))

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
        rm_dir(source_code_dir)

        # create __init__.py
        (local_dir / "__init__.py").touch()

        print('patch applied to {}'.format(local_dir))
