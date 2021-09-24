# Copyright 2021 Huawei Technologies Co., Ltd.All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup."""

import sys
import os
import shutil
import stat
import platform
import shlex
import subprocess
import types
from importlib import import_module
from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py
from setuptools.command.install import install


package_name = 'mindspore-xai'
import_name = 'mindspore_xai'


def get_version():
    """
    Get version with date.

    Returns:
        str, xai version.
    """
    machinery = import_module('importlib.machinery')
    version_path = os.path.join(os.path.dirname(__file__), import_name, '_version.py')
    module_name = '__msxaiversion__'
    version_module = types.ModuleType(module_name)
    loader = machinery.SourceFileLoader(module_name, version_path)
    loader.exec_module(version_module)
    return version_module.__version__


def get_platform():
    """
    Get platform name.

    Returns:
        str, platform name in lowercase.
    """
    return platform.system().lower()


def get_description():
    """
    Get description.

    Returns:
        str, wheel package description.
    """
    os_info = get_platform()
    cpu_info = platform.machine()

    cmd = "git log --format='[sha1]:%h, [branch]:%d' -1"
    process = subprocess.Popen(
        shlex.split(cmd),
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, _ = process.communicate()
    if not process.returncode:
        git_version = stdout.decode().strip()
        return '%s platform: %s, cpu: %s, git version: %s' % (package_name, os_info, cpu_info, git_version)

    return '%s platform: %s, cpu: %s' % (package_name, os_info, cpu_info)


def get_install_requires():
    """
    Get install requirements.

    Returns:
        list, list of dependent packages.
    """
    with open('requirements.txt') as file:
        return file.read().splitlines()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


def run_script(script):
    """
    Run script.

    Args:
        script (str): Target script file path.

    Returns:
        int, return code.
    """
    cmd = '/bin/bash {}'.format(script)
    process = subprocess.Popen(
        shlex.split(cmd),
        shell=False
    )
    return process.wait()


class EggInfo(egg_info):
    """Egg info."""

    def run(self):
        egg_info_dir = os.path.join(os.path.dirname(__file__), f'{import_name}.egg-info')
        shutil.rmtree(egg_info_dir, ignore_errors=True)
        super().run()
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """Build py files."""

    def run(self):
        xai_lib_dir = os.path.join(os.path.dirname(__file__), 'build', 'lib', import_name)
        shutil.rmtree(xai_lib_dir, ignore_errors=True)
        super().run()
        update_permissions(xai_lib_dir)


class Install(install):
    """Install."""

    def run(self):
        super().run()
        if sys.argv[-1] == 'install':
            pip = import_module('pip')
            xai_dir = os.path.join(os.path.dirname(pip.__path__[0]), import_name)
            update_permissions(xai_dir)


if __name__ == '__main__':
    version_info = sys.version_info
    if (version_info.major, version_info.minor) < (3, 7):
        sys.stderr.write('Python version should be at least 3.7\r\n')
        sys.exit(1)

    setup(
        name=package_name,
        version=get_version(),
        author='The MindSpore XAI Authors',
        author_email='contact@mindspore.cn',
        url='https://www.mindspore.cn/xai',
        download_url='https://gitee.com/mindspore/xai/tags',
        project_urls={
            'Sources': 'https://gitee.com/mindspore/xai',
            'Issue Tracker': 'https://gitee.com/mindspore/xai/issues',
        },
        description=get_description(),
        packages=find_packages(),
        platforms=[get_platform()],
        include_package_data=True,
        cmdclass={
            'egg_info': EggInfo,
            'build_py': BuildPy,
            'install': Install,
        },
        python_requires='>=3.7',
        install_requires=get_install_requires(),
        classifiers=[
            'Development Status :: 1 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords='mindspore machine learning xai',
    )
