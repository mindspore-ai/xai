# Explainable AI (XAI)

[查看中文](./README_CN.md)

<!-- TOC --->

- [What is Explainable AI (XAI)](#what-is-explainable-ai-xai)
    - [System Architecture](#system-architecture)
- [Installation](#installation)
    - [System Requirements](#system-requirements)
    - [Installing by pip](#installing-by-pip)
    - [Installing from Source Code](#installing-from-source-code)
    - [Installation Verification](#installation-verification)
- [Note](#note)
- [Quick Start and API References](#quick-start-and-api-references)
- [Community](#community)
    - [Governance](#governance)
- [Contributing](#contributing)
- [License](#license)

<!-- /TOC -->

## What is Explainable AI (XAI)

XAI is an explainable AI toolbox based on [MindSpore](https://www.mindspore.cn/en). Currently, most deep learning models are black-box models with good performance but poor explainability. XAI aims to provide users decision explanations of the model, helps users better understand the model, trust the model, and improve the model when an error occurs. Besides a variety of explanation methods, we also provide a set of evaluation methods to evaluate the explanation methods from various dimensions. It helps users compare and select the explanation methods that are most suitable for a particular scenario.

### System Architecture

![sys_arch](./images/sys_arch_en.png)

## Installation

### System Requirements

- OS: EulerOS-aarch64, CentOS-aarch64, CentOS-x86, Ubuntu-aarch64 or Ubuntu-x86
- Device: Ascend 910 or GPU CUDA 10.1, 11.1
- Python 3.7.5 or 3.9.0
- MindSpore 1.7 or above

### Installing by pip

The simplest way to do it:

```bash
pip install mindspore-xai
```

Or you may install with a downloaded `.whl` package from [MindSpore XAI download page](https://www.mindspore.cn/versions/en):

```bash
pip install mindspore_xai-{version}-py3-none-any.whl
```

### Installing from Source Code

1. Download source code from gitee.com:

```bash
git clone https://gitee.com/mindspore/xai.git
```

2. Install the dependency python modules:

```bash
cd xai
pip install -r requirements.txt
```

3. Install the XAI module from source code:

```bash
python setup.py install
```

4. Optionally, you may build a `.whl` package for installation without step 3:

```bash
bash package.sh
pip install output/mindspore_xai-{version}-py3-none-any.whl
```

### Installation Verification

Upon successful installation, importing 'mindspore_xai' module in python will cause no error:

```python
import mindspore_xai
print(mindspore_xai.__version__)
```

## Quick Start and API References

For the quick start guide and API references, please checkout [XAI Homepage](https://www.mindspore.cn/xai/docs/en/master/index.html).

## Community

### Governance

Checkout how MindSpore Open Governance [works](<https://gitee.com/mindspore/community/blob/master/governance.md>)

## Contributing

Welcome contributions. See our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for more details.

## License

[Apache License 2.0](LICENSE)
