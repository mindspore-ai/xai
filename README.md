# Explainable AI (XAI)

[查看中文](./README_CN.md)

<!-- TOC --->

- [What is Explainable AI (XAI)](#what-is-explainable-ai-xai)
    - [System Architecture](#system-architecture)
    - [Internal Components](#internal-components)
- [Installation](#installation)
    - [System Requirements](#system-requirements)
    - [Install by pip](#install-by-pip)
    - [Install from Source Code](#install-from-source-code)
    - [Verifying Successful Installation and Version](#verifying-successful-installation-and-version)
- [Note](#note)
- [Quick Start](#quick-start)
- [Docs](#docs)
- [Community](#community)
    - [Governance](#governance)
- [Contributing](#contributing)
- [License](#license)

<!-- /TOC -->

## What is Explainable AI (XAI)

This is an explainable AI framework base on [MindSpore](https://www.mindspore.cn/en). Currently, most deep learning models are black-box models with good performance but poor explainability. The model explanation module aims to provide users with explanation of the model decision basis, help users better understand the model, trust the model, and improve the model when an error occurs in the model. Besides a variety of explanation methods, we also provide a set of evaluation methods to evaluate the explanation methods from various dimensions. It helps users compare and select the explanation methods that are most suitable for a particular scenario.

### System Architecture

![sys_arch](./images/sys_arch_en.png)

### Internal Components

![internal](./images/internal_en.png)

## Installation

### System Requirements

- OS: EulerOS-aarch64, CentOS-aarch64, CentOS-x86, Ubuntu-aarch64 or Ubuntu-x86
- Device: Ascend 910 or GPU CUDA 10.1
- Python 3.7.5 or above
- MindSpore 1.3

### Install by pip

Download the `.whl` package from [MindSpore XAI download page](https://www.mindspore.cn/versions/en) and install with `pip`.

```bash
pip install mindspore_xai-{version}-py3-none-any.whl
```

### Install from Source Code

1. Download source code from gitee.com:

```bash
git clone https://giee.com/mindspore/xai.git
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

4. Opitonally, you may build a `.whl` package for installation:

```bash
bash package.sh
pip install output/mindspore_xai-{version}-py3-none-any.whl
```

### Verifying Successful Installation and Version

Upon successful installation, importing 'mindspore_xai' module in python will cause no error:

```python
import mindspore_xai
print(mindspore_xai.__version__)
```

## Note

[MindInsight](https://gitee.com/mindspore/mindinsight/blob/master/README.md) is an optional tool for visualizing the model explanation from XAI. Please checkout [Tutorials](https://www.mindspore.cn/en) for more details.

## Quick Start

For a quick start of generating model explanations, please checkout [Tutorials](https://www.mindspore.cn/en).

## Docs

For more details about installation, tutorials and APIs, please checkout [User Documentation](https://www.mindspore.cn/en).

## Community

### Governance

Checkout how MindSpore Open Governance [works](<https://gitee.com/mindspore/community/blob/master/governance.md>)

## Contributing

Welcome contributions. See our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for more details.

## License

[Apache License 2.0](LICENSE)
