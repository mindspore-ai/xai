# MindSpore XAI Examples

<!-- TOC -->

- [Description](#description)
- [Preparations](#preparations)

<!-- /TOC -->

## Description

The follow files are example scripts of using XAI:

```bash
examples/
├── xai_examples_data/
│    ├── ckpt/
│    │    └── resent50.ckpt
│    ├── train/
│    └── test/
├── common/
│    ├── dataset.py
│    └── resnet.py
├── using_cv_benchmarks.py
├── using_cv_explainers.py
├── using_rise_plus.py
├── using_tabsim.py
└── using_tabular_explainers.py
```

- `xai_examples_data/`: The data package has to be downloaded for the examples, please refer to [Preparations](#preparations) for the details.
- `xai_examples_data/ckpt/resent50.ckpt`: ResNet50 checkpoint file.
- `xai_examples_data/test`: Example test dataset.
- `xai_examples_data/train`: Example training dataset.
- `common/dataset.py`: Dataset loader.
- `common/resnet.py`: ResNet model definitions.
- `using_cv_benchmarks.py`: Example of using cv benchmarks, please refer to [Using Benchmarks](https://www.mindspore.cn/xai/docs/en/master/using_cv_benchmarks.html) for the details.
- `using_cv_explainers.py`: Example of using cv explainers, please refer to [Using Explainers](https://www.mindspore.cn/xai/docs/en/master/using_cv_explainers.html) for the details.
- `using_rise_plus.py`: Example of using RISEPlus explainer, the way of using it is different from other explainers, please refer to [Using RISEPlus](https://www.mindspore.cn/xai/docs/en/master/using_cv_explainers.html#using-riseplus) for the details.
- `using_tabsim.py`: Example of using TabSim, please refer to [Using TabSim](https://www.mindspore.cn/xai/docs/en/master/using_tabsim.html) for the details.
- `using_tabular_explainers.py`: Example of using tabular explainers, please refer to [Using Tabular Explainers](https://www.mindspore.cn/xai/docs/en/master/using_tabular_explainers.html) for the details.

## Preparations

Proper installations of [MindSpore](https://www.mindspore.cn/install) and [XAI](https://www.mindspore.cn/xai/docs/en/master/installation.html) are required. Besides that, `xai_examples_data` has to be downloaded as well:

```bash
wget https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/xai/xai_examples_data.tar.gz
```

Then extract the data package and move it underneath `xai/examples/`:

```bash
tar -xf xai_examples_data.tar.gz
mv xai_examples_data xai/examples/
```

Now example scripts can be run under the working directory of `xai/examples/`:

```bash
cd xai/examples/
python using_cv_explainers.py
```
