# Contents

- [Contents](#contents)
    - [TB-Net Description](#tb-net-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Arguments](#script-arguments)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
            - [Inference and Explanation Performance](#inference-explanation-performance)
    - [Description of Random Situation](#description-of-random-situation)

# [TB-Net Description](#contents)

TB-Net is a knowledge graph based explainable recommender system.

Paper: Shendi Wang, Haoyang Li, Xiao-Hui Li, Caleb Chen Cao, Lei Chen. Tower Bridge Net (TB-Net): Bidirectional Knowledge Graph Aware Embedding Propagation for Explainable Recommender Systems

# [Model Architecture](#contents)

TB-Net constructs subgraphs in knowledge graph based on the interaction between users and items as well as the feature of items, and then calculates paths in the graphs using bidirectional conduction algorithm. Finally we can obtain explainable recommendation results.

# [Dataset](#contents)

We provide an example dataset that created from
[Interaction of users and games](https://www.kaggle.com/tamber/steam-video-games) and [games' feature data](https://www.kaggle.com/nikdavis/steam-store-games?select=steam.csv) of the game platform Steam that are publicly available on Kaggle.

Please refer to [Downloading Data Package](https://www.mindspore.cn/xai/docs/en/master/using_tbnet.html#downloading-data-package) for the way of obtaining the example dataset and the file format descriptions.

# [Environment Requirements](#contents)

- Hardware
    - Supports GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
    - [MindSpore XAI](https://www.mindspore.cn/xai/docs/en/master/index.html)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

Please refer to [Using TB-Net](https://www.mindspore.cn/xai/docs/en/master/using_tbnet.html) for the quick start guide.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
└─tbnet
  ├─README.md
  ├─README_CN.md
  ├─data
  │ └─steam
  │   ├─LICENSE
  │   ├─config.json                 # hyper-parameters and training configuration
  │   ├─src_infer.csv               # source datafile for inference
  │   ├─src_test.csv                # source datafile for evaluation
  │   └─src_train.csv               # source datafile for training
  ├─src
  │ ├─dataset.py                    # dataset loader
  │ ├─embedding.py                  # embedding module
  │ ├─metrics.py                    # model metrics
  │ ├─path_gen.py                   # data preprocessor
  │ ├─recommend.py                  # result aggregator
  │ └─tbnet.py                      # TB-Net architecture
  ├─export.py                       # export MINDIR/AIR script
  ├─preprocess.py                   # data pre-processing script
  ├─eval.py                         # evaluation script
  ├─infer.py                        # inference and explaining script
  └─train.py                        # training script
```

## [Script Arguments](#contents)

- preprocess.py arguments

```text
--dataset <DATASET>        'steam' dataset is supported currently (default 'steam')
--same_relation            only generate paths that relation1 is same as relation2
```

- train.py arguments

```text
--dataset <DATASET>        'steam' dataset is supported currently (default 'steam')
--train_csv <TRAIN_CSV>    the train csv datafile inside the dataset folder (default 'train.csv')
--test_csv <TEST_CSV>      the test csv datafile inside the dataset folder (default 'test.csv')
--epochs <EPOCHS>          number of training epochs (default 20)
--device_id <DEVICE_ID>    device id (default 0)
--run_mode <MODE>          run model in 'GRAPH' mode or 'PYNATIVE' mode (default 'GRAPH')
```

- eval.py arguments

```text
--checkpoint_id <CKPT_ID>    the id of the checkpoint(.ckpt) to be used
--dataset <DATASET>          'steam' dataset is supported currently (default 'steam')
--csv <CSV>                  the test csv datafile inside the dataset folder (default 'test.csv')
--device_id <DEVICE_ID>    device id (default 0)
--run_mode <MODE>          run model in 'GRAPH' mode or 'PYNATIVE' mode (default 'GRAPH')
```

- infer.py arguments

```text
--checkpoint_id <CKPT_ID>    the id of the checkpoint(.ckpt) to be used
--dataset <DATASET>          'steam' dataset is supported currently (default 'steam')
--csv <CSV>                  the infer csv datafile inside the dataset folder (default 'infer.csv')
--items <ITEMS>              no. of items to be recommended (default 3)
--explanations <EXP>         no. of recommendation reasons to be shown (default 3)
--device_id <DEVICE_ID>      device id (default 0)
--run_mode <MODE>            run model in 'GRAPH' mode or 'PYNATIVE' mode (default 'GRAPH')
```

- export.py arguments

```text
--config_path <CFG_PATH>        config (config.json) file path
--checkpoint_path <CKPT_PATH>   checkpoint (.ckpt) file path
--file_name <FILENAME>          export filename
--file_format <FORMAT>          export format 'MINDIR' or 'AIR' (default 'MINDIR')
--device_id <DEVICE_ID>         device id (default 0)
--run_mode <MODE>               run model in 'GRAPH' mode or 'PYNATIVE' mode (default 'GRAPH')
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | GPU                                                         |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | TB-Net                                                      |
| Resource                   | Tesla V100-SXM2-32GB                                        |
| Uploaded Date              | 2021-08-01                                                  |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | steam                                                       |
| Training Parameter         | epoch=20, batch_size=1024, lr=0.001                         |
| Optimizer                  | Adam                                                        |
| Loss Function              | Sigmoid Cross Entropy                                       |
| Outputs                    | AUC=0.8596, Accuracy=0.7761                                 |
| Loss                       | 0.57                                                        |
| Speed                      | 1pc: 90ms/step                                              |
| Total Time                 | 1pc: 297s                                                   |
| Checkpoint for Fine Tuning | 104.66M (.ckpt file)                                        |

### Evaluation Performance

| Parameters                | GPU                           |
| ------------------------- | ----------------------------- |
| Model Version             | TB-Net                        |
| Resource                  | Tesla V100-SXM2-32GB          |
| Uploaded Date             | 2021-08-01                    |
| MindSpore Version         | 1.3.0                         |
| Dataset                   | steam                         |
| Batch Size                | 1024                          |
| Outputs                   | AUC=0.8252, Accuracy=0.7503   |
| Total Time                | 1pc: 5.7s                     |

### Inference and Explaining Performance

| Parameters                | GPU                                   |
| --------------------------| ------------------------------------- |
| Model Version             | TB-Net                                |
| Resource                  | Tesla V100-SXM2-32GB                  |
| Uploaded Date             | 2021-08-01                            |
| MindSpore Version         | 1.3.0                                 |
| Dataset                   | steam                                 |
| Outputs                   | Recommendation Result and Explanation |
| Total Time                | 1pc: 3.66s                            |

# [Description of Random Situation](#contents)

- Initialization of embedding matrix in `tbnet.py` and `embedding.py`.
