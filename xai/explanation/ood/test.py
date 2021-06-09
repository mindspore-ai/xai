import numpy as np
import mindspore as ms
import mindspore.dataset as de
from mindspore.train.callback import Callback

from xai.explanation import OODNet
from xai.explanation import OODResNet50


num_classes = 10
num_samples = 128
batch_size = 64
image_size = 224
channel = 3


class Print_info(Callback):
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print(f"epoch {cb_params.cur_epoch_num} step {cb_params.cur_step_num}")

    def epoch_end(self, _):
        print("epoch_end")


def test_infer():

    classifier = OODResNet50(num_classes)
    ood_net = OODNet(classifier, num_classes)
    ood_net.set_train(False)

    batch_x = ms.Tensor(np.random.random((1, 3, image_size, image_size)), dtype=ms.float32)
    ood_scores = ood_net.score(batch_x)
    print(f'ood_scores: {ood_scores}')


def ds_generator():
    for i in range(num_samples):
        image = np.random.random((channel, image_size, image_size)).astype(np.float32)
        labels = np.random.randint(0, num_classes, 3)
        one_hot = np.zeros(num_classes, dtype=np.float32)
        for label in labels:
            one_hot[label] = 1.0
        yield image, one_hot


def test_train():

    ds = de.GeneratorDataset(source=ds_generator, num_samples=num_samples,
                             column_names=['data', 'label'], column_types=[ms.float32, ms.float32])
    ds = ds.batch(batch_size)
    ds.dataset_size = int(num_samples / batch_size)

    classifier = OODResNet50(num_classes)
    ood_net = OODNet(classifier, num_classes)
    ood_net.train(ds, callbacks=Print_info(), epoch=60, multi_label=True)

    batch_x = ms.Tensor(np.random.random((1, channel, 224, 224)), dtype=ms.float32)
    ood_scores = ood_net.score(batch_x)
    print(f'ood_scores.shape {ood_scores.shape}')
    print(str(ood_scores))


if __name__ == "__main__":
    test_train()
