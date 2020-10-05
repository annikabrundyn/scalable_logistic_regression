# Scalable Logistic Regression with PyTorch Lightning

This logistic regression model allows you to scale to much bigger datasets by having the option to train on multiple GPUS and TPUS. I implemented this model in the [PyTorch Lightning Bolts](https://github.com/PyTorchLightning/pytorch-lightning-bolts) library, where it has been rigorously tested and documented.

I've also implemented the SklearnDataModule - a class that conveniently puts any Numpy array dataset into PyTorch DataLoaders.

Train this model on any Numpy dataset as follows (here I'm demonstrating with the Sklearn Iris dataset):

```python
from pl_bolts.models.regression import LogisticRegression
from pl_bolts.datamodules import SklearnDataModule
import pytorch_lightning as pl

from sklearn.datasets import load_iris

# use any numpy or sklearn dataset
X, y = load_iris(return_X_y=True)
dm = SklearnDataModule(X, y)

# build model
model = LogisticRegression(input_dim=4, num_classes=3)

# fit
trainer = pl.Trainer(gpus=2, precision=16)
trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

trainer.test(test_dataloaders=dm.test_dataloader(batch_size=12))
```

To specify the number of GPUs or TPUs, just specify the flag in the Trainer. You can also enable 16-bit precision in the Trainer.

```python
# 1 GPU
trainer = pl.Trainer(gpus=1)

# 8 TPUs
trainer = pl.Trainer(tpu_cores=8)

# 16 GPUs and 16-bit precision
trainer = pl.Trainer(gpus=16, precision=16)
```
