from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.classification import accuracy
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer


class LogisticRegression(pl.LightningModule):

    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 bias: bool = True,
                 learning_rate: float = 1e-4,
                 optimizer: Optimizer = Adam,
                 l1_strength: float = 0.0,
                 l2_strength: float = 0.0,
                 **kwargs):
        """
        Logistic regression model
        Args:
            input_dim: number of dimensions of the input (at least 1)
            num_classes: number of class labels (binary: 2, multi-class: >2)
            bias: specifies if a constant or intercept should be fitted (equivalent to fit_intercept in sklearn)
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default='Adam')
            l1_strength: L1 regularization strength (default=None)
            l2_strength: L2 regularization strength (default=None)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer

        self.linear = nn.Linear(in_features=self.hparams.input_dim, out_features=self.hparams.num_classes, bias=bias)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        # flatten any input
        x = x.view(x.size(0), -1)

        y_hat = self(x)

        # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = F.cross_entropy(y_hat, y, reduction='sum')

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum(param.pow(2).sum() for param in self.parameters())
            loss += self.hparams.l2_strength * l2_reg

        loss /= x.size(0)

        tensorboard_logs = {'train_ce_loss': loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'loss': loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        acc = accuracy(y_hat, y)
        return {'val_loss': F.cross_entropy(y_hat, y), 'acc': acc}

    def validation_epoch_end(self, outputs):
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_ce_loss': val_loss, 'val_acc': acc}
        progress_bar_metrics = tensorboard_logs
        return {
            'val_loss': val_loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        acc = accuracy(y_hat, y)
        return {'test_loss': F.cross_entropy(y_hat, y), 'acc': acc}

    def test_epoch_end(self, outputs):
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_ce_loss': test_loss, 'test_acc': acc}
        progress_bar_metrics = tensorboard_logs
        return {
            'test_loss': test_loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--input_dim', type=int, default=None)
        parser.add_argument('--num_classes', type=int, default=None)
        parser.add_argument('--bias', default='store_true')
        parser.add_argument('--batch_size', type=int, default=16)
        return parser


def cli_main():
    from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule

    pl.seed_everything(1234)

    # Example: Iris dataset in Sklearn (4 features, 3 class labels)
    try:
        from sklearn.datasets import load_iris
    except ImportError:
        raise ImportError('You want to use `sklearn` which is not installed yet,'  # pragma: no-cover
                          ' install it with `pip install sklearn`.')

    X, y = load_iris(return_X_y=True)
    loaders = SklearnDataModule(X, y)

    # args
    parser = ArgumentParser()
    parser = LogisticRegression.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # model
    # model = LogisticRegression(**vars(args))
    model = LogisticRegression(input_dim=4, num_classes=3, l1_strength=0.01, learning_rate=0.01)

    # train
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, loaders.train_dataloader(args.batch_size), loaders.val_dataloader(args.batch_size))


if __name__ == '__main__':
    cli_main()
