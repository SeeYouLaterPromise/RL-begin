import abc
import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os

from supervised.ExamManager import ExamManager
from utils.metric_plot import plot_loss, plot_acc, plot_classify_metrics
from utils.logger import save_training_config

"""
save_dir: log save path
"""
class TrainBase(abc.ABC):
    def __init__(self, model, optimizer, criterion, scheduler, train_loader, val_loader, save_dir, draw=True, training_config=None, device=None):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        weight_path = os.path.join(str(self.save_dir), "weights")
        os.makedirs(weight_path, exist_ok=True)
        self.last_model_path = os.path.join(str(weight_path), "last_model.pt")
        self.best_model_path = os.path.join(str(weight_path), "best_model.pt")
        self.draw = draw
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if training_config is not None:
            save_training_config(training_config, str(self.save_dir))

    def run(self, epochs: int, name: str):
        train_logs = []
        val_logs = []
        best_mean_distance = 0.0
        best_test_logs = None
        for epoch in range(1, epochs + 1):
            # print(f"\nEpoch {epoch}")
            train_metrics = self.train_epoch(epoch)

            val_metrics, best_test_logs, best_mean_distance = self.val_epoch(epoch, best_test_logs, best_mean_distance, )
            train_logs.append(train_metrics)
            val_logs.append(val_metrics)
            # self.on_epoch_end(epoch, train_metrics, val_metrics)

        # Save logs to CSV
        train_df = pd.DataFrame(train_logs)
        train_df.to_csv(f'{self.save_dir}/{name}_training_logs.csv', index=False)

        val_df = pd.DataFrame(val_logs)
        val_df.to_csv(f"{self.save_dir}/{name}_test_logs.csv", index=False)

        best_df = pd.DataFrame([best_test_logs])
        best_df.to_csv(f'{self.save_dir}/{name}_best_test_logs.csv', index=False)

        if self.draw:
            plot_loss(train_df, val_df, self.save_dir, name=name)
            plot_acc(train_df, val_df, self.save_dir, name)
            plot_classify_metrics(val_df, self.save_dir, name)

        torch.save(self.model.state_dict(), self.last_model_path)
        print(f"Last model saved to {self.last_model_path}")
        return train_df, val_df



    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        # 可以在这里做日志记录、调整学习率、早停判断、保存模型等
        # print(f"[Epoch {epoch}] "
        #       f"Train loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.2f}% | "
        #       f"Val loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.2f}%")
        pass

    @abc.abstractmethod
    def train_step(self, batch):
        """
        子类必须实现：
        - 解包 batch
        - 前向计算 outputs = model(...)
        - 计算 loss = criterion(outputs, targets)
        - 返回 (loss, preds, targets)
        """
        pass

    @abc.abstractmethod
    def val_step(self, batch):
        """
        子类必须实现：
        - 解包 batch
        - 前向计算 outputs = model(...)
        - 计算 loss = criterion(outputs, targets)
        - 返回 (loss, preds, targets)
        """
        pass


