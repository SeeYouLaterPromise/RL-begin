import abc
import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
from utils.metric_plot import plot_loss, plot_acc, plot_classify_metrics
import time
from utils.logger import save_training_config

"""
save_dir: log save path
"""
class TrainBase(abc.ABC):
    def __init__(self, model, optimizer, criterion, scheduler, train_loader, val_loader, save_dir, draw=True, training_config=None, device=None, experiment_name=None):
        if experiment_name is None:
            experiment_name = time.strftime("%d-%H-%M", time.localtime())

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
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
        best_test_accuracy = 0.0
        best_test_logs = None
        for epoch in range(1, epochs + 1):
            # print(f"\nEpoch {epoch}")
            train_metrics = self.train_epoch(epoch)
            val_metrics, best_test_logs, best_test_accuracy = self.val_epoch(epoch, best_test_logs, best_test_accuracy)
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
            plot_loss(train_df, val_df, self.save_dir, name)
            plot_acc(train_df, val_df, self.save_dir, name)
            plot_classify_metrics(val_df, self.save_dir, name)

        torch.save(self.model.state_dict(), self.last_model_path)
        print(f"Last model saved to {self.last_model_path}")
        return train_df, val_df

    def train_epoch(self, epoch):
        # —— Training Phase with tqdm ——
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in train_bar:
            self.optimizer.zero_grad()
            loss, preds, targets = self.train_step(batch)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total += targets.size(0)
            correct += (preds == targets).sum().item()

        train_loss = running_loss / len(self.train_loader)
        train_acc = 100 * correct / total

        print(f" Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")

        return {
                'Epoch': epoch,
                'Train Loss': train_loss,
                'Train Accuracy': train_acc
                }

    def val_epoch(self, epoch, best_test_logs, best_test_accuracy):
        # —— Validating Phase with tqdm ——
        self.model.eval()
        test_loss, correct, total = 0.0, 0, 0
        all_preds, all_tgts = [], []
        batch_accs, batch_ps, batch_rs, batch_fs = [], [], [], []

        with torch.no_grad():
            for batch in self.val_loader:
                loss, preds, targets = self.val_step(batch)
                test_loss += loss.item()

                all_preds.extend(preds.cpu().numpy())
                all_tgts.extend(targets.cpu().numpy())
                total += targets.size(0)
                correct += (preds == targets).sum().item()

                # batch‐wise metrics
                ba = 100 * (preds == targets).sum().item() / targets.size(0)
                bp = precision_score(targets.cpu(), preds.cpu(), average='macro', zero_division=0)
                br = recall_score(targets.cpu(), preds.cpu(), average='macro')
                bf = f1_score(targets.cpu(), preds.cpu(), average='macro')
                batch_accs.append(ba)
                batch_ps.append(bp)
                batch_rs.append(br)
                batch_fs.append(bf)

        test_loss /= len(self.val_loader)
        test_acc = 100 * correct / total
        prec = precision_score(all_tgts, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_tgts, all_preds, average='macro')
        f1 = f1_score(all_tgts, all_preds, average='macro')
        cm = confusion_matrix(all_tgts, all_preds)

        # stddevs
        test_acc_std = np.std(batch_accs)
        prec_std = np.std(batch_ps)
        rec_std = np.std(batch_rs)
        f1_std = np.std(batch_fs)

        # scheduler
        self.scheduler.step(test_loss)

        print(f" Val:   loss={test_loss:.4f}, acc={test_acc:.2f}% (±{test_acc_std:.2f}), "
              f"prec={prec:.2f}(±{prec_std:.2f}), rec={rec:.2f}(±{rec_std:.2f}), f1={f1:.2f}(±{f1_std:.2f})")

        test_log = {
                'Epoch': epoch,
                'Test Loss': test_loss,
                'Test Accuracy': test_acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1,
                'Test Accuracy Std': test_acc_std,
                'Precision Std': prec_std,
                'Recall Std': rec_std,
                'F1 Score Std': f1_std,
                'Confusion Matrix': cm
            }

        #  & early stopping
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            # no_improvement = 0
            best_test_logs = test_log
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"Best model saved with acc={test_acc:.2f}% → {self.best_model_path}")

        return test_log, best_test_logs, best_test_accuracy

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


