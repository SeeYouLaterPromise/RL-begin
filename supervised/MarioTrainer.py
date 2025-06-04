from supervised.ExamManager import ExamManager
from utils.ClassifyTrainBase import TrainBase
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

class MarioTrainer(TrainBase):
    def __init__(self, model, optimizer, criterion, scheduler, train_loader, val_loader,
                 save_dir, penalty_mode="False", penalty_weight=0.5, penalty_type="maxprob", draw=True, training_config=None, device=None):
        
        # 正确地调用父类构造函数
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            draw=draw,
            training_config=training_config,
            device=device,
        )
        
        self.penalty_mode = penalty_mode
        self.penalty_weight = penalty_weight
        self.penalty_type = penalty_type
    
    def maxprob_loss(self, logits):
        """
        logits: (batch_critical, num_actions)
        惩罚：最大 softmax 概率不能太高（不自信）
        """
        probs = torch.softmax(logits, dim=1)
        max_probs, _ = probs.max(dim=1)
        return max_probs.mean()  # 惩罚的是越自信，loss 越高
    
    def penalty_loss(self, logits):
        """
        logits: (N, C)
        根据设定的策略返回 penalty loss
        """
        probs = torch.softmax(logits, dim=1)

        if self.penalty_type == "maxprob":
            max_probs, _ = probs.max(dim=1)
            return max_probs.mean()

        elif self.penalty_type == "neglog":
            max_probs, _ = probs.max(dim=1)
            return -torch.log(1.0 - max_probs + 1e-6).mean()  # 加 epsilon 防止 log(0)

        elif self.penalty_type == "kl_uniform":
            uniform = torch.full_like(probs, 1.0 / probs.size(1))
            kl_div = probs * (probs.log() - uniform.log())
            return kl_div.sum(dim=1).mean()

        else:
            raise ValueError(f"Unknown penalty_type: {self.penalty_type}. Supported: ['maxprob', 'neglog', 'kl_uniform']")



    def loss(self, outputs, targets, is_critical=None):
        loss_bc = self.criterion(outputs, targets)

        if self.penalty_mode and is_critical is not None and is_critical.any():
            crit_logits = outputs[is_critical]
            loss_penalty = self.penalty_loss(crit_logits)
            total_loss = loss_bc + self.penalty_weight * loss_penalty

            # ✅ 记录组件
            self._last_bc_loss = loss_bc.item()
            self._last_penalty_loss = loss_penalty.item()
        else:
            total_loss = loss_bc
            self._last_bc_loss = loss_bc.item()
            self._last_penalty_loss = 0.0

        return total_loss



    def train_step(self, batch):
        if self.penalty_mode:
            imgs, targets, is_critical = batch
            imgs, targets, is_critical = imgs.to(self.device), targets.to(self.device), is_critical.to(self.device)
            outputs = self.model(imgs)
            loss = self.loss(outputs, targets, is_critical)
            preds = outputs.argmax(dim=1)
        else:
            imgs, targets = batch
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            outputs = self.model(imgs)                   # (batch_size, num_actions)
            loss = self.loss(outputs, targets, None)     
            preds = outputs.argmax(dim=1)                # 分类预测
        return loss, preds, targets

    def val_step(self, batch, render=False):
        if self.penalty_mode:
            imgs, targets, is_critical = batch
            imgs, targets, is_critical = imgs.to(self.device), targets.to(self.device), is_critical.to(self.device)
            outputs = self.model(imgs)
            loss = self.loss(outputs, targets, is_critical)
            preds = outputs.argmax(dim=1)
        else:
            imgs, targets = batch
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            outputs = self.model(imgs)
            loss = self.loss(outputs, targets)
            preds = outputs.argmax(dim=1)
        return loss, preds, targets



    def train_epoch(self, epoch):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        bc_losses, penalty_losses = [], []

        train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch in train_bar:
            self.optimizer.zero_grad()
            loss, preds, targets = self.train_step(batch)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total += targets.size(0)
            correct += (preds == targets).sum().item()

            # 🔹 来自 self.loss() 中记录的子损失
            bc_losses.append(self._last_bc_loss)
            penalty_losses.append(self._last_penalty_loss)

        train_loss = running_loss / len(self.train_loader)
        train_acc = 100 * correct / total
        train_bc_loss = sum(bc_losses) / len(bc_losses)
        train_penalty_loss = sum(penalty_losses) / len(penalty_losses)

        print(f" Train: loss={train_loss:.4f}, acc={train_acc:.2f}%, "
            f"BC={train_bc_loss:.4f}, Penalty={train_penalty_loss:.4f}")

        # ✅ 可视化：本轮每 batch 的 loss 曲线
        # if self.draw:
        #     import matplotlib.pyplot as plt
        #     plt.plot(bc_losses, label="BC Loss")
        #     plt.plot(penalty_losses, label="Penalty Loss")
        #     plt.title(f"Train Loss Composition (Epoch {epoch})")
        #     plt.xlabel("Batch Index")
        #     plt.ylabel("Loss")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.tight_layout()

        #     # 保存为图像文件（自动命名）
        #     save_path = os.path.join(self.save_dir, f"train_loss_epoch_{epoch}.png")
        #     plt.savefig(save_path)
        #     plt.close()

        return {
            'Epoch': epoch,
            'Train Loss': train_loss,
            'Train Accuracy': train_acc,
            'Train BC Loss': train_bc_loss,
            'Train Penalty Loss': train_penalty_loss
        }



    def val_epoch(self, epoch, best_test_logs, best_mean_distance):
        self.model.eval()

        test_loss, correct, total = 0.0, 0, 0
        all_preds, all_tgts = [], []
        batch_accs, batch_ps, batch_rs, batch_fs = [], [], [], []

        bc_losses, penalty_losses = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                loss, preds, targets = self.val_step(batch)
                test_loss += loss.item()

                all_preds.extend(preds.cpu().numpy())
                all_tgts.extend(targets.cpu().numpy())
                total += targets.size(0)
                correct += (preds == targets).sum().item()

                # 记录 batch-wise metrics
                ba = 100 * (preds == targets).sum().item() / targets.size(0)
                bp = precision_score(targets.cpu(), preds.cpu(), average='macro', zero_division=0)
                br = recall_score(targets.cpu(), preds.cpu(), average='macro')
                bf = f1_score(targets.cpu(), preds.cpu(), average='macro')
                batch_accs.append(ba)
                batch_ps.append(bp)
                batch_rs.append(br)
                batch_fs.append(bf)

                # 记录 loss 组成部分
                bc_losses.append(self._last_bc_loss)
                penalty_losses.append(self._last_penalty_loss)

        # 整体指标
        test_loss /= len(self.val_loader)
        test_acc = 100 * correct / total
        prec = precision_score(all_tgts, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_tgts, all_preds, average='macro')
        f1 = f1_score(all_tgts, all_preds, average='macro')
        cm = confusion_matrix(all_tgts, all_preds)

        # 损失细分
        bc_loss = sum(bc_losses) / len(bc_losses)
        penalty_loss = sum(penalty_losses) / len(penalty_losses)

        # 标准差统计
        test_acc_std = np.std(batch_accs)
        prec_std = np.std(batch_ps)
        rec_std = np.std(batch_rs)
        f1_std = np.std(batch_fs)
        self.scheduler.step(test_loss)


        # 📌 行为评估（考试式）
        behavior_result = None
        examiner = ExamManager(self.model, device=self.device, render=False)
        behavior_result = examiner.run_exam(n_episodes=3)
        print(f" Exam: distance={behavior_result['mean_distance']:.1f}, "
              f"exp={behavior_result['exp']:.1f}, pass={behavior_result['pass_rate']:.2f}")



        # 汇总日志
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
            'Confusion Matrix': cm,
        }

        test_log.update({
            "Mean Distance": behavior_result["mean_distance"],
            "Pass Rate": behavior_result["pass_rate"],
            "Exp": behavior_result["exp"]
        })

        # 🌟 按“行为指标”保存模型：仅当 mean_distance 更优时保存
        save_model = False
        if behavior_result:
            save_model = behavior_result["mean_distance"] > best_mean_distance

        if save_model:
            best_mean_distance = behavior_result["mean_distance"]
            best_test_logs = test_log
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"Best model saved → {self.best_model_path}")

        return test_log, best_test_logs, best_mean_distance
