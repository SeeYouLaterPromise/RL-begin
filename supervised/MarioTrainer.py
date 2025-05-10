from utils.ClassifyTrainBase import TrainBase

class MarioTrainer(TrainBase):
    def train_step(self, batch):
        imgs, targets = batch
        imgs, targets = imgs.to(self.device), targets.to(self.device)
        outputs = self.model(imgs)                   # (batch_size, num_actions)
        loss = self.criterion(outputs, targets)     # CrossEntropyLoss
        preds = outputs.argmax(dim=1)               # 分类预测
        return loss, preds, targets

    def val_step(self, batch):
        imgs, targets = batch
        imgs, targets = imgs.to(self.device), targets.to(self.device)
        outputs = self.model(imgs)
        loss = self.criterion(outputs, targets)
        preds = outputs.argmax(dim=1)
        return loss, preds, targets
