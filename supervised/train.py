from torch.utils.data import DataLoader
from MarioDataset import MarioDataset, train_valid_split
from MarioBCModel import MarioBCModel
from MarioTrainer import MarioTrainer
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_game import COMPLEX_MOVEMENT, SUPERVISED_DATA_DIR, RESULT_SAVE_DIR



# 参数设置
BASE_DIR = SUPERVISED_DATA_DIR
SAVE_DIR = RESULT_SAVE_DIR
NUM_ACTIONS = len(COMPLEX_MOVEMENT)
BATCH_SIZE = 32
NUM_EPOCHS = 150
VAL_RATIO = 0.2
LR = 1e-3
PATIENCE = 2


if __name__ == "__main__":
    # 加载数据
    dataset = MarioDataset(BASE_DIR)
    train_set, val_set = train_valid_split(dataset, val_ratio=VAL_RATIO)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = MarioBCModel(NUM_ACTIONS)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE)

    # 保存参数日志
    training_config = {
        "BASE_DIR": BASE_DIR,
        "NUM_ACTIONS": NUM_ACTIONS,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "VAL_RATIO": VAL_RATIO,
        "LR": LR,
        "PATIENCE": PATIENCE
    }

    # 实例化 Trainer 并运行
    trainer = MarioTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=SAVE_DIR,
        draw=True,
        training_config=training_config,
    )

    train_logs, val_logs = trainer.run(epochs=NUM_EPOCHS, name="mario_behavior_clone")
