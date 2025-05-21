from torch.utils.data import DataLoader
from MarioDataset import MarioDataset, train_valid_split
from supervised.model.CNN2D import MarioBCModel
from MarioTrainer import MarioTrainer
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_game import COMPLEX_MOVEMENT, SUPERVISED_DATA_DIR, RESULT_SAVE_DIR
import time
from utils.metric_plot import plot_loss



# 参数设置
experiment_time = time.strftime("%d-%H-%M", time.localtime())
BASE_DIR = SUPERVISED_DATA_DIR
SAVE_DIR = os.path.join(RESULT_SAVE_DIR, experiment_time)
NUM_ACTIONS = len(COMPLEX_MOVEMENT)
BATCH_SIZE = 32
NUM_EPOCHS = 100
VAL_RATIO = 0.2
LR = 1e-4
PATIENCE = 2
FRAME_STACK = 8  #帧堆叠数

# Pure BC or BC with Penalty
PENALTY_MODE = True
PENALTY_WEIGHT = 0.6
PENALTY_TYPE = "maxprob"
PENALTY_N = 15

if __name__ == "__main__":
    # 加载数据
    # 创建带帧堆叠的数据集
    dataset = MarioDataset(
        base_dir=SUPERVISED_DATA_DIR,
        penalty_mode = PENALTY_MODE,
        penalty_N = PENALTY_N,
        frame_stack=FRAME_STACK  # 堆叠帧数
    )
    train_set, val_set = train_valid_split(dataset, val_ratio=VAL_RATIO)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = MarioBCModel(NUM_ACTIONS,frame_stack=FRAME_STACK)
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
        "PATIENCE": PATIENCE,
        "PENALTY_MODE": PENALTY_MODE,
        "PENALTY_WEIGHT": PENALTY_WEIGHT,
        "PENALTY_TYPE": PENALTY_TYPE,
        "PENALTY_N": PENALTY_N,
        "FRAME_STACK": FRAME_STACK,
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
        penalty_mode=PENALTY_MODE,
        penalty_weight=PENALTY_WEIGHT,
        penalty_type=PENALTY_TYPE,
        draw=True,
        training_config=training_config,
    )

    train_df, val_df = trainer.run(epochs=NUM_EPOCHS, name="all")
    plot_loss(train_df, val_df, SAVE_DIR, train_plot_ls=['Train BC Loss', 'Train Penalty Loss'], test_plot_ls=['Test BC Loss', 'Test Penalty Loss'], name="additional")
