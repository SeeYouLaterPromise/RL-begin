from torch.utils.data import DataLoader
from configs.config_game import COMPLEX_MOVEMENT, SUPERVISED_DATA_DIR
from MarioDataset import MarioDataset, train_valid_split
from MarioBCModel import MarioBCModel
from MarioTrainer import MarioTrainer
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss


# 参数设置
BASE_DIR = SUPERVISED_DATA_DIR
SAVE_DIR = "result"
num_actions = len(COMPLEX_MOVEMENT)
batch_size = 32
num_epochs = 10
val_ratio = 0.2
lr = 1e-3
patience = 2

# 加载数据
dataset = MarioDataset(BASE_DIR)
train_set, val_set = train_valid_split(dataset, val_ratio=val_ratio)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 初始化模型
model = MarioBCModel(num_actions)
optimizer = Adam(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=patience)

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
)

train_logs, val_logs = trainer.run(epochs=num_epochs, name="mario_behavior_clone")
