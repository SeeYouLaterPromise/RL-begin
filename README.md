# Begin to learn Reinforcement Learning for Super Mario
Attention: when executing `pip install -r requirements.txt`, you should quit the VPN.

## GAME Preliminary
### 🎮 game handle buttons controlling
在马里奥 NES 游戏（也就是任天堂红白机上的经典《Super Mario Bros.》）中，**`A` 和 `B` 是游戏手柄上的两个主要动作按钮**。它们对应的功能如下：

| 按钮  | 作用              | 在马里奥中的实际效果               |
|-----|-----------------|--------------------------|
| `A` | **跳跃（Jump）**    | 控制马里奥跳起来                 |
| `B` | **加速（Run）或射火球** | 跑步加速、游泳时加速、发射火球（如果有火球能力） |

---

### Keyboard buttons controlling
`human_op.py` can provide you with mario playing.
> 🎮 控制说明：D=右, A=左, K=跳, J=加速, W=上, S=下，支持组合键，如 D+K


## Task Allocation

### work zone
协作过程中不要去修改其他人负责的工作区！
- `Supervised` (Yonghai Yue)
- `Unsupervised` (Weiwei Lin)
- `Semi-supervised` (Yexin Liu Lu)


## Supervised Learning
input: game frame

output: action_id

### Data Collection
`collect_data.py`: Designed for collecting data.

玩家使用须知：
- 你可以自己**设定关卡**进行收集训练数据。
- 每当**通关**或**死亡**会结束数据采集。

采集数据设置：
- grayscale
- resize: `RESIZE_SHAPE = (84, 84)`
- frame skipping / subsampling: `FRAME_SKIP = 4 `
- recording after first moving: 检查是否开始采集: 玩家开始移动，开始采集数据


数据集文件夹结构：
```
supervised/
└── mario_data/ --------------- Total
    ├── 1-1/  ----------------- Level (关卡)
    │   ├── 10-12-37/ --------- Exp   (实验)
    │   │   ├── frames/ ------- Image
    │   │   └── trajectory.json
    │   ├── 11-09-05/
    │   │   ├── frames/
    │   │   └── trajectory.json
    ├── 1-2/
    │   └── 10-13-00/
    │       ├── frames/
    │       └── trajectory.json
```

`MarioDataset.py` defines the `MarioDataset` class and the `train_val_split` function, both of which are used for data preparation prior to training.

### Training
`MarioTrainer.py` implements a subclass of `ClassifyTrainBase` with built-in functionalities for logging and plotting during training.


## Unsupervised


## Semi-supervised