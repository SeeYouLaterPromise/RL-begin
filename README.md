# Begin to learn Reinforcement Learning for Super Mario

Attention: when executing `pip install -r requirements.txt`, you should quit the VPN.

## Preliminary

### Environment Configuration

Thanks for "https://www.bilibili.com/video/BV1CERYY3EjA?spm_id_from=333.788.videopod.sections&vd_source=be632c59a4ce49cc99bcd97058a50691&p=11"'s help:

```bash
conda create -n rl_mario python=3.8
conda activate rl_mario

pip install setuptools==65.5.0
pip install wheel==0.38.4
python.exe -m pip install pip==20.2.4

pip install -r requirement.txt
```

Pay attention: `gym==0.21.0` (must), if you follow the video, get the updatest version of `gym`, you will meet error like:

```
Traceback (most recent call last):
  File ".\human_op.py", line 49, in <module>
    state, _, done, _ = env.step(current_action)
  File "D:\Anaconda\envs\rl\lib\site-packages\nes_py\wrappers\joypad_space.py", line 74, in step
    return self.env.step(self._action_map[action])
  File "D:\Anaconda\envs\rl\lib\site-packages\gym\wrappers\time_limit.py", line 50, in step
    observation, reward, terminated, truncated, info = self.env.step(action)
ValueError: not enough values to unpack (expected 5, got 4)
```

### gym

上面的代码是我写来采集玩家行为数据的：只有一次机会，无论是否通关都会结束数据采集。但是，我想根据玩家是否通关给 trajectory.json 命名为 trajectory_success.json 或 trajectory_failure.json。我原本想通过 info.['life']来实现，但是测试发现不奏效，请你帮我想想办法。下面是 info 的内容结构：

```
reward: 0.0,
done: False,
info: {
  'coins': 0,
  'flag_get': False,
  'life': 2,
  'score': 0,
  'stage': 1,
  'status': 'small',
  'time': 341, 'world': 1, 'x_pos': 40, 'x_pos_screen': 40, 'y_pos': 79}
```

| 键                 | 含义                              |
| ------------------ | --------------------------------- |
| `info['flag_get']` | ✅ 玩家是否成功跳到旗杆（即通关） |
| `info['life']`     | 仅当死亡时偶尔更新，但有延迟      |
| `info['status']`   | `small`, `big`, `fireball` 状态   |
| `done`             | 游戏是否结束（可能是死亡/通关）   |

### 🎮 game handle buttons controlling

在马里奥 NES 游戏（也就是任天堂红白机上的经典《Super Mario Bros.》）中，**`A` 和 `B` 是游戏手柄上的两个主要动作按钮**。它们对应的功能如下：

| 按钮 | 作用                    | 在马里奥中的实际效果                             |
| ---- | ----------------------- | ------------------------------------------------ |
| `A`  | **跳跃（Jump）**        | 控制马里奥跳起来                                 |
| `B`  | **加速（Run）或射火球** | 跑步加速、游泳时加速、发射火球（如果有火球能力） |

---

### Keyboard buttons controlling

`human_op.py` can provide you with mario playing.

> 🎮 控制说明：D=右, A=左, K=跳, J=加速, W=上, S=下，支持组合键，如 D+K

## Task Allocation

### work zone

- `Supervised` (Yonghai Yue)
- `Unsupervised` (Weiwei Lin)
- `Semi-supervised` (Yexin Liu Lu)

## Supervised Learning

input: game frame

output: action_id

### Data Collection

`collect_data.py`: Designed for collecting data.

采集使用须知：

- 你可以自己**设定关卡**进行收集训练数据。
- 每当**通关**或**死亡**会结束数据采集。
- 通关后`trajectory.json`会被标注为 success。

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
    │   │   └── trajectory_success.json ---------- (通关结束)
    │   ├── 11-09-05/
    │   │   ├── frames/
    │   │   └── trajectory_failure.json ---------- (死亡结束/手动退出)
    ├── 1-2/
    │   └── 10-13-00/
    │       ├── frames/
    │       └── trajectory_success.json
```

`MarioDataset.py` defines the `MarioDataset` class and the `train_val_split` function, both of which are used for data preparation prior to training.

我现在已经实现了下面的数据集文件夹结构，将是否存在死亡帧的实验记录标注处理（通过命名 json 文件）trajectory_failure.json 中最后一条数据的 is_dead 为 true，即死亡帧。我现在在思考：死亡帧是不是由于在死亡帧发生的前一段时间玩家的错误判断导致的，所以这些帧是不是也要让 agent 作为反面教材学习？考虑到人的反应时间（假设 1s），跳帧（4 帧）和帧率（FPS=60），所以死亡帧前 15 帧是否需要让 agent 作为反面教材学习？

### Training

`MarioTrainer.py` implements a subclass of `ClassifyTrainBase` with built-in functionalities for logging and plotting during training.

## Unsupervised

## Semi-supervised

```

```
