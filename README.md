# Begin to learn Reinforcement Learning for Super Mario
Attention: when executing `pip install -r requirements.txt`, you should quit the VPN.



## Human control mario

使用`pynput.Listener`遇到的问题： 
若Focus聚焦在mario的游戏的窗口就监听不到键盘输入；移开，点击桌面其他地方就可以正常监听。但是感觉回传延迟很高，画面很卡顿。


当前的方案采用了 `pygame` 来监听键盘输入，并通过 `gym_super_mario_bros` 环境控制马里奥的移动。

---

## 当前运行中的窗口布局

| 窗口         | 来源                          | 功能            |
|------------|-----------------------------|---------------|
| Mario 窗口   | `env.render()`（SDL/NESPy）   | 显示游戏画面        |
| pygame 小窗口 | `pygame.display.set_mode()` | 激活事件系统，监听键盘输入 |

> ⚠️ 注意：键盘输入是发送给**当前焦点窗口**的，因此确保运行中不要手动点到别的程序窗口（否则会丢键盘焦点）！

---

## 🎮 NES 游戏手柄按钮含义
在马里奥 NES 游戏（也就是任天堂红白机上的经典《Super Mario Bros.》）中，**`A` 和 `B` 是游戏手柄上的两个主要动作按钮**。它们对应的功能如下：

| 按钮  | 作用              | 在马里奥中的实际效果               |
|-----|-----------------|--------------------------|
| `A` | **跳跃（Jump）**    | 控制马里奥跳起来                 |
| `B` | **加速（Run）或射火球** | 跑步加速、游泳时加速、发射火球（如果有火球能力） |

---