很好！你提到的这三种行为克隆中用于“惩罚死亡帧”的 **penalty loss 设计**：

- **uniform KL loss**
- **负 log prob**
- **max prob loss**

它们的目标是一致的：

> ✴️ 在关键帧（如死亡前）**让模型的输出更“模糊”或“不自信”**，即避免过度确定性选择一个动作。

但它们在**数学形式、优化目标、训练效果**方面有所不同。

---

## ✅ 1. Max-Prob Penalty Loss（最大概率惩罚）

### 📌 公式：

$$
\mathcal{L}_{\text{penalty}} = \mathbb{E}_{x \in \mathcal{D}_{\text{critical}}} \left[ \max(\text{softmax}(f(x))) \right]
$$

### 🔧 代码：

```python
probs = torch.softmax(logits, dim=1)
max_probs, _ = probs.max(dim=1)
loss = max_probs.mean()
```

### ✅ 优点：

- 实现简单；
- 有效惩罚高置信度预测（让模型保持“犹豫”）；
- 通常在 early-stage 的探索策略中表现好。

### ⚠️ 局限：

- 没有明确指定“理想分布”；
- 不提供方向性：仅要求“别太确定”，但不告诉模型“更像什么”。

---

## ✅ 2. 负 log prob 惩罚（负对数最大概率）

### 📌 公式：

$$
\mathcal{L}_{\text{penalty}} = -\log\left(1 - \max(\text{softmax}(f(x))) + \epsilon \right)
$$

（其中 ε 是防止 log(0) 的极小值，如 1e-6）

### 🔧 代码：

```python
probs = torch.softmax(logits, dim=1)
max_probs, _ = probs.max(dim=1)
loss = -torch.log(1.0 - max_probs + 1e-6).mean()
```

### ✅ 优点：

- 惩罚越强：**max prob 越接近 1，惩罚越大**；
- 提供了“激进压制”，更明显地区分“模糊 vs 自信”状态。

### ⚠️ 局限：

- 梯度可能爆炸或消失（尤其当 max prob → 1）；
- 训练初期不稳定，通常适合 fine-tuning。

---

## ✅ 3. KL 散度到 Uniform 分布（推荐结构性方式）

### 📌 目标：

强制模型的输出概率接近均匀分布：

$$
\mathcal{L}_{\text{penalty}} = \text{KL}(p_{\text{model}} \parallel \text{Uniform})
= \sum_i p_i \log\left( \frac{p_i}{1/K} \right)
= \sum_i p_i \log(K \cdot p_i)
$$

其中 $K$ 是类别数。

### 🔧 代码：

```python
probs = torch.softmax(logits, dim=1)
num_classes = probs.size(1)
uniform = torch.full_like(probs, fill_value=1.0/num_classes)
loss = torch.nn.functional.kl_div(probs.log(), uniform, reduction="batchmean")
```

### ✅ 优点：

- 提供结构化目标：“向均匀分布靠近”；
- 不仅压制最大值，还促使整个输出分布平滑；
- 与 many-shot setting 下的表示学习思想一致。

### ⚠️ 局限：

- 要求 logits 非常稳定（否则 KL 不稳定）；
- 对于类别极不平衡时，KL ≠ 实际风险最小化。

---

## ✅ 核心对比总结

| 方法              | 惩罚机制                | 表现   | 优点                   | 局限               |
| ----------------- | ----------------------- | ------ | ---------------------- | ------------------ |
| **Max Prob**      | 惩罚最大 softmax 输出   | 温和   | 实现简单、鲁棒         | 无明确参考分布     |
| **-log(1 - max)** | 惩罚“接近 1”的 max prob | 激进   | 精确控制不确定性       | 梯度不稳定、收敛慢 |
| **KL → Uniform**  | 惩罚整分布偏离均匀      | 结构化 | 适合对比或模糊输出学习 | 训练早期易不稳定   |

---

## ✅ 应用建议：

| 训练阶段 | 推荐方案                    |
| -------- | --------------------------- |
| 初期探索 | Max Prob（稳定温和）        |
| 强化约束 | KL Uniform（可结构指导）    |
| 后期收敛 | -log(1 - max)（压制高置信） |

---

是否需要我帮你把这三种 penalty loss 都封装成一个模块，并允许你用参数 `penalty_type='max'/'log'/'kl'` 切换使用？
