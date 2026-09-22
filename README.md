# G0.5 YAM 复现 —— ActionCodec 微调

本仓库是「在 YAM 双臂机器人上复现 G0.5」的**补丁集**，不是完整代码。
它只包含 4 个新文件，需要拷贝进官方 [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) 仓库后运行，其余全部复用官方的 `src/g05/` 与 `configs/`。

| 本仓库文件 | 落到官方仓库的位置 | 作用 |
|---|---|---|
| `configs/task/yam.yaml` | `configs/task/yam.yaml` | 任务配置（27 维、训练超参、codec 路径） |
| `configs/data/yam.yaml` | `configs/data/yam.yaml` | 数据配置（shape_meta、变换、数据集目录） |
| `scripts/train_tokenizer.py` | `scripts/train_tokenizer.py` | **ActionCodec 微调脚本（本文档主题）** |
| `tools/yam/mcap_to_lerobot.py` | `tools/yam/mcap_to_lerobot.py` | MCAP → LeRobot 数据转换 |

---

## ActionCodec 微调是干什么的

官方 ActionCodec 是一个 RVQ-VAE，把 32 帧动作窗口编码成离散 token。
「微调」= 用 YAM 自己的动作数据，**继续自监督训练官方权重**，让码本适配 YAM 的动作分布。
它不需要标签，训练信号就是「重建误差」（reconstruction MSE + commitment loss）。

## 数据流（喂给 codec 之前）

```
MCAP 录包
  └─ mcap_to_lerobot.py → LeRobot 数据集（14 维绝对关节角：6+1, 6+1）
        └─ RelativeJointTransform → 相对增量
              └─ GroupedPaddingMerger → 补零合并成 27 维
                    └─ z-score 归一化（stats）→ 切窗 horizon=32
                          └─ (B, 32, 27)  ← codec 输入
```

> 三个维度：`B`=batch 大小，`32`=时间窗口长度（30Hz 下约 1.07s），`27`=每帧动作维度（9+1+9+1+7）。

---

## 前置条件

1. 官方 [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) 仓库 + 依赖装好（`uv sync` 或等价方式）。
2. 官方 codec 权重（约 484MB）。
3. YAM 的原始 MCAP 录包（在采集它的机器/磁盘上，需先拷到本机）。

---

## 步骤

以下命令都在 **官方仓库根目录**（即 `GalaxeaVLA/`）下执行。

### 0. 拷贝补丁文件

```bash
cp G0.5/configs/task/yam.yaml      GalaxeaVLA/configs/task/yam.yaml
cp G0.5/configs/data/yam.yaml      GalaxeaVLA/configs/data/yam.yaml
cp G0.5/scripts/train_tokenizer.py GalaxeaVLA/scripts/train_tokenizer.py
cp G0.5/tools/yam/mcap_to_lerobot.py GalaxeaVLA/tools/yam/mcap_to_lerobot.py
```

### 1. 下载 codec 权重

只下这一个文件（无需下全量 55GB）：

```bash
huggingface-cli download OpenGalaxea/G05 action_tokenizer.pt --local-dir checkpoints
```

确认 `checkpoints/action_tokenizer.pt`（约 484MB）就位，路径与
`configs/tokenizer/actioncodec.yaml` 里的 `ckpt_dir` 一致。

### 2. 转换数据（MCAP → LeRobot）

```bash
python tools/yam/mcap_to_lerobot.py --input-root /data/YAM --output-root data/yam
```

产出 `data/yam/TASK-YAM-XXXX_lerobot/` 目录，正好对上 `configs/data/yam.yaml`
里 `dataset_dirs` 列出的路径。

### 3. 基线评估（先验证数据管线通不通）

```bash
python scripts/train_tokenizer.py --task yam --eval-only \
    --ckpt checkpoints/action_tokenizer.pt
```

这一步会：组装配置 → 加载数据 → 若 `data/stats/yam_stats.json` 不存在则**现场算出并存盘**
→ 用官方权重评估逐键重建 RMSE + 码本利用率。

> 若 `default_collate` 那条隐患（数据集返回变长字段时可能报错）存在，会在此步暴露，先修再继续。

### 4. 微调

```bash
python scripts/train_tokenizer.py --task yam \
    --init-ckpt checkpoints/action_tokenizer.pt \
    --out checkpoints/action_tokenizer_yam.pt \
    --freeze encoder --epochs 2 --micro-batch 64 --grad-accum 4 --lr 5e-5
```

产出 `checkpoints/action_tokenizer_yam.pt`。

### 5. 评估微调结果

```bash
python scripts/train_tokenizer.py --task yam --eval-only \
    --ckpt checkpoints/action_tokenizer_yam.pt
```

对比第 3 步与第 5 步的逐键 RMSE：若 YAM 与官方分布确有差异，微调后 RMSE 应下降、码本利用率更均衡。

---

## 收尾

`configs/task/yam.yaml` 已写死 `tokenizer.vq_config.ckpt_dir: checkpoints/action_tokenizer_yam.pt`，
所以微调完**不用改配置**，直接进入下一步 backbone 微调即可。

---

## 命令速查

| 用途 | 命令 |
|---|---|
| 基线评估 | `python scripts/train_tokenizer.py --task yam --eval-only --ckpt checkpoints/action_tokenizer.pt` |
| 微调 | `python scripts/train_tokenizer.py --task yam --init-ckpt checkpoints/action_tokenizer.pt --out checkpoints/action_tokenizer_yam.pt --freeze encoder --epochs 2 --micro-batch 64 --grad-accum 4 --lr 5e-5` |
| 评估微调后 | `python scripts/train_tokenizer.py --task yam --eval-only --ckpt checkpoints/action_tokenizer_yam.pt` |

## 冻结策略（`--freeze`）

| 模式 | 冻结什么 | 说明 |
|---|---|---|
| `encoder`（默认） | conv_in + encoder | 最稳，训后续所有层 |
| `encoder_cb0` | 上面 + 第 0 层主码本 | 更激进（`plan_deepseek.md` 推荐） |
| `none` | 不冻结 | 全量微调 |
| `all_nn` | 全部 | 仅 `--eval-only` 用 |

## 常见问题

- **`default_collate` 报错**：脚本用 `default_collate`（单 embodiment 固定 shape 一般没事）；若数据集返回 `instruction` 等变长字段会挂，改为官方 `collate_fn_pad_sequences` 或只取 `batch["action"]`。
- **`yam_stats.json` 哪来的**：`build_data` 里若不存在/为空会遍历全量数据现算；codec 与 backbone 训练必须用同一份 stats。
- **`num_residuals: 2`**：继承官方 r1lite，4 个码本只启用 2 层残差，每个 step 的动作 token 数 = 2 × tokens_per_key。
- **动作是绝对还是相对**：`mcap_to_lerobot.py` 存的是**绝对**关节角；`RelativeJointTransform` 在训练时转成**相对增量**。部署客户端需做逆变换（cumsum）还原绝对关节。
