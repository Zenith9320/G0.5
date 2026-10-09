# G0.5 YAM 复现 —— ActionCodec + Backbone 微调

本仓库是「在 YAM 双臂机器人上复现 G0.5」的**补丁集**，不是完整代码。
它只包含 4 个新文件，需要拷贝进官方 [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) 仓库后运行，其余全部复用官方的 `src/g05/` 与 `configs/`。

| 本仓库文件 | 落到官方仓库的位置 | 作用 |
|---|---|---|
| `configs/task/yam.yaml` | `configs/task/yam.yaml` | 任务配置（27 维、训练超参、codec 路径） |
| `configs/data/yam.yaml` | `configs/data/yam.yaml` | 数据配置（shape_meta、变换、数据集目录） |
| `scripts/train_tokenizer.py` | `scripts/train_tokenizer.py` | **ActionCodec 微调脚本** |
| `tools/yam/mcap_to_lerobot.py` | `tools/yam/mcap_to_lerobot.py` | MCAP → LeRobot 数据转换 |
| `tools/yam/policy_client.py` | `tools/yam/policy_client.py` | 真机部署 client（微调前后共用） |
| `requirements.txt` | （不拷贝，本地装环境用） | Python 依赖清单 |

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
cp G0.5/tools/yam/policy_client.py GalaxeaVLA/tools/yam/policy_client.py
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

## Backbone 微调是干什么的

ActionCodec 只负责把动作窗口编成离散 token；backbone（Qwen3.5 2B + 视觉编码器）才是真正
「看图 → 生成 CoT + 动作 token」的 VLA 主体。「Backbone 微调」= 用官方 g05-base 权重继续
监督训练，让它学会 YAM 的「观测 → 动作」映射。此时 ActionCodec **冻结**，只当离散 tokenizer 用。

## 前置条件（Backbone）

1. backbone 权重（约 11GB）：
   ```bash
   huggingface-cli download OpenGalaxea/G05 g05-base --local-dir checkpoints
   ```
   确认 `checkpoints/g05-base/checkpoints/model_state_dict.pt` 就位。
2. processor 权重（约 22MB）：
   ```bash
   huggingface-cli download OpenGalaxea/G05 qwen3_5_2b_base_processor --local-dir checkpoints
   ```
3. `ffmpeg`（训练抽视频帧用）。
4. YAM 数据已转换（`data/yam/TASK-YAM-XXXX_lerobot/`）且 `data/stats/yam_stats.json` 已生成。
5. ActionCodec 微调已完成（`checkpoints/action_tokenizer_yam.pt`）。

## 步骤（Backbone）

以下命令在官方仓库根目录执行，走**官方脚本** `scripts/run/finetune.sh`（不是 `train_tokenizer.py`）。

### 1. 设置输出目录

```bash
export G05_OUTPUT_DIR=/path/to/output
```

### 2. 干跑（验证配置组装、不 OOM）

```bash
bash scripts/run/finetune.sh 1 yam --dry-run --max_datasets 1
```

`1` = GPU 数，`yam` = 自动解析成 `configs/task/yam.yaml`。

### 3. 冒烟测试（跑几步确认能反向传播）

```bash
bash scripts/run/finetune.sh 1 yam --test --max_datasets 1 model.max_steps=10
```

### 4. 正式训练

```bash
bash scripts/run/finetune.sh 8 yam
```

## Backbone 命令速查

| 用途 | 命令 |
|---|---|
| 干跑 | `bash scripts/run/finetune.sh 1 yam --dry-run --max_datasets 1` |
| 冒烟测试 | `bash scripts/run/finetune.sh 1 yam --test --max_datasets 1 model.max_steps=10` |
| 正式训练 | `bash scripts/run/finetune.sh 8 yam` |

## 关键点（Backbone）

- **配置已就绪，无需再改代码**：`configs/task/yam.yaml` 里 `action_dim=27`、`learning_rate=4e-5`、
  `batch_size=8`、`max_epochs=10`、`vision_lr_multiplier=0.1`、`backbone_lr_multiplier=1.0`、
  `tokenizer.vq_config.ckpt_dir: checkpoints/action_tokenizer_yam.pt` 均已写好。
- **显存**：官方标注 >70GB，需 A100 80GB / H20 96GB 这一档；`batch_size=8` + 多卡时先用
  `--max_datasets 1` 探清单卡容量。
- **顺序**：codec 微调 → backbone 微调；两者共用 `data/stats/yam_stats.json`，backbone 会冻结 codec 权重。

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
- **动作是绝对还是相对**：`mcap_to_lerobot.py` 存的是**绝对**关节角；`RelativeJointTransform` 在训练时转成**相对增量**。部署时的逆变换（反归一化 + cumsum）由 **server 端 `processor.postprocess` 自动完成**，client 拿到的直接就是绝对关节角，**无需手动逆变换**。

---

## 真机部署 + 微调前后对比（A/B）

### 架构

```
YAM ROS 节点(policy_client.py，要写) ──WebSocket/msgpack──► serve_policy.py(官方，复用)
  读观测 → 拼 raw_obs → infer → 下发动作     返回已还原的绝对关节角   加载权重 → 推理 → postprocess
```

server 端官方 `serve_policy.py` 已完成「反归一化 + 反 RelativeJointTransform（cumsum）」，
client 拿到的 `action` 直接是**绝对关节角**，直接下发即可（见「常见问题」最后一条）。

### 微调前 baseline（现在就能做）

1. 下载 backbone + processor 权重：
   ```bash
   huggingface-cli download OpenGalaxea/G05 g05-base --local-dir checkpoints
   huggingface-cli download OpenGalaxea/G05 qwen3_5_2b_base_processor --local-dir checkpoints
   ```
2. GPU 上起 server（官方权重）：
   ```bash
   python scripts/serve_policy.py \
       --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt \
       --port 8765 eval_embodiment=galaxea_r1lite
   ```
3. 机器人侧跑 client：
   ```bash
   python tools/yam/policy_client.py --host <gpu-host> --port 8765 --task "把杯子放到杯垫上"
   ```

### 微调后

只换 server 的 `--ckpt_path`，其余不变：

```bash
python scripts/serve_policy.py \
    --ckpt_path $G05_OUTPUT_DIR/checkpoints/step_XXX/model_state_dict.pt \
    --port 8765 eval_embodiment=galaxea_r1lite
```

client 命令两次**完全相同**（`policy_client.py` 微调前后共用，写一次即可）。

### 端口

WebSocket 监听端口，server `--port` 与 client `--port` 一致即可，默认 **8765**（官方约定）。
跨机器需防火墙放行该端口；查占用：`ss -tlnp | grep 8765`。

### A/B 对比评估协议

要公平对比，遵循：

1. **同一任务 + 同一初始位姿**，前后各跑 N 次（如各 10 次）。
2. **录视频**（相机流或外部相机），分别存 `eval/before/` 与 `eval/after/`。
3. **记成功率**：每次成功/失败打钩，算 `成功次数 / N`。
4. **记动作质量**（可选）：记录下发动作序列，比对方差/抖动。

| 指标 | 微调前（官方权重） | 微调后 |
|---|---|---|
| 成功率 | | |
| 动作是否平滑 | | |
| 完成任务次数 / N | | |

### 注意

- 官方权重是 R1Lite 数据训的（7 轴臂 + ee_pose），YAM 是 6 轴臂，**微调前动作是乱的**，
  仅作管道验证 + 基线对照，别拿它做真实操作。
- `policy_client.py` 里 `CONFIG` 的 topic 名是**占位符**，需先跟 YAM SDK 核对改成真实名字。
- 安全：baseline 阶段动作可能大幅乱动，先小幅度/低速测试，或先不接夹爪。
