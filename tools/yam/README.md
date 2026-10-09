# G0.5 → YAM 部署(工控机客户端)

把 G0.5(单流自回归 VLA,Qwen3.5-2B + ActionCodec)部署到 I2RT YAM 双臂机器人上。
推理在 **GPU 机**跑,工控机只负责「读状态 → 发 obs → 收 action → 控臂」。

## 文件结构

```
g05_yam_deploy/
├── config.yaml        # 部署参数(server/CAN/相机/策略),上真机前要填真实值
├── ws_transport.py    # 独立 WebSocket 传输层,只依赖 msgpack/numpy/websockets
├── policy_client.py   # 控制主循环(YAMRobot 读状态 + 发 obs + 控臂)
├── README.md          # 本文件(数据格式契约)
├── TODO.md            # 上真机要做的清单 + 要补全的代码
└── tests/             # 5 个冒烟测试 + 共享 helper(见下)
    ├── _helpers.py
    ├── test_env.py
    ├── test_cameras.py
    ├── test_transport.py
    ├── test_predict.py
    └── test_live.py
```

## 数据流

```
YAMRobot.get_joint_state()  ──┐
ZMQClientCamera.read()       ──┤→ build_raw_obs() → WebSocket → serve_policy.py (GPU)
                               │                                    │
YAMRobot.command_joint_state() ←─ command(action) ←── resp.action ←─┘
```

## 实验室环境(真机,来自「YAM 使用指北」)

- 机器名 `lowcost`,代码在 `/home/xiangyushun/workspace/YAM/gello_software`,venv 在
  `.venv`。
- **CAN 接口名是 `can_follower_l` / `can_follower_r`**(`ip -details link show
  can_follower_l`),不是公开 gello 配置里的 `can_left`/`can_right`。
- 实验室自己加了 `scripts/teleop`(采集)和 `scripts/safety`(status/hold/gravity/off)
  两个脚本——它们不在公开 wuphilipp/gello_software 仓库里。**真机上这两份脚本是权威**,
  上机后先读一遍,确认 channel 名、关节约定、gripper 缩放、相机启动方式。
- GELLO 主臂走 `/dev/serial/by-id/` 的 FTDI 串口(LEFT: FTC2FNQP, RIGHT: FTBSX7UI);
  我们部署策略时**不经过主臂**,直接用从臂(策略输出直接控从臂),所以只需 CAN + 相机。

## 数据格式契约(与服务端 shape_meta 对照)

服务端 `serve_policy.py` 的 `_validate_obs()` 拿 **`embodiment_type` 对应的
processor 的 `shape_meta`** 来校验。YAM 部署用 `embodiment_type="galaxea_r1lite"`,
processor 是 `configs/data/yam.yaml`,其 `shape_meta` 为:

| 类型 | key | raw_shape | dtype |
|---|---|---|---|
| state | `left_arm` | 6 | float32 |
| state | `left_gripper` | 1 | float32 |
| state | `right_arm` | 6 | float32 |
| state | `right_gripper` | 1 | float32 |
| image | `head_rgb` | [3, 480, 640] | uint8 |
| image | `left_wrist_rgb` | [3, 480, 640] | uint8 |
| image | `right_wrist_rgb` | [3, 480, 640] | uint8 |

我们的 `policy_client.py` 发送的 `raw_obs`:

```python
{
    "task":            str,        # 自然语言指令
    "embodiment_type": "galaxea_r1lite",
    "frequency":       30,         # 控制频率(与服务端一致)
    "images": {                    # [C,H,W] uint8
        "head_rgb":         ...,   # [3, 480, 640]
        "left_wrist_rgb":  ...,
        "right_wrist_rgb": ...,
    },
    "state": {                     # [D] float32
        "left_arm":      [6],
        "left_gripper":  [1],
        "right_arm":     [6],
        "right_gripper": [1],
    },
}
```

### 与官方 r1lite experiments 的对照

| 项目 | 官方 r1lite 客户端 | 我们的 policy_client.py | 结论 |
|---|---|---|---|
| 顶层键 | images / state / task / embodiment_type | + frequency(无害) | ✅ |
| state 结构 | `{name: [D] float32}` | 同 | ✅ |
| state 的 key | 8 个(含 torso/chassis/ee_pose) | 4 个(arm/gripper) | ⚠️ YAM 没有躯干/底盘,用 yam.yaml 的 4 key,一致 |
| image 结构 | `[C,H,W] uint8`(BGR→RGB 后 transpose) | `transpose(2,0,1)` | ✅ 形状一致,颜色顺序待确认 |
| image 的 key | head_rgb / left_wrist_rgb / right_wrist_rgb | 同 | ✅ |

**结论:以 `yam.yaml` 启动 serve_policy 时,我们的数据格式会 100% 通过 `_validate_obs`。**

官方 r1lite 之所以有 8 个 state key,是因为它是全尺寸人形(有 torso 躯干、chassis
底盘、left/right_ee_pose 末端位姿);YAM 是双臂+双爪,`yam.yaml` 只定义这 4 个 key。

## 运行

工控机(在 gello_software 的 venv 里):

```bash
cd /home/xiangyushun/workspace/YAM/gello_software && source .venv/bin/activate
uv pip install websockets msgpack numpy pyyaml
python ../g05_yam_deploy/policy_client.py --config ../g05_yam_deploy/config.yaml
```

GPU 机:

```bash
# 用 yam.yaml 启动 serve_policy.py,--port 与 config.yaml 的 server.port 一致
python scripts/serve_policy.py ...  # 见 GalaxeaVLA 官方启动参数
```

## 冒烟测试(分层,由静到动)

`tests/` 里 5 个独立程序,前 4 个都不动机械臂,只有 `test_live.py` 才下发命令。
共享逻辑在 `tests/_helpers.py`。每个都可单独跑(从仓库根目录):

```bash
python tests/test_env.py         # import + CAN 连接 + 读状态(只读)
python tests/test_cameras.py     # 每个相机读一帧并存 PNG(只读,看颜色)
python tests/test_transport.py   # WS 连服务端 + dummy obs 往返(不动机械臂、不用相机)
python tests/test_predict.py     # 真 obs -> action,只打印不控制
python tests/test_live.py        # 真正下发(危险,先确认 predict 的 delta 合理)
```

`test_transport` 用全零 dummy obs 就能验证 wire 格式和服务端 `_validate_obs`;
`test_predict` 会把「当前关节 / 预测动作 / 差值」都打出来,跑 `test_live` 前先看
max |delta| 是否合理。

## 关键约定(上真机前必须核对)

1. **关节符号**:`joint_signs=[1,-1,-1,-1,1,1]` 只作用于 GELLO 主臂(Dynamixel 算
   offset 用);`YAMRobot.get_joint_state()` 返回的是 i2rt **原始关节**。要确认 G0.5
   训练数据记录的是不是同一份原始关节。
2. **图像颜色顺序**:官方做 `BGR2RGB`;我们的 `read_images()` 目前只 transpose 没转色。
3. **gripper 缩放**:G0.5 输出归一化 gripper,YAMRobot 第 7 维可能是 i2rt 原单位。
4. **安全停止**:策略跑完/打断后,用实验室的 `scripts/safety hold` 让两臂保持受控支撑;
   需要时 `scripts/safety gravity`(重力补偿)或 `scripts/safety off`(关电机,关之前
   必须托住两臂)。

详见 `TODO.md`。
