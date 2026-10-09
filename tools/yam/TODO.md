# YAM 部署 — 上真机 TODO 清单

按顺序做。环境已在 `lowcost` 机器上配好(venv、gello、i2rt、teleop 都能跑),
所以重点是「确认权威细节 + 补代码 + 联调」。

## A. 环境补充(工控机,已完成大半)

- [ ] A1. 激活环境并装 client 额外依赖(teleop 能跑说明 gello/i2rt 已就绪):
      ```bash
      cd /home/xiangyushun/workspace/YAM/gello_software && source .venv/bin/activate
      uv pip install websockets msgpack numpy pyyaml
      ```
- [ ] A2. 验证 import:
      ```bash
      python -c "from gello.robots.yam import YAMRobot; print('ok')"
      ```

## B. 读实验室自己的脚本(权威来源,先做这个)

> 实验室在公开 gello_software 之上加了 `scripts/teleop` 和 `scripts/safety`,
> 这两份**不在公开仓库里**,才是这台机器控制方式的真源。上机后先读它们,再填代码。

- [ ] B1. 读 `scripts/teleop`,确认它怎么实例化 `YAMRobot`:
      - **channel 名**到底传的是 `can_follower_l`/`can_follower_r` 还是别的
        (PDF 里 `ip link show can_follower_l` 只是确认物理接口存在)。
      - 它**读状态**时用 `get_joint_state()` 的原始值,还是乘了 `joint_signs`。
      - 它**下发给从臂**时 gripper 用的什么单位/缩放。
- [ ] B2. 读 `scripts/safety`,记下 `status/hold/gravity/off` 的调用方式(部署停止时用)。
- [ ] B3. 确认相机:PDF 没提相机。看 teleop / 采集流程里相机是怎么起的(ZMQ 端口、
      几个相机、分辨率、RGB/BGR)。G0.5 需要 `head_rgb`/`left_wrist_rgb`/`right_wrist_rgb`
      三路 [3,480,640]。

## C. 硬件 / 通信确认(真机)

- [ ] C1. **CAN 通道名**:以 B1 读到的为准,填进 `config.yaml` 的
      `robot.left_channel` / `right_channel`(默认已写成 `can_follower_l/r`)。
- [ ] C2. **GPU 机连通性**:填 `config.yaml` 的 `server.host`(GPU 机 IP)和 `server.port`,
      确认工控机能 `ping` 通、端口没被防火墙挡(`nc -zv host port`)。

## D. 代码补全(policy_client.py / config.yaml)

- [ ] D1. **图像颜色顺序** — `policy_client.py: read_images()`
      目前只 `img.transpose(2,0,1)`。若相机输出是 BGR,改成:
      ```python
      img = img[:, :, ::-1]   # BGR -> RGB, if needed
      images[key] = np.ascontiguousarray(img.transpose(2, 0, 1))
      ```
- [ ] D2. **gripper 缩放** — `policy_client.py: command()`
      G0.5 输出归一化 gripper;按 B1 读到的 i2rt gripper 单位,在
      `left_cmd[6]` / `right_cmd[6]` 处做映射。
- [ ] D3. **关节符号 / 偏移(关键)** — `policy_client.py: read_state()` / `command()`
      以 B1 读到的 teleop 实现为准。`YAMRobot.get_joint_state()` 本身返回 i2rt 原始关节、
      不乘 sign;`joint_signs=[1,-1,-1,-1,1,1]` 只用于给主臂(Dynamixel)算 offset。
      若 teleop 里对从臂读数/下发做了 sign 或 offset 处理,我们这里要照着做一遍,
      否则机械臂会跳变。
- [ ] D4. **config.yaml** 填真实值:host/port、CAN 通道名、相机端口。

## E. GPU 机服务端

- [ ] E1. 下载权重:`g05-base`(约 11GB)+ `qwen3_5_2b_base_processor`(约 22MB)。
- [ ] E2. 用 `configs/data/yam.yaml` 启动 `serve_policy.py`,确认 `--port` 与
      `config.yaml` 的 `server.port` 一致。

## F. 联调验证(用 tests/ 里的 5 个程序,由静到动,避免机械臂跳变)

- [ ] F1. `python tests/test_env.py` — CAN 连接 + 读状态,确认 7 维、有限、形状对。
- [ ] F2. `python tests/test_cameras.py` — 读帧存 PNG,肉眼看颜色(RGB/BGR)和画面。
- [ ] F3. `python tests/test_transport.py` — dummy obs 过服务端,确认不报
      shape / 键名错误(这层不动机械臂、不需要相机)。
- [ ] F4. `python tests/test_predict.py` — 真 obs → action,看「当前关节 vs 预测
      动作」的 delta。**max |delta| 过大(>0.5 rad)说明符号/偏移有问题,先别进 live。**
- [ ] F5. `python tests/test_live.py` — 确认 F4 的 delta 合理后再真正下发。

## G. 安全停止(每次都记得)

- [ ] G1. 策略跑完 / Ctrl+C 打断后,执行:
      ```bash
      scripts/safety hold       # 两臂保持受控支撑(默认最安全)
      # scripts/safety gravity  # 需要重力补偿时
      # scripts/safety off      # 关电机前必须用手托住两臂
      ```
