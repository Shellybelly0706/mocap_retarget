# Neon Companion 与 Ubuntu 时钟偏移监视器

`clock_offset_monitor.py` 使用 Pupil Labs Time Echo 协议，持续估计并记录
Neon Companion 手机时钟与 Ubuntu 系统时钟之间的偏移。Neon 录制仍由
Neon Monitor 发起并保存在手机上；本程序不接收 gaze、IMU 或视频数据。

## 时间约定

程序使用 Pupil Labs API 的偏移定义：

```text
offset = Ubuntu 时间 - Neon Companion 时间
```

因此，将之后导出的 Neon Unix 纳秒时间戳转换到 Ubuntu 时间域时：

```text
ubuntu_timestamp_ns = neon_timestamp_ns + offset_ms * 1_000_000
```

Time Echo 的时间戳分辨率是 1 ms。程序通过重复测量、选择低往返延迟样本和
持续记录偏移来降低网络抖动影响，但不能从统计结果证明或保证绝对误差始终小于
1 ms。终端中的 `stable=yes` 仅表示该批测量的重复性与网络延迟通过设定阈值。

## 安装

建议使用现有的 `pupil_labs` Conda 环境：

```bash
conda run --no-capture-output -n pupil_labs \
  python -m pip install -r src/neon/requirements.txt
```

手机与 Ubuntu 电脑必须在同一个稳定局域网中，且 Neon Companion App 保持运行。

## 建议录制流程

1. 在开始 Neon 录制前启动本程序。
2. 等终端出现至少一条 `stable=yes` 的估计。
3. 在电脑浏览器的 Neon Monitor 中开始录制。
4. 整个录制过程中保持本程序运行。
5. 在 Neon Monitor 中停止并保存录制。
6. 再按 `Ctrl-C` 停止本程序。
7. 将手机中的离线录制导出到 Ubuntu，并使用对应时刻的偏移校正时间戳。

## 运行

自动发现 Companion，并每 5 秒执行一批 100 次 Time Echo 测量：

```bash
conda run --no-capture-output -n pupil_labs PYTHONPATH=src python -m neon
```

如果自动发现失败，可指定手机 IP：

```bash
conda run --no-capture-output -n pupil_labs PYTHONPATH=src python -m neon \
  --ip 192.168.1.169
```

短时间测试：

```bash
conda run --no-capture-output -n pupil_labs PYTHONPATH=src python -m neon \
  --duration 30
```

直接运行重命名后的模块也可以：

```bash
conda run --no-capture-output -n pupil_labs \
  PYTHONPATH=src python -m neon.clock_offset_monitor --help
```

## 输出

默认输出目录：

```text
src/neon/time_sync/YYYYMMDD_HHMMSS/
├── session.json
└── clock_offsets.jsonl
```

- `session.json`：设备信息、参数、偏移方向、开始/结束时间及汇总。
- `clock_offsets.jsonl`：每批估计一行，包含测量时间、推荐偏移、RTT、稳定性
  检查、全部原始测量值，以及同一测量中点分别位于 Ubuntu 和 Companion
  时钟域的 `interpolation_anchor`。

离线处理时，不要对整段录制只使用一次固定偏移。应根据每个 Neon 样本时间附近
的两条 `interpolation_anchor.companion_unix_ns` 对偏移做线性插值，以补偿手机与
Ubuntu 时钟之间随时间变化的漂移；之后再将插值得到的偏移加到 Neon 时间戳上。

查看所有参数：

```bash
conda run --no-capture-output -n pupil_labs PYTHONPATH=src python -m neon --help
```
