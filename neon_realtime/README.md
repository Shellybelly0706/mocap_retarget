# Pupil Labs Neon Wi-Fi 实时接收器

这个命令行程序使用 Pupil Labs 官方 Real-Time Python API，从运行 Neon
Companion App 的手机经 Wi-Fi 实时接收：

- `gaze`：场景相机像素坐标系中的注视点、佩戴状态和 Unix 时间戳；
- `imu`：陀螺仪、加速度、姿态四元数和 Unix 时间戳；
- `events`：眨眼、注视和扫视事件；
- `scene`：与 gaze 时间匹配的场景视频，视频中会画出红色注视点。

## 1. 安装

需要 Python 3.10 或更高版本。建议使用独立虚拟环境：

```bash
python3 -m venv .venv-neon
source .venv-neon/bin/activate
python -m pip install --upgrade pip
python -m pip install -r neon_realtime/requirements.txt
```

在 Neon Companion App 中连接眼镜，并保持 App 在运行状态。手机和运行本程序的
电脑必须连接同一个局域网。

## 2. 运行

自动发现设备，接收 gaze 和 IMU（默认）：

```bash
python -m neon_realtime
```

默认的 IMU 模式是 `--imu-mode direct`：直接异步读取 RTSP 原始 Protobuf 包，
不经过 Simple API 的单元素最新值缓存。需要复现旧接收方式时才使用
`--imu-mode simple`。

接收所有支持的流并预览场景视频：

```bash
python -m neon_realtime \
  --streams gaze imu events scene \
  --preview
```

如果自动发现失败，在 Companion App/手机 Wi-Fi 设置中查看手机 IP 后直接连接：

```bash
python -m neon_realtime \
  --ip 192.168.1.169 \
  --streams gaze imu events scene
```

只检查实时数据而不写磁盘，30 秒后自动结束：

```bash
python -m neon_realtime --no-save --duration 30
```

按 `Ctrl-C` 可安全停止；视频预览窗口中也可按 `q` 或 `Esc` 停止。

## 3. 输出

默认写入
`/media/fiona/LinuxData/mocap_retarget/neon_realtime/neon_data/YYYYMMDD_HHMMSS/`：

```text
session.json        # 设备、连接、流、开始/结束时间及数据计数
gaze.jsonl          # 每行一条 gaze 数据
imu.jsonl           # 每行一条 IMU 数据，同时包含秒和精确纳秒时间戳
imu/
├── imu ps1.raw     # Export 兼容的定长 IMU 二进制记录（48 字节/样本）
├── imu ps1.time    # Export 兼容的 int64 纳秒时间戳
├── imu.dtype       # 二进制字段定义
├── extimu ps1.raw  # 长度前缀 + 原始 Protobuf 包
├── extimu ps1.time # 原始包对应的 int64 纳秒时间戳
├── imu.proto       # Protobuf 字段定义
└── quality.json    # 采样率、最大间隔和估计缺失样本数
events.jsonl        # 每行一条眼动事件
scene.mp4           # 带红色 gaze 标记的场景视频
scene_gaze.jsonl    # 每帧时间戳和匹配的 gaze
```

未选择的数据流不会创建对应文件。JSONL 中同时保留设备时间戳和电脑收到该包时的
`received_at_unix_ns`，方便之后与动作捕捉数据对齐。

`imu/imu ps1.raw` 的字段顺序与 Companion Export 一致：`timestamp_ns`、
`gyro_x/y/z`、`accel_x/y/z`、`quaternion_w/x/y/z`。Wi-Fi/RTSP 无法保证物理上
零丢包；每次停止后请查看 `imu/quality.json`。其中
`estimated_missing_samples: 0` 表示没有从时间戳间隔检测到缺样，但仍不等同于
手机本地记录的绝对无损保证。

## 4. 常见问题

- 自动发现依赖局域网 mDNS。访客 Wi-Fi、企业 Wi-Fi 或路由器的 AP/client
  isolation 可能阻止设备互相发现；这种情况优先使用 `--ip`。
- `events` 需要在 Companion App 中启用 **Compute fixations**。不同 App
  版本可能返回不同 gaze/eye-event 类型，本程序会完整保存官方对象的全部字段。
- gaze 的 `x/y` 是场景相机图像中的像素坐标，并非 0 到 1 的归一化坐标。
- 如果只需要结构化数据，可以不选择 `scene`；这能明显降低 Wi-Fi、CPU 和磁盘负载。
- 如果要求 IMU 尽可能接近手机本地 Export，请使用默认的 `direct` 模式、稳定的
  独立 Wi-Fi 或有线网络，并避免同时传输 `scene` 和 `eyes` 视频。
- 查看全部参数：`python -m neon_realtime --help`。

官方资料：

- [Getting Started](https://pupil-labs.github.io/pl-realtime-api/dev/getting-started/)
- [Connect to a Device](https://pupil-labs.github.io/pl-realtime-api/dev/methods/simple/connect-to-a-device/)
- [Streaming Gaze](https://pupil-labs.github.io/pl-realtime-api/dev/methods/simple/streaming/gaze/)

