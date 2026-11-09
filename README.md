# DexJoCo H200 Training

DexJoCo 六个任务分别训练，每次单机使用 **4 张 H200**：PickBucket、Microwave、ClickMouse、FoldGlasses、Hanoi、UnlockiPad。
每个任务从同一份 Motus/Thinker-WM 预训练权重独立初始化；默认每任务 20,000 步、每 5,000 步保存与开发集离线验证。

完整的数据格式、环境要求、配置生成、四卡冒烟、逐任务训练、队列与续训命令见 **[训练指南](docs/DEXJOCO_TRAINING.md)**。

## 下载

Windows PowerShell、Linux 或 macOS 中运行（需要已安装 Git；公开仓库无需 GitHub 登录）：

```powershell
git clone https://github.com/wuyimingai2004-cloud/DexJoCo-H200-Training.git
cd DexJoCo-H200-Training
```

Windows 可以下载和编辑代码。四卡训练入口面向 Linux CUDA/H200 环境；下载源码不等于已经安装训练环境。

## 下载六任务训练数据（Windows）

本轮使用第三方 `rand_obj` **LeRobot v2 ZIP**，六个包约 4.03 GB；解压后约 4.05 GB。
无需继续下载 Raw/Zarr。以下脚本只依赖 Python 标准库和 Windows 自带 `curl.exe`，
支持 `.part` 续传、固定版本和 SHA256 校验；已有下载不覆盖，原 Raw 文件也不删除。

```powershell
python .\scripts\download_dexjoco_data.py --output "C:\Users\Admin\Desktop\yiming_wang\DexJoCo_training_data"
```

中断用 Ctrl+C；同一命令重跑可续传。浏览器下载的六个 ZIP 也可放到上述目录，
加 `--verify-only` 仅校验。校验通过后将每个 ZIP **解压到同一个数据根目录**，
各包自带任务文件夹；保留 `meta/`、`data/`、`videos/` 全部内容。
不要再递归解压 MP4/Parquet。训练时数据路径应为 Linux 上解压后的位置。

数据来源和 manifest 写法见训练指南的「LeRobot v2 数据」部分。
本入口不支持官方 LeRobot v3 的共享视频分片；不能把 v3 数据直接替换进来。

## 在 Linux GPU 环境运行

使用包含 PyTorch/CUDA、Accelerate、DeepSpeed、Transformers 等 WAM 依赖的 Python 3.10+ 环境；完整依赖说明见训练指南。
专家轨迹、模型权重和训练输出需单独准备，不随源码分发。

从仓库根目录查看入口：

```bash
python -m thinker_wm.data_prepare.dexjoco --help
```

按训练指南依次运行 `prepare` → `configs` → `train` / `queue`。
`configs` 要求显式指定真实数据/权重路径和你选择的 `--name-prefix`；不会覆盖已有配置。
`train` / `queue` 默认只预览，加 `--check` 做环境检查，只有加 `--execute` 才启动训练。
正式运行前先做 100–200 步四卡冒烟。

## 验证状态

已完成本地数据/启动逻辑、22/44 维小模型训练和 Joint24 回归验证；原始 Zarr 与 MP4 读取也有本地验证。新增 LeRobot v2 六任务数据通路有 CPU 回归测试。
尚未在真实 H200、完整 5B 模型、正式数据或模拟器闭环上验证；具体测试与已知限制见训练指南。

## 目录

- `third_party/Thinker-WM/data_prepare/dexjoco/`：六任务契约、数据准备、配置和训练启动。
- `third_party/Thinker-WM/models/`、`train/`：模型与训练器；包含本次显式动作接口迁移。
- `thinker_wm.py`：模型源码的统一 Python 导入入口。
- `tests/test_dexjoco_training.py`、`scripts/test_dexjoco_model.py`：本次数据与模型测试。
- `docs/DEXJOCO_TRAINING.md`：训练操作说明。
- `execution/trajectory.py`：模型推理接口共用的动作哈希依赖。
- `third_party/Thinker-WM/libs/`：训练数据加载包与本地计时/日志 API；不包含监控服务、远程同步或管理工具。

来源与公开快照整理说明见 [PUBLICATION.md](docs/PUBLICATION.md)；第三方组件保留原有文件头、许可证与来源记录。
