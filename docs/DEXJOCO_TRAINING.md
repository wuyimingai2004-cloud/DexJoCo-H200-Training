# DexJoCo 六任务四卡训练

按用户提供的《DexJoCo实验训练计划报告_20261108(1).docx》实现。
代码复用 `third_party/Thinker-WM/train/train.py`，数据与启动逻辑位于
`third_party/Thinker-WM/data_prepare/dexjoco/`。以下命令均从本仓库根目录运行。

## 固定实验设置

| 报告任务 | 官方 task ID | 动作维度 | 原生状态维度 | 默认训练 seed |
|---|---|---:|---:|---:|
| PickBucket | `pick_bucket` | 22 | 23 | 2026110901 |
| Microwave | `bimanual_microwave_cook` | 44 | 46 | 2026110902 |
| ClickMouse | `click_mouse` | 22 | 23 | 2026110903 |
| FoldGlasses | `fold_glasses` | 22 | 23 | 2026110904 |
| Hanoi | `bimanual_hanoi` | 44 | 46 | 2026110905 |
| UnlockiPad | `bimanual_unlock_ipad` | 44 | 46 | 2026110906 |

每次单机四卡、每任务独立初始化和训练，顺序如表。
默认每卡 batch 8、梯度累积 1、全局 batch 32；显存不足时生成 batch 4 配置，
全局 batch 变为 16，不自动改变累积次数。每任务 20k **优化器更新步**，
每 5k 保存完整 checkpoint 并做开发数据的动作/视频与 A2V 离线验证。
验证结果不能解释为模拟器闭环成功率；报告中的 π0.5/WAM/WAM+A2V 闭环比较需另接评测器。

训练模式 `joint:A2V=0.60:0.40`；两项 loss 权重均为 1。
AdamW LR `1e-5`、betas `(0.9,0.95)`、weight decay `0.01`、梯度裁剪 `0.5`；
沿用 linear scheduler：warmup 100、cycle 1,000,000、乘数 `0.1 → 1.0 → 0.4`。
BF16、ZeRO-1、gradient checkpointing，首轮 RGB，深度与历史分支关闭。
30 Hz 动作，窗口 24 步；条件帧 `t`，未来视频 `t+2,t+4,…,t+24`，即 15 fps / 12 帧。
推理参数保留视频/动作 10/4 步去噪、CFG 5/1，记录执行前缀 16 步；训练 CLI 不执行环境动作。

## 数据契约

支持官方逐轨迹 `replay.zarr/data/{state,action_rotvec,timestamp}` 与 `videos/*.mp4`。
没有 `action_rotvec` 时，仅接受官方 23/46 维 quaternion `action` 并显式转换。
另外支持内容相同的 `replay.npz` 与 uint8 `[T,H,W,3]` `.npy` RGB，用于数据互换和 CPU fixture。
所有帧与状态/动作必须严格对齐为 30 Hz；不会截短、补齐或静默跳过坏轨迹。

单臂输出 `[xyz,rotvec,hand16]`；双臂输出
`[right_xyz,right_rotvec,right_hand16,left_xyz,left_rotvec,left_hand16]`。
原生双臂 state 是 `[right_pose7,left_pose7,right_hand16,left_hand16,…]`，
四元数为 **wxyz**。读取时只取前 23/46 个自身状态维度，转成与动作同序的 22/44 维模型状态；
后面的物体位置、任务真值等不进入模型。归一化使用本任务 **train action** 的均值和标准差，
标准差下限 `1e-3`；模型边界对转换后的 state 和 action 使用同一组统计，保持原训练器行为。

画布 480×640：上方外部相机；单臂腕相机在下方居中；双臂右腕在下方左侧、左腕在下方右侧。
保留纵横比并填黑边。默认相机为 `random_camera`、`wrist`，或
`random_camera`、`wrist_right`、`wrist_left`。使用其他相机时，manifest 中显式列出同序 `cameras`。
VLM 的布局描述同步使用上述顺序。

接口依据官方固定提交
[`8d23b0f` 自定义策略契约](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/docs/custom_policy_integration.md)、
[状态/动作转换](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py)、
[录制器](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/scripts/record_demos_zarr.py)。
没有复制或修改官方 DexJoCo 源码。

## 1 准备冻结的数据清单

清单必须人工核对原始轨迹组及布局；不能仅凭文件名声称布局未见。
每任务目标 100 条合格原始专家轨迹进入 train，开发轨迹来自另行准备的 dev 场景；
test 只登记分组，准备程序不会读取其轨迹内容。
同一原始轨迹的相机、增强、重放变体共用 `source_group`，不得跨 split。
默认也禁止同一任务的 `layout_id` 跨 split；若明确做常规场景测试而非严格 Layout 测试，
可传 `--allow-layout-overlap`，release 会记录这一限制。

`manifest.json` 的最小结构如下；实际清单要列全轨迹。
`episode_dir` 相对 manifest 所在目录解析，也可使用绝对路径。
`instruction` 必须来自相应任务/轨迹，不从测试答案或状态真值生成。

```json
{
  "schema": "dexjoco-training-manifest/v1",
  "alignment": "obs_t_action_t",
  "episodes": [
    {
      "task": "pick_bucket",
      "episode_dir": "raw/pick_bucket/demo_001",
      "source_group": "pick_bucket_original_001",
      "layout_id": "train_layout_001",
      "split": "train",
      "instruction": "Place the boxed food into the bucket and then lift the bucket."
    },
    {
      "task": "pick_bucket",
      "episode_dir": "raw/pick_bucket/dev_001",
      "source_group": "pick_bucket_dev_001",
      "layout_id": "dev_layout_001",
      "split": "dev",
      "instruction": "Place the boxed food into the bucket and then lift the bucket."
    }
  ]
}
```

```bash
python -m thinker_wm.data_prepare.dexjoco prepare \
  --manifest /path/to/manifest.json --output /path/to/prepared_release
```

可加 `--tasks pick_bucket` 先准备一项。数据不足 100 条的接口调试必须显式传
`--target-episodes N`，该数量写入 release，不能报告成 100 条正式实验。
检查包括低维长度、有限数值、四元数、30 Hz 时间戳、视频逐帧解码和帧数对齐。
产物包括任务独立 train/dev Catalog、SQLite ranges、模型状态/动作 `.npy`、统计和哈希。
准备失败会保留未完成目录供排查；只有生成 `release.json` 的 release 可启动。
重新准备时选择新输出目录，不覆盖已有 release。

## 2 生成六份配置

需要一个结构兼容的 **Motus/Thinker-WM checkpoint** 作为六项共同初始化来源，
以及 Wan2.2-TI2V-5B 和 Qwen3-VL-2B-Instruct 的本地目录（VAE 也必须存在）。
π0.5 checkpoint 不能用于这个训练器。
正式数据、初始化权重路径和配置文件名需要使用实际值；代码没有写入任何旧集群路径。

仓库要求用户决定配置名称和是否新建。生成器要求显式 `--name-prefix`，只创建新文件，
存在同名文件时失败。`CONFIG_PREFIX` 应设置为你决定的名称；如遵循分支规则，可采用
该 dev 分支对应的 `stage2_<分支日期>_v1` 前缀。不要未经确认将示例名当作正式实验名。

```bash
python -m thinker_wm.data_prepare.dexjoco configs \
  --release /path/to/prepared_release \
  --wan-dir /path/to/Wan2.2-TI2V-5B \
  --vlm-dir /path/to/Qwen3-VL-2B-Instruct \
  --init-checkpoint /path/to/shared_motus_checkpoint \
  --output-root /path/to/dexjoco_runs \
  --config-dir /path/to/task_configs \
  --name-prefix "$CONFIG_PREFIX"
```

这会生成 `${CONFIG_PREFIX}_pick_bucket.yaml` 等六份完整配置。
每项 seed 默认各不相同；新实验若需不同采样序列，应在生成后为新配置指定新的非零 seed。
batch 4 回退需加 `--batch-size 4` 并选择新配置目录/名称，保留 batch 8 的实测记录。

## 3 检查与四卡冒烟

训练 Python 环境需要原 WAM 依赖，包括 PyTorch/CUDA、Accelerate、DeepSpeed、Transformers、
Diffusers、OmegaConf、Qwen VL 工具、safetensors、TensorBoard、NumPy、Pillow、OpenCV；
准备官方 raw 数据额外需要 Zarr（官方原始数据是 Zarr v2）。
本仓库 motus-data-factory / motus-monitor 路径由启动器注入。
纯离线训练不实例化 MuJoCo；EGL 属于后续环境测试/闭环评测要求。

先预览实际命令和输出位置，不训练、不创建 run 目录：

```bash
python -m thinker_wm.data_prepare.dexjoco train \
  --config "/path/to/task_configs/${CONFIG_PREFIX}_pick_bucket.yaml" \
  --run-name smoke_200 --smoke-steps 200
```

加 `--check` 核对权重、资产哈希、依赖和可见的四张 GPU；加 **`--execute` 才启动训练**。
启动器只启动单机 `torchrun --nproc_per_node=4`，不连接集群、不提交 K8s 作业、不自动重试。
使用已分配的四卡环境；按需要在外部设置 `CUDA_VISIBLE_DEVICES`，启动器不改动 GPU 分配。

```bash
python -m thinker_wm.data_prepare.dexjoco train \
  --config "/path/to/task_configs/${CONFIG_PREFIX}_pick_bucket.yaml" \
  --run-name smoke_200 --smoke-steps 200 --check
```

```bash
python -m thinker_wm.data_prepare.dexjoco train \
  --config "/path/to/task_configs/${CONFIG_PREFIX}_pick_bucket.yaml" \
  --run-name smoke_200 --smoke-steps 200 --execute
```

冒烟输出与正式 run 分开。冒烟末尾保存并运行开发数据离线验证。
检查 `train.log`、`initialization_audit.json`、`step_timing/step_timing_rank*.jsonl`、
`validation/` 和 checkpoint 完成标记。
计时记录包含训练耗时及每 10 步的峰值显存采样；稳态吞吐须排除启动、保存和验证步骤。
本地 CPU 测试不能确定每卡 batch 8 是否能放下，也不能确定真实四卡吞吐。

## 4 分任务正式训练

例如只训练 PickBucket（同一命令替换配置名即可训练其他五项）：

```bash
python -m thinker_wm.data_prepare.dexjoco train \
  --config "/path/to/task_configs/${CONFIG_PREFIX}_pick_bucket.yaml" \
  --run-name train_20k --execute
```

顺序运行全部六项：

```bash
python -m thinker_wm.data_prepare.dexjoco queue \
  --configs /path/to/task_configs/"${CONFIG_PREFIX}"_*.yaml \
  --run-name train_20k --execute
```

queue 先检查六项齐全、seed 独立、共同初始化路径；按上表顺序每项占用全部四卡。
任何任务报错、提前停止或缺少目标完整 checkpoint，就立即停止队列。
不会用上一任务的已训 checkpoint 初始化下一任务。
默认无 `--execute` 时只是预览；也可为六项 queue 加 `--smoke-steps 100`。

输出路径为：

```text
<output-root>/<task>/checkpoints/<config-stem>/<run-name>/
  inputs/<config-name>.yaml
  run_identity.json
  action_stats.json
  initialization_audit.json
  train.log
  tensorboard/
  step_timing/
  checkpoint_step_5000/ ... checkpoint_step_20000/
```

初始化迁移明确重置 action embedding、state embedding、action unembedding、robot tokens；
即使张量形状碰巧一致也不复用旧 joint14 的动作语义。
其余共享权重必须完整且形状一致；只允许丢弃已移除旧模块和本轮禁用的深度/历史模块，
任何其他缺失或不匹配均报错。rank 0 保存源权重 SHA256 和逐键加载/丢弃/初始化清单。
真实权重未提供时，只能验证迁移规则，无法证明某个实际 checkpoint 兼容。

## 5 同任务续训

只接受本任务、本配置、本 run 下带完成标记的 checkpoint；恢复模型、优化器、调度器和训练状态。
关闭自动查找 latest。先确认开发结果再将累计预算扩到 30k 或 60k：

```bash
python -m thinker_wm.data_prepare.dexjoco train \
  --config "/path/to/task_configs/${CONFIG_PREFIX}_pick_bucket.yaml" \
  --run-name train_20k \
  --resume "/path/to/dexjoco_runs/pick_bucket/checkpoints/${CONFIG_PREFIX}_pick_bucket/train_20k/checkpoint_step_20000" \
  --max-steps 30000 --execute
```

原始配置、数据 release、统计不能在同一 run 中更换。重新定义数据/归一化需要新实验。
普通中断续训可去掉 `--max-steps`，仍以 20k 为目标。队列不隐式续训；中断任务单独恢复后，
后续任务按各自配置启动。

## 本地验证

```bash
python -m unittest discover -s tests -p 'test_dexjoco_training.py' -v
python scripts/test_dexjoco_model.py
```

覆盖合成单/双臂数据准备、特权信息剔除、训练统计隔离、布局/原始轨迹分组隔离、
四 rank 数据分片、真实 CPU batch、动作/视频时间对齐、完整开发集分片、
checkpoint 迁移拒绝规则、启动预览和跨任务续训拒绝。
另以现有 tiny WAM harness 检查 22/44 维 joint/A2V 前后向、参数更新，以及从 joint14 迁移的实际加载方法。
该 harness 替换 VAE/VLM 和 CUDA attention kernel；不包含真实专家数据验收、完整 5B 模型前后向、
ZeRO checkpoint 重载、GPU 显存/吞吐或闭环成功率。

本次本地验证（2026-11-09）：数据/启动测试 13 项、tiny WAM 3 项、原 Joint24 9 项、
原训练模式 15 项，共 40 项通过。Zarr 在主测试环境缺依赖而跳过，另用已安装的 Zarr 2.12.0
单独验证了实际 Zarr 单/双臂读取、quaternion 动作转换和坏时间戳拒绝；MP4 测试使用真实编码/解码。
既有 `test_training_stop_and_timing.py` 的正常保存/停止/重载用例通过，但后续验证 barrier
在 macOS MPS 后端报错，整个双进程套件未通过。该测试涉及的训练循环、保存和重载方法本次未改动。
H200 使用 CUDA/NCCL，仍需执行上面的四卡冒烟来验收其实际分布式路径。
