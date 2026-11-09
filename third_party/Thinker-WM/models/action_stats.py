"""Per-robot action/state 逐维归一化统计量（H 集群训练集上计算，剔除离群值）。

统计来源：``data_prepare/compute_action_stats.py`` 对每个 catalog 的 train
split（holdout 1%）逐 episode 读取 reader 归一化后的完整 state/action 时间线，
reservoir 采样后取 [0.5%, 99.5%] 分位数边界，边界内重算 trimmed mean/std。

约定：
- 键为 ``dataset.robot_names`` 下的规范机器人名（见 configs/*.yaml）。
- robotwin / agibot_beta 的 state 与 action 来自同一时间线（qpos / canonical
  timeline），共用同一组 mean/std。
- 未注册机器人回退为恒等归一化（mean=0, std=1），行为与旧版一致。
- 归一化在模型边界执行：``UBT_WM.training_step`` / ``inference_step`` /
  ``inference_action_step`` 的输入（state/actions）先归一化、输出反归一化，
  使 flow matching 全程在单位尺度空间进行，损失在归一化空间计算。

两套策略（``model.wm.action_norm_policy`` 选择）：
- ``zscore``（旧，默认）：全维 per-robot z-score（``normalize_batch`` 等）。
- ``gripper_hand_minmax``（B3，20260811 对齐 MotusBrain 论文）：仅夹爪/
  灵巧手维归一化 [-1,1]，其余维保持物理量纲——见文件末尾
  ``GripperHandNormPolicy`` / ``build_gripper_hand_policy``。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# 规范机器人名 -> 逐维 mean/std（训练所见原始数据单位；std 为总体标准差，
# 已做防零保护：常数维记为 1.0）。
ACTION_STATS: dict[str, dict[str, list[float]]] = {
    # robotwin_3d Aloha（Flex-π pack ``dataset_stats.json`` action.global_*，
    # 27500 episode / 6.075M 帧）。catalog ``robot`` 字段是 ``aloha``；与
    # aloha-agilex（旧 Motus 四本体 catalog）分开。state/action 共用这一套
    # （预测目标是 action；Flex-π 本身分两套，mean 最多差 ~0.09）。
    "aloha": {
        "mean": [-0.243576, 1.16261, 0.831938, -0.336797, 0.0648053, -0.0738379, 0.646813, 0.242021, 1.19650, 0.870308, -0.369014, -0.0276741, 0.0772718, 0.637836],
        "std": [0.399440, 1.01129, 0.787508, 0.700197, 0.289987, 0.701046, 0.460550, 0.363268, 1.02408, 0.804712, 0.721390, 0.272609, 0.723612, 0.463131],
        "q_low": [None, None, None, None, None, None, 0.0, None, None, None, None, None, None, 0.0],
        "q_high": [None, None, None, None, None, None, 1.0, None, None, None, None, None, None, 1.0],
    },
    # RoboDojo ARX-X5 绝对关节 14D（RoboDojo_depth HDF5，field map
    # configs/robodojo/arx_x5_joint14_v1；catalog robodojo_3task_v1 的 ranges
    # train split，285 条、95684 行，不含 15 条 holdout；action
    # [0.5%,99.5%] 分位内 trimmed）。state/action 同一时间线（action[t]==state[t+1]）。
    "robodojo_arx_x5": {
        "mean": [-0.177566, 0.993921, 0.612433, -0.0110625, 0.004984, 0.228026, 0.775148, 0.0707126, 0.682775, 0.442828, -0.102493, -0.0593313, -0.0788455, 0.836054],
        "std": [0.307465, 0.878235, 0.627221, 0.774438, 0.287318, 0.7714, 0.333578, 0.223485, 0.860413, 0.611829, 0.620991, 0.189002, 0.422271, 0.318055],
        "q_low": [None, None, None, None, None, None, 0.0, None, None, None, None, None, None, 0.0],
        "q_high": [None, None, None, None, None, None, 1.0, None, None, None, None, None, None, 1.0],
    },
    # robotwin（20260727_full_v1 catalog，train split holdout 1%，[0.5%,99.5%]
    # 分位数内 trimmed；5.97M/4.03M/3.01M/3.68M 行）
    "aloha-agilex": {
        "mean": [-0.214682, 1.10437, 0.787094, -0.313082, 0.0628897, -0.0360381, 0.66737, 0.216109, 1.07516, 0.777772, -0.332098, -0.0234468, 0.0449654, 0.67792],
        "std": [0.283355, 0.999229, 0.762081, 0.657518, 0.245168, 0.50608, 0.452709, 0.278225, 1.01866, 0.782167, 0.667579, 0.213928, 0.526117, 0.448801],
        # B3（gripper_hand_minmax）：夹爪维（local 6/13）[-1,1] 分位边界——
        # 物理域即 [0,1]（median=1.0、min/max 触界；micro_data 32 条 subset
        # 暂行值，分布与全量一致，待集群全量重算确认；逐维列表，None=无值）
        "q_low": [None, None, None, None, None, None, 0.0, None, None, None, None, None, None, 0.0],
        "q_high": [None, None, None, None, None, None, 1.0, None, None, None, None, None, None, 1.0],
    },
    "ARX-X5": {
        "mean": [-0.291721, 1.0762, 0.715878, -0.356285, 0.0385862, -0.0957774, 0.631572, 0.269593, 0.974688, 0.634734, -0.306191, 0.00478614, 0.0558114, 0.682175],
        "std": [0.341721, 0.932221, 0.664728, 0.598518, 0.245744, 0.560197, 0.458225, 0.348887, 0.963835, 0.66588, 0.649232, 0.247403, 0.574078, 0.441444],
        # B3（gripper_hand_minmax）：夹爪维 [-1,1] 分位边界（同 aloha-agilex 注释）
        "q_low": [None, None, None, None, None, None, 0.0, None, None, None, None, None, None, 0.0],
        "q_high": [None, None, None, None, None, None, 1.0, None, None, None, None, None, None, 1.0],
    },
    "piper": {
        "mean": [-0.192398, 1.13913, -0.795218, 0.0882102, 0.24513, -0.103635, 0.654652, 0.18643, 1.03546, -0.714769, -0.0326975, 0.249886, 0.0824065, 0.678908],
        "std": [0.28169, 0.997943, 0.723764, 0.476901, 0.605396, 0.665859, 0.449859, 0.270312, 1.00622, 0.716065, 0.378484, 0.561563, 0.598688, 0.44159],
        # B3（gripper_hand_minmax）：夹爪维 [-1,1] 分位边界（同 aloha-agilex 注释）
        "q_low": [None, None, None, None, None, None, 0.0, None, None, None, None, None, None, 0.0],
        "q_high": [None, None, None, None, None, None, 1.0, None, None, None, None, None, None, 1.0],
    },
    "ur5-wsg": {
        "mean": [-1.67738, -1.79957, -1.47424, -1.2423, 1.51986, 0.0654678, 0.636124, -1.20465, -1.82222, -1.49942, -1.17616, 1.54915, 0.169396, 0.678895],
        "std": [0.269102, 0.26012, 0.275923, 0.54256, 0.249201, 0.62455, 0.455711, 0.481679, 0.319839, 0.431514, 0.57802, 0.363802, 0.561752, 0.441826],
        # B3（gripper_hand_minmax）：夹爪维 [-1,1] 分位边界（同 aloha-agilex 注释）
        "q_low": [None, None, None, None, None, None, 0.0, None, None, None, None, None, None, 0.0],
        "q_high": [None, None, None, None, None, None, 1.0, None, None, None, None, None, None, 1.0],
    },
    # agibot_beta（20260723_concise_v2 catalog，train split holdout 1%；reader 内
    # 已做 eef z-score，此处为其基础上的二次归一化；292.6M 行。注意原始 raw std
    # 在夹爪维 12/25 高达 ~390（离群值），trimmed 后 0.74/0.57）
    # 修复 #7（20260810 问题报告 §3.5/§4.6 实锤）：灵巧手占位维（local 7-12/
    # 20-25）在 reader 输出中 99.9% 样本恒 0（6272 eval 样本实测 p99=0，仅 5
    # 条离群样本非 0），而下方的原始烘焙 stats 非退化（陈旧）——归一化后
    # 12/26 维是平凡常数目标，占 action loss 权重 46%。这些维已在
    # stage2_20260807_v1 的 unified_dim_map 中以 -1 摘除（不进 scatter/loss），
    # 此处 stats 同步置恒等（mean=0/std=1：常数 0 归一化后仍为 0），
    # 避免陈旧统计把恒 0 输入变成非 0 常数。全量重算（compute_action_stats.py）
    # 为集群待办（报告 §7.2）。
    "agibot-a2-d": {
        "mean": [-0.0436013, 0.959705, 0.120185, 0.640076, 0.478484, -0.529595, 0.0571307, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0731212, -0.862537, 0.123467, -0.599304, -0.30555, 0.385161, 0.131722, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "std": [0.93622, 0.286331, 0.948347, 0.684591, 1.0184, 1.26378, 1.02352, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.971636, 0.291833, 0.93572, 0.626313, 0.482674, 0.45775, 1.01653, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        # ---- B3（gripper_hand_minmax 策略，20260811）----
        # reader_z_*：agibot reader 内置全维 z-score 的每臂 13 维常量（两臂
        # 平铺成 26 维；复制自 libs/motus-data-factory/motus_data/datasets/
        # agibot_common.py 的 AGIBOT_ACTION_MEAN/STD，submodule 更新需同步）。
        # 策略先按 x*std+mean 逆变换还原物理量纲（pos 米 / rotvec 弧度 /
        # 夹爪 mm），输出端再还原回 reader 空间。
        "reader_z_mean": [0.6623259, -0.02415569, 0.65893349, -0.91274923, 0.93820975, -0.62816762, 64.11719045790569, -0.01470473, 0.27863706, 0.4925647, 0.47402732, 0.51696799, -1.0560014, 0.6623259, -0.02415569, 0.65893349, -0.91274923, 0.93820975, -0.62816762, 64.11719045790569, -0.01470473, 0.27863706, 0.4925647, 0.47402732, 0.51696799, -1.0560014],
        "reader_z_std": [0.08038633, 0.3268345, 0.19179442, 1.23554225, 1.35022756, 0.92738767, 37.00465005224069, 0.212378374, 0.309977879, 0.475664546, 0.469941935, 0.490734943, 0.00011713679, 0.08038633, 0.3268345, 0.19179442, 1.23554225, 1.35022756, 0.92738767, 37.00465005224069, 0.212378374, 0.309977879, 0.475664546, 0.469941935, 0.490734943, 0.00011713679],
        # 夹爪维（local 6/19）[-1,1] 归一化的分位边界（物理 mm 空间）：
        # 待集群全量重算（compute_action_stats.py，raw 空间 q_low/q_high），
        # 暂置 None → 回退物理空间 z-score（phys_*，取值即 reader 常量本身
        # ——它们是 raw 空间统计）。q_*/phys_* 为逐维列表，None = 无值。
        "q_low": [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        "q_high": [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        "phys_mean": [None, None, None, None, None, None, 64.11719045790569, None, None, None, None, None, None, None, None, None, None, None, None, 64.11719045790569, None, None, None, None, None, None],
        "phys_std": [None, None, None, None, None, None, 37.00465005224069, None, None, None, None, None, None, None, None, None, None, None, None, 37.00465005224069, None, None, None, None, None, None],
    },
    # agibot-beta-parallel-gripper（20260805_full_v1 effector profile catalog，
    # train split holdout 1%，[0.5%,99.5%] 分位内 trimmed；15.9 万 episode /
    # 2.83 亿行，20260815 集群重算）。reader 侧 catalog-bound native_zscore_v1
    # 全维 z-score；mean/std 为 reader 输出（z）空间 = 模型输入空间统计（仅供
    # 维度推断与 zscore 策略回退）。B3 策略经 reader_z_*（native_zscore_v1
    # artifact 的物理常量：pos 米 / rotvec 弧度 / 夹爪 mm）先还原物理量纲，
    # 与 egoverse 等 raw 物理源保持量纲一致；夹爪维（local 6/13）min-max
    # 边界为物理 mm 空间（由 z 空间分位边界按 q*std+mean 换算，std>0 保序）。
    "agibot-beta-parallel-gripper": {
        "mean": [-0.0068267, 0.0071278, 0.0113428, 0.0196091, -0.01688, 0.0157987, 0.00726448, -0.000126806, 0.00873992, 0.00247849, 0.000185441, 0.0225361, 0.013058, 0.00386169],
        "std": [0.962361, 0.956941, 0.984157, 0.999048, 0.993642, 1.00494, 1.00364, 0.956933, 0.957766, 0.981494, 0.929862, 0.945679, 0.941209, 0.999642],
        "reader_z_mean": [0.65938509488871, 0.29021300670865025, 0.6806331973336623, -0.14976532111439056, 1.6215168083323597, -1.145697808654197, 65.9033332997984, 0.6675156260657193, -0.30856974711449575, 0.6811275972669674, -1.6748283863716942, 0.5143456943626512, -0.2753479066685935, 68.8355509635989],
        "reader_z_std": [0.07717464765967154, 0.09740468135132897, 0.18451895935765378, 0.8321926659225375, 1.3829463910947344, 1.1526983064659666, 38.280844045565104, 0.08126659091144207, 0.09939216114639468, 0.18182083637487145, 0.7721671107614402, 0.6825192614939956, 0.44637761780783874, 38.23006706929753],
        "q_low": [None, None, None, None, None, None, 34.488886, None, None, None, None, None, None, 34.666664],
        "q_high": [None, None, None, None, None, None, 123.000016, None, None, None, None, None, None, 122.916377],
    },
    # agibot-beta-dexterous-hand（20260805_full_v1 effector profile catalog，
    # train split holdout 1%，[0.5%,99.5%] 分位内 trimmed；7380 episode（reader
    # 数值质量策略排除 1 条已知异常 episode agibot_beta:554:913085）/ 920 万行，
    # 20260816 集群重算）。mean/std 为 reader 输出（z）空间统计（仅供维度推断
    # 与 zscore 策略回退）；reader_z_* 为 native_zscore_v1 artifact 物理常量
    # （pos 米 / rotvec 弧度 / 灵巧手关节 rad）。U=20 布局无灵巧手 band：12 个
    # 手关节维 dim_map=-1 摘除、gripper 列不映射（norm01 维为空，无 q 边界）
    # ——条目作用是让 B3 策略把 pos/rot 逆变换还原为物理量纲，与其他源一致。
    "agibot-beta-dexterous-hand": {
        "mean": [-0.251801, -0.0706278, 0.0598486, 0.1064841, -0.1457308, 0.0766947, 0.0600639, -0.0934358, -0.0868256, -0.0851934, -0.0887274, 0.0472835, -0.0276027, 0.0495577, 0.1988229, 0.1360557, -0.0776372, -0.0178062, 0.0921043, -0.1041345, -0.0935713, -0.087815, -0.0819908, 0.0506291],
        "std": [1.1989747, 1.0035249, 1.1440238, 0.9628721, 1.027037, 0.9972054, 0.9736369, 0.9242025, 0.9272554, 0.9328515, 0.9397341, 1.0467572, 0.9384273, 0.9652903, 1.1284027, 0.9027017, 1.0680587, 1.1619127, 0.9963019, 0.973295, 0.9866478, 0.9938134, 1.0008784, 1.0492293],
        "reader_z_mean": [0.6705765568717524, 0.2646777270476422, 0.6737923187506836, 0.23458874691138176, 0.5757046865371129, -0.35717915951465207, 0.17761420736970898, -0.004278507875947204, 0.09258548594235456, 0.09851235188255762, 0.13399831884214405, -0.9599579827645404, 0.6717023681661592, -0.25560862262140965, 0.6842753358136874, -1.1357371432911203, 0.9103478433247453, -0.29993969787802477, 0.11461439947051341, 0.1279916523695502, 0.2328801777162228, 0.2359581643072533, 0.27994130871335476, -0.959718035345144],
        "reader_z_std": [0.13290108000755624, 0.08651604016521619, 0.15855997464531318, 1.4775737824723263, 1.3798929991909812, 1.6563089936166615, 0.1672980411805863, 0.27522100487578344, 0.450622702467097, 0.45877064034688764, 0.5004293504329183, 0.24022173047378292, 0.10212489941853471, 0.10262862071343758, 0.15828283524527503, 0.5230653178799581, 0.7470139608017677, 0.5662883499591238, 0.19311576614551929, 0.35461761095870953, 0.476967642648462, 0.4782673112752327, 0.5244234531429841, 0.24069409964067792],
    },
    # egoverse-eva-bimanual（20260730_first6_v1 catalog，train split holdout
    # 1%，分位内 trimmed；2847 episode / 321 万行，20260815 集群重算）。
    # reader raw 物理量纲无归一化，统计空间=模型输入空间；夹爪维（local
    # 7/15）物理域即 [0,1]（median=1），min-max 即 2x-1。egoverse 其余
    # profile（aria/scale/mecka）无夹爪维，无需条目。
    "egoverse-eva-bimanual": {
        "mean": [0.211442, -0.0228707, 0.23042, -0.0859851, 0.175279, -0.0443099, 0.277082, 0.609663, 0.23198, 0.0194213, 0.231108, 0.135204, -0.0379267, 0.17799, 0.0104926, 0.584206],
        "std": [0.0950797, 0.100975, 0.0716965, 0.387394, 0.425018, 0.491059, 0.546118, 0.478985, 0.0986902, 0.112003, 0.0880428, 0.432204, 0.412032, 0.561123, 0.513406, 0.481818],
        "q_low": [None, None, None, None, None, None, None, 0.0, None, None, None, None, None, None, None, 0.0],
        "q_high": [None, None, None, None, None, None, None, 1.0, None, None, None, None, None, None, None, 1.0],
    },
    # robotmind×4（micro_data 4-8 条 subset 暂行值，20260815 本机计算，仅供
    # 本机 micro debug 的 B3 夹爪 min-max 使用；生产全量统计待集群有
    # robotmind 全量 catalog 后重算替换）。
    "robotmind-tianyi": {
        "mean": [-0.427882, 0.0745409, -0.271584, -0.328756, 0.163995, -0.125799, -0.0472314, 0.773178, -0.216084, 0.215249, 0.631396, -0.811021, 0.150223, 0.247359, 0.988036, 0.712401],
        "std": [0.541816, 0.113581, 0.314962, 0.244095, 0.574888, 0.151046, 0.141871, 0.620121, 0.251879, 0.238295, 0.429868, 0.78112, 0.170614, 0.26329, 0.034272, 0.212427],
        "q_low": [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.81564, 0.497],
        "q_high": [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1.0, 0.933],
    },
    "robotmind-arkmobile": {
        "mean": [-0.153932, 0.58687, 0.324133, 0.317494, -0.0539328, 0.0236447, -0.0282564, 1.26225, 0.820307, 0.226381, -0.201268, 0.00837183, 3.5267, 0.925483, -1.24015, 0.439571, 0, 0, 0, 0.898184],
        "std": [0.195623, 0.679388, 0.364707, 0.377654, 0.15424, 0.0373434, 0.160515, 0.881159, 0.699169, 0.28992, 0.178315, 0.122913, 2.20056, 1.37598, 2.19405, 1.06257, 1, 1, 1, 1.36834],
        "q_low": [None, None, None, None, None, None, None, None, None, None, None, None, -0.0165486, -0.0110631, None, None, None, None, None, None],
        "q_high": [None, None, None, None, None, None, None, None, None, None, None, None, 4.90463, 4.90349, None, None, None, None, None, None],
    },
    "robotmind-ark": {
        "mean": [-0.0471533, 0.764777, 0.44368, 0.197758, 0.191175, 0.437761, -14.1962, 0.0842037, 1.129, 0.850643, -0.125153, -0.243221, -0.548597, -8.17126],
        "std": [0.283581, 0.618599, 0.505742, 0.365137, 0.332275, 0.60071, 15.5435, 0.188802, 0.737334, 0.603956, 0.4545, 0.377535, 0.968326, 17.5],
        "q_low": [None, None, None, None, None, None, -31.6577, None, None, None, None, None, None, -31.8446],
        "q_high": [None, None, None, None, None, None, 6.80217, None, None, None, None, None, None, 23.8375],
    },
    "robotmind-tienkung": {
        "mean": [-0.596879, 0.329675, -0.444054, -1.01177, 0.0363376, -0.141965, -0.222943, 0.611204, -0.555362, 0.228675, 0.582189, -0.124027, 0.168295, 0.262078, 0.745434, 0.747232],
        "std": [0.295733, 0.267008, 0.378902, 0.370164, 1.21146, 0.292469, 0.174141, 0.655697, 0.222331, 0.44977, 0.865648, 1.21923, 0.265729, 0.22034, 0.423668, 0.387692],
        "q_low": [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.0117647, 0.0117647],
        "q_high": [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1.0, 1.0],
    },
}


def get_stats(robot_name: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """返回该机器人的 (mean, std) 张量；未注册返回 None。"""
    stats = ACTION_STATS.get(str(robot_name))
    if not stats:
        return None
    mean = stats.get("mean")
    std = stats.get("std")
    if not mean or not std or len(mean) != len(std):
        return None
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(
        std, dtype=torch.float32
    )


def _stats_like(
    stats: Tuple[torch.Tensor, torch.Tensor], ref: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std = stats
    return (
        mean.to(device=ref.device, dtype=ref.dtype),
        std.to(device=ref.device, dtype=ref.dtype),
    )


def normalize(value: torch.Tensor, robot_name: str) -> torch.Tensor:
    """单机器人归一化：(x - mean) / std（value 末维为机器人维度）。"""
    stats = get_stats(robot_name)
    if stats is None:
        return value
    mean, std = _stats_like(stats, value)
    return (value - mean) * (1.0 / std)


def denormalize(value: torch.Tensor, robot_name: str) -> torch.Tensor:
    """单机器人反归一化：x * std + mean。"""
    stats = get_stats(robot_name)
    if stats is None:
        return value
    mean, std = _stats_like(stats, value)
    return value * std + mean


def _names_list(
    robot_names: Optional[Sequence[str]], batch: int
) -> Optional[list[str]]:
    if robot_names is None:
        return None
    if isinstance(robot_names, str):
        return [robot_names] * batch
    names = [str(name) for name in robot_names]
    if len(names) != batch:
        raise ValueError(
            f"robot_names length {len(names)} != batch {batch}"
        )
    return names


def _apply_per_name(
    value: torch.Tensor,
    names: Optional[list[str]],
    fn,
) -> torch.Tensor:
    """按 batch 内逐样本 robot_name 应用 fn（value: [B, ..., D]）。"""
    if value is None:
        return None
    if names is None:
        return value
    out = value.clone()
    for name in set(names):
        stats = get_stats(name)
        if stats is None:
            continue
        mask = [n == name for n in names]
        out[mask] = fn(value[mask], name)
    return out


def normalize_batch(
    state: Optional[torch.Tensor],
    actions: Optional[torch.Tensor],
    robot_names: Optional[Sequence[str]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """批量归一化 state/actions（逐样本 robot_name）。

    state: [B, D] 或 None；actions: [B, T, D] 或 None。
    robot_names: None（恒等）/ str（整批同一机器人）/ list[str]（逐样本）。
    """
    if robot_names is None:
        return state, actions
    batch = None
    if state is not None:
        batch = int(state.shape[0])
    elif actions is not None:
        batch = int(actions.shape[0])
    if batch is None:
        return state, actions
    names = _names_list(robot_names, batch)
    if names is None:
        return state, actions
    state = _apply_per_name(state, names, normalize)
    actions = _apply_per_name(actions, names, normalize)
    return state, actions


def denormalize_batch(
    actions: torch.Tensor,
    robot_names: Optional[Sequence[str]],
) -> torch.Tensor:
    """批量反归一化 actions（逐样本 robot_name），未注册/None 原样返回。"""
    if robot_names is None or actions is None:
        return actions
    names = _names_list(robot_names, int(actions.shape[0]))
    if names is None:
        return actions
    return _apply_per_name(actions, names, denormalize)


# ---------------------------------------------------------------------------
# B3（20260811，对齐 MotusBrain 论文）：gripper_hand_minmax 归一化策略。
#
# 仅夹爪/灵巧手维归一化到 [-1,1]（ACTION_STATS 条目的 q_low/q_high =
# 0.5%/99.5% 分位边界 min-max；缺边界回退物理空间 z-score（phys_mean/
# phys_std）；再缺恒等 + warning），其余维（含 dim_map -1 摘除维、rot
# group 维、关节/底盘维）保持物理量纲恒等。
#
# norm01 维分类规则：本地维经 dim_map 落入名称以 ``_gripper`` 结尾或以
# ``hand_`` 开头的 unified band（band 注册表 append-only，新 gripper/hand
# band 按同规则命名即自动覆盖）。
#
# agibot 的 reader 内置全维 z-score（不动 submodule，保住 build 期归一化
# 单位下构建的 ranges 有效性）：策略先按 ACTION_STATS 条目烘焙的
# reader_z_mean/reader_z_std 逆变换（x*std+mean）还原物理量纲，再做
# norm01 映射；denormalize 严格对称，输出回 reader 空间（下游
# periodic_eval/sample/monitor 零改动）。
# ---------------------------------------------------------------------------


def _norm01_local_dims(
    dim_map: Optional[Sequence[int]],
    bands: Optional[Dict[str, Tuple[int, int]]],
) -> List[int]:
    """dim_map 中落入 gripper/hand 类 band 的本地维下标（升序）。"""
    if not dim_map or not bands:
        return []
    dims = []
    for k, c in enumerate(dim_map):
        c = int(c)
        if c < 0:
            continue
        for band_name, (start, end) in bands.items():
            if start <= c < end and (
                band_name.endswith("_gripper") or band_name.startswith("hand_")
            ):
                dims.append(k)
                break
    return dims


class GripperHandNormPolicy:
    """B3 归一化策略：逐样本 robot_name 应用（与 normalize_batch 同调用约定）。

    state: [B, D] 或 None；actions: [B, T, D] 或 None。
    robot_names: None（恒等）/ str / list[str]。
    """

    def __init__(self, specs: Dict[str, Dict[str, Optional[torch.Tensor]]]):
        self._specs = specs

    def normalize(
        self,
        state: Optional[torch.Tensor],
        actions: Optional[torch.Tensor],
        robot_names: Optional[Sequence[str]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._apply(state, actions, robot_names, inverse=False)

    def denormalize(
        self,
        actions: Optional[torch.Tensor],
        robot_names: Optional[Sequence[str]],
    ) -> Optional[torch.Tensor]:
        _, actions = self._apply(None, actions, robot_names, inverse=True)
        return actions

    # ---- 内部 ----

    def _apply(self, state, actions, robot_names, inverse: bool):
        if robot_names is None:
            return state, actions
        batch = None
        if state is not None:
            batch = int(state.shape[0])
        elif actions is not None:
            batch = int(actions.shape[0])
        if batch is None:
            return state, actions
        names = _names_list(robot_names, batch)
        if names is None:
            return state, actions
        for name in set(names):
            spec = self._specs.get(str(name))
            if spec is None:
                continue
            mask = [n == name for n in names]
            if state is not None:
                state = state.clone()
                state[mask] = self._map_value(state[mask], spec, inverse)
            if actions is not None:
                actions = actions.clone()
                actions[mask] = self._map_value(actions[mask], spec, inverse)
        return state, actions

    @staticmethod
    def _to_ref(spec_tensor: Optional[torch.Tensor], ref: torch.Tensor):
        if spec_tensor is None:
            return None
        return spec_tensor.to(device=ref.device, dtype=ref.dtype)

    def _map_value(self, value, spec, inverse: bool):
        x = value
        dims = spec["norm01_dims"]
        rz_mean = self._to_ref(spec["reader_z_mean"], x)
        rz_std = self._to_ref(spec["reader_z_std"], x)
        if inverse:
            x = self._map_norm01(x, spec, dims, inverse=True)
            if rz_mean is not None:
                x = (x - rz_mean) * (1.0 / rz_std)
        else:
            if rz_mean is not None:
                x = x * rz_std + rz_mean
            x = self._map_norm01(x, spec, dims, inverse=False)
        return x

    def _map_norm01(self, x, spec, dims, inverse: bool):
        if dims is None or not dims.numel():
            return x
        dims = dims.to(device=x.device)
        dims = dims[dims < x.shape[-1]]
        if not dims.numel():
            return x
        q_low = self._to_ref(spec["q_low"], x)[dims]
        q_high = self._to_ref(spec["q_high"], x)[dims]
        p_mean = self._to_ref(spec["phys_mean"], x)[dims]
        p_std = self._to_ref(spec["phys_std"], x)[dims]
        use_q = (
            torch.isfinite(q_low)
            & torch.isfinite(q_high)
            & ((q_high - q_low).abs() > 1e-12)
        )
        use_z = (
            ~use_q
            & torch.isfinite(p_mean)
            & torch.isfinite(p_std)
            & (p_std.abs() > 1e-12)
        )
        x = x.clone()
        cur = x[..., dims]
        if not inverse:
            if use_q.any():
                cur[..., use_q] = (
                    2.0 * (cur[..., use_q] - q_low[use_q]) / (q_high[use_q] - q_low[use_q]) - 1.0
                )
            if use_z.any():
                cur[..., use_z] = (cur[..., use_z] - p_mean[use_z]) * (1.0 / p_std[use_z])
        else:
            if use_q.any():
                cur[..., use_q] = (
                    (cur[..., use_q] + 1.0) / 2.0 * (q_high[use_q] - q_low[use_q]) + q_low[use_q]
                )
            if use_z.any():
                cur[..., use_z] = cur[..., use_z] * p_std[use_z] + p_mean[use_z]
        x[..., dims] = cur
        return x


def build_gripper_hand_policy(
    robot_layout: Optional[Dict[str, Dict]],
    unified_bands: Optional[Dict[str, Sequence]],
) -> GripperHandNormPolicy:
    """按 robot_layout + unified_bands 构建 B3 归一化策略（模型边界调用）。

    每机器人 spec：norm01 本地维（dim_map → gripper/hand band 推导）+
    ACTION_STATS 的 q_low/q_high/phys_mean/phys_std/reader_z_*（逐维列表，
    None/缺失 = 无值；长度与 stats mean 或 dim_map 不一致视为全缺）。
    """
    bands = None
    if unified_bands:
        bands = {
            str(band): (int(span[0]), int(span[1]))
            for band, span in unified_bands.items()
        }
    specs: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
    for name, entry in (robot_layout or {}).items():
        dim_map = entry.get("dim_map")
        dims = _norm01_local_dims(dim_map, bands)
        stats = ACTION_STATS.get(str(name)) or {}
        dim = len(stats.get("mean") or []) or (len(dim_map) if dim_map else 0)

        def _vec(key: str) -> torch.Tensor:
            raw = stats.get(key)
            if not raw or len(raw) != dim:
                return torch.full((dim,), float("nan"), dtype=torch.float32)
            return torch.tensor(
                [float(v) if v is not None else float("nan") for v in raw],
                dtype=torch.float32,
            )

        def _z_vec(key: str) -> Optional[torch.Tensor]:
            raw = stats.get(key)
            if not raw or len(raw) != dim:
                return None
            return torch.tensor([float(v) for v in raw], dtype=torch.float32)

        q_low, q_high = _vec("q_low"), _vec("q_high")
        phys_mean, phys_std = _vec("phys_mean"), _vec("phys_std")
        if dims:
            d = torch.tensor(dims, dtype=torch.long)
            has_q = (
                torch.isfinite(q_low[d]) & torch.isfinite(q_high[d])
            )
            has_z = torch.isfinite(phys_mean[d]) & torch.isfinite(phys_std[d])
            if not bool((has_q | has_z).all()):
                logger.warning(
                    "gripper_hand_minmax: robot %s 的夹爪/灵巧手维 %s 在 "
                    "ACTION_STATS 中既无 q_low/q_high 也无 phys_mean/phys_std，"
                    "这些维将保持物理量纲恒等（未注册机器人同理）",
                    name, dims,
                )
        specs[str(name)] = {
            "norm01_dims": torch.tensor(dims, dtype=torch.long),
            "q_low": q_low,
            "q_high": q_high,
            "phys_mean": phys_mean,
            "phys_std": phys_std,
            "reader_z_mean": _z_vec("reader_z_mean"),
            "reader_z_std": _z_vec("reader_z_std"),
        }
    return GripperHandNormPolicy(specs)
