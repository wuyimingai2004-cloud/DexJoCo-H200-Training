"""FP8 (e4m3) Linear for the uni trunk / video decoder blocks.

训练模式（默认）：包装 bf16 master weight，``self.weight`` / ``self.bias`` 复用
原 ``nn.Linear`` 的 Parameter 对象（state_dict 键与 ``nn.Linear`` 完全一致，
checkpoint 双向兼容），前向/反向走自定义 autograd.Function 的三个 GEMM：

  fprop：激活 per-token 动态量化（e4m3，rowwise scale_a (M,1)）× 权重按
         per-output-channel 量化（e4m3 [N,K]，scale_b (1,N)）→
         ``torch._scaled_mm`` → bf16 输出（bias 走 _scaled_mm 的 epilogue，
         不支持时退回 GEMM 外 add）；
  dgrad：grad_output 同粒度量化 × weight.t()（B = [N,K] 列主序 = cuBLASLt TN；
         其收缩维是 N、输出列是 K，故权重需**另一份按 per-input-channel 量化
         的副本** w8_t [K,N]，scale_b_t (1,K)；复用 fprop 的 (1,N) 在 N≠K 时
         报 "Invalid scaling configuration"、N=K 时语义错误）；
  wgrad：首轮保持 bf16 matmul（``MOTUS_FP8_WGRAD=1`` 才切 fp8，A=e5m2 /
         B=e4m3——torch 2.7.1 硬编码不支持 e5m2×e5m2）。

推理模式 ``weight_only=True``（env ``MOTUS_W8A16=1``）：权重以 fp8 +
per-output-channel scale 存储（原 bf16 权重释放），前向 dequant 到 bf16 后走
普通 ``F.linear``。**此模式下 master 权重已被 fp8 取代，只能用于推理/eval，
不得用于训练**（fp8 权重 requires_grad=False，且不含训练所需精度）。

构造不消耗全局 RNG（20261012）：``nn.Linear.__init__`` 的 ``reset_parameters()``
会推进入参的全局 RNG，而 ``from_linear`` 两条路径随后都用原 ``nn.Linear`` 的
Parameter 顶掉这份随机权重（非 weight_only 直接复用对象、weight_only 由
``_install_weight_only`` 重写）——抽样纯属浪费，留着还会让 ``MOTUS_FP8_LINEAR=1``
这个**开关本身平移全局 RNG 流**：转换发生在 ``UniWorldModel.__init__`` 末尾，初始
权重不受影响（实测开/关两跑 state_dict sha1 相同），但此后每个 step 采样的噪声/
timestep/σ jitter 全部错位，A/B 的差异里混进"换了一条随机轨迹"。故 ``__init__`` 内
包一层 CPU/CUDA RNG 状态保存-恢复（``torch.random.get_rng_state`` +
``torch.cuda.get_rng_state_all``，CUDA 未初始化则只保 CPU），构造 ``FP8Linear`` 对外部
RNG 流零影响。推论：直接 ``FP8Linear(...)``（不走 ``from_linear``）连续构造多个会拿到
**同一份**初始权重——需要不同初始化的调用方自己先定种/重采样。

env：
  MOTUS_FP8_GRANULARITY   rowwise（默认）| tensorwise
  MOTUS_FP8_QUANT         triton（默认）| torch：rowwise 量化实现（见下"量化成本"）
  MOTUS_FP8_WEIGHT_CACHE  1 打开量化权重缓存（**默认关**：生产
                          gradient_accumulation_steps=1、每 step 一次前向时
                          缓存零收益还要付守卫开销；推理/eval 多次前向可显式开）
  MOTUS_FP8_WGRAD         1 打开 fp8 wgrad（默认 bf16）
  MOTUS_W8A16             1 转换走 weight_only 模式（由 convert_to_fp8_linear）
  MOTUS_FP8_LINEAR        1 打开转换（由 models/world_model.py 读取）

量化成本（20261012 二轮，H200 首测 rowwise 端到端净变慢后重做）：torch 逐元素
链（abs→amax→除法→clamp→cast）走 5 个 pass、中间还落两张 fp32 全尺寸临时张量
（go 58840x14336 时 2×3.4 GB），实测 25 B/输入元素；rowwise 量化改由 triton 单
kernel 融合（一行一个 program：首轮扫 amax、次轮命中 L1 再 scale+clamp+cast，
只额外写 1 张 fp8），DRAM 流量降到 3 B/元素，实测本机 8.6~9.5×、与 torch.compile
（Inductor 同样融成单 kernel）持平，且不引入 dynamo 依赖。rowwise 的 scale 在
两种实现里都是**先做 fp32 除法再 clamp**（此前 torch 路径因 bf16 张量除以
Python 标量而把 scale 落到 bf16 精度，已修正为 fp32）。两种实现的 fp8 输出逐位
一致，唯一次正规区（|q| < 2^-6）平局舍入方向不同：实测 195M 元素里 e4m3 有 7 个
（3.6e-8）差 1 个次正规 ULP（2^-9），e5m2 为 0。

粒度探测：首次前向对当前设备试一次小矩阵 ``_scaled_mm``，rowwise 报
RuntimeError 则 warn 一次并降级 tensorwise（结果按设备缓存）。本机 sm120 实测
报 "Rowwise scaling is not currently supported on your device"，sm90 待集群验证。

量化权重缓存（默认关，``MOTUS_FP8_WEIGHT_CACHE=1`` 开）：存两份布局——fprop 的
per-n ``w8`` [N,K] 与 dgrad 的 per-k ``w8_t`` [K,N]（tensorwise 下两者数值相同，
仅布局不同）。失效判据除 ``weight._version`` / data_ptr 外还有**内容校验和**：
PyTorch 的 ``weight.data`` 路径写入不推进该 Parameter 的 version counter（实测
``p.data.add_(0.01)`` 后 ``p._version`` 不变），而 DeepSpeed ZeRO-1/2 的 bf16
参数回写正是 ``.data`` 路径（stage_1_and_2.py:2123
``bit16_partitions[pid].data.copy_(fp32_partition.data)`` + :725/:730
``param.data = <flat buffer view>.data``），只靠 _version 会让缓存永远停在旧权重
上。校验和为每张权重 8 个等距 512 元素块的原始位模式精确求和（与权重大小无关，
实测 12 µs/张、生产 248 张合计 5.8 ms），每"前向趟"比较一次（根模块 forward
pre-hook 划定趟边界；无 hook 时逐次调用比较）。
"""
import contextlib
import logging
import os
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - triton 缺失时全程走 torch 实现
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
_FP8_DTYPES = (E4M3, E5M2)
_FP8_MAX = {E4M3: torch.finfo(E4M3).max, E5M2: torch.finfo(E5M2).max}
_SCALE_MIN = 1e-12
_MIN_FP8_DIM = 16
_GUARD_BLOCKS = 8
_GUARD_BLOCK = 512
_QUANT_BLOCK_K = 2048
_QUANT_NUM_WARPS = 8

_ROWWISE_SUPPORT = {}
_TRITON_ENABLED = _TRITON_AVAILABLE
_BIAS_FUSION_OK = None


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true")


def requested_granularity() -> str:
    """MOTUS_FP8_GRANULARITY 的解析值（不探测设备）。"""
    gran = os.environ.get("MOTUS_FP8_GRANULARITY", "rowwise").strip().lower()
    if gran not in ("rowwise", "tensorwise"):
        logger.warning("MOTUS_FP8_GRANULARITY=%r 未知，回退 rowwise", gran)
        return "rowwise"
    return gran


def _device_key(device: torch.device):
    index = device.index
    if device.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return device.type, index


def _rowwise_supported(device: torch.device) -> bool:
    key = _device_key(device)
    if key in _ROWWISE_SUPPORT:
        return _ROWWISE_SUPPORT[key]
    a = torch.ones(_MIN_FP8_DIM, _MIN_FP8_DIM, device=device, dtype=E4M3)
    b = torch.ones(_MIN_FP8_DIM, _MIN_FP8_DIM, device=device, dtype=E4M3)
    sa = torch.ones(_MIN_FP8_DIM, 1, device=device)
    sb = torch.ones(1, _MIN_FP8_DIM, device=device)
    try:
        torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                         out_dtype=torch.bfloat16)
        ok, reason = True, ""
    except RuntimeError as exc:
        ok, reason = False, str(exc)
    _ROWWISE_SUPPORT[key] = ok
    if not ok:
        logger.warning(
            "FP8 rowwise scaling unsupported on %s (%s); "
            "falling back to tensorwise", device, reason)
    return ok


def resolve_granularity(device: torch.device) -> str:
    """当前设备实际可用的缩放粒度（rowwise 探测失败则降级 tensorwise）。"""
    gran = requested_granularity()
    if gran == "rowwise" and device.type == "cuda" \
            and not _rowwise_supported(device):
        return "tensorwise"
    return gran


def _quantize_rowwise(x2d: torch.Tensor, dtype=E4M3):
    fast = _quantize_rowwise_triton(x2d, dtype)
    if fast is not None:
        return fast
    max_val = _FP8_MAX[dtype]
    scale = (x2d.abs().amax(dim=1, keepdim=True).float() / max_val).clamp(
        min=_SCALE_MIN)
    return (x2d / scale).clamp(-max_val, max_val).to(dtype), scale


def _quantize_tensorwise(x2d: torch.Tensor, dtype=E4M3):
    max_val = _FP8_MAX[dtype]
    scale = (x2d.abs().amax().float() / max_val).clamp(min=_SCALE_MIN)
    return (x2d / scale).clamp(-max_val, max_val).to(dtype), scale


if _TRITON_AVAILABLE:

    @triton.jit
    def _rowwise_quant_kernel(x_ptr, out_ptr, scale_ptr, K, stride_xm, stride_om,
                              max_val, scale_min, BLOCK_K: tl.constexpr):
        """一行一个 program 的融合 per-row 量化：首轮扫 amax，次轮 scale+cast。

        第二次读同一行的数据命中 L1（行 ≤ 28 KB），故 DRAM 流量 = 读 1 次 +
        写 1 次（3 B/元素，bf16→fp8），比"归约 kernel + 逐元素 kernel"分离实现
        （5 B/元素）再省 1.67×；不依赖寄存器驻留整行，K 无上限。
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            m = k0 + offs < K
            v = tl.load(x_ptr + row * stride_xm + k0 + offs, mask=m,
                        other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(v))
        scale = tl.maximum(tl.max(acc, axis=0) / max_val, scale_min)
        for k0 in range(0, K, BLOCK_K):
            m = k0 + offs < K
            v = tl.load(x_ptr + row * stride_xm + k0 + offs, mask=m,
                        other=0.0).to(tl.float32)
            q = tl.minimum(tl.maximum(v / scale, -max_val), max_val)
            tl.store(out_ptr + row * stride_om + k0 + offs,
                     q.to(out_ptr.dtype.element_ty), mask=m)
        tl.store(scale_ptr + row, scale)


def _env_quant_impl() -> str:
    impl = os.environ.get("MOTUS_FP8_QUANT", "triton").strip().lower()
    if impl not in ("triton", "torch"):
        logger.warning("MOTUS_FP8_QUANT=%r 未知，回退 triton", impl)
        return "triton"
    return impl


def _quantize_rowwise_triton(x2d: torch.Tensor, dtype=E4M3):
    """融合版 per-row 量化；不适用（无 triton/非行主序/被关闭）时返回 None。

    与 torch 实现（``abs→amax→fp32 除法→clamp→cast``）逐位一致：amax 是精确
    取最大（与顺序无关），scale 的 fp32 除法与 clamp、fp8 cast 的舍入都用同一
    IEEE 语义；torch 的 fp32→fp8 cast 越界会出 NaN，故 clamp 必需。
    """
    global _TRITON_ENABLED
    if not _TRITON_ENABLED or _env_quant_impl() != "triton":
        return None
    if x2d.dim() != 2 or x2d.stride(-1) != 1 or x2d.numel() == 0:
        return None
    try:
        rows, cols = x2d.shape
        block_k = min(_QUANT_BLOCK_K, triton.next_power_of_2(cols))
        scale = torch.empty((rows, 1), device=x2d.device, dtype=torch.float32)
        out = torch.empty((rows, cols), device=x2d.device, dtype=dtype)
        _rowwise_quant_kernel[(rows,)](
            x2d, out, scale, cols, x2d.stride(0), out.stride(0),
            _FP8_MAX[dtype], _SCALE_MIN, BLOCK_K=block_k,
            num_warps=_QUANT_NUM_WARPS)
        return out, scale
    except Exception as exc:  # noqa: BLE001 — 编译/启动失败即永久回退 torch
        _TRITON_ENABLED = False
        logger.warning("FP8 triton 量化不可用（%s），回退 torch 实现", exc)
        return None


def _quantize_activation(x2d: torch.Tensor, granularity: str, dtype=E4M3):
    """[M, K] → (fp8, scale_a)：rowwise 为 (M, 1)，tensorwise 为 0 维标量。"""
    if granularity == "rowwise":
        return _quantize_rowwise(x2d, dtype)
    return _quantize_tensorwise(x2d, dtype)


def _quantize_weight_output_channel(w2d: torch.Tensor, granularity: str,
                                    dtype=E4M3):
    """fprop 用：[N, K] → (fp8 [N, K], scale_b (1, N) 或标量)。

    fprop 的 B = w8.t()（[K, N] 列主序）收缩维是 K、输出列是 N，per-output-
    channel（per-n）缩放恰好落在输出列上。
    """
    if granularity == "rowwise":
        q, scale = _quantize_rowwise(w2d, dtype)
        return q, scale.t().contiguous()
    return _quantize_tensorwise(w2d, dtype)


def _quantize_weight_input_channel(w2d: torch.Tensor, granularity: str,
                                   dtype=E4M3):
    """dgrad 用：[N, K] → ([K, N] 行主序 fp8, scale_b (1, K) 或标量)。

    dgrad 是 grad_x = grad_out @ w，_scaled_mm 的 B 为 [N, K] 列主序（= 本张量
    的 .t()）——**收缩维是 N、输出列是 K**，故 rowwise 的 scale_b 必须是 (1, K)，
    即按 ``w.t()`` 的每行（per-k）量化。沿用 fprop 的 (1, N) 在 N≠K 时直接报
    "Invalid scaling configuration"，N=K 时虽不报错但等于把 per-n 的 scale 错
    贴到输出列 k 上。tensorwise 下两者数值相同（标量 scale）。
    """
    w_t = w2d.t().contiguous()
    if granularity == "rowwise":
        q, scale = _quantize_rowwise(w_t, dtype)
        return q, scale.t().contiguous()
    return _quantize_tensorwise(w_t, dtype)


def _wgrad_fp8(go2d: torch.Tensor, x2d: torch.Tensor, granularity: str):
    """grad_w = grad_out.T @ x：A=e5m2（梯度侧动态范围更宽）/ B=e4m3。"""
    go_t = go2d.t().contiguous()
    x_t = x2d.t().contiguous()
    go8, scale_a = _quantize_activation(go_t, granularity, E5M2)
    x8, scale_b = _quantize_activation(x_t, granularity, E4M3)
    if granularity == "rowwise":
        scale_b = scale_b.t().contiguous()
    return torch._scaled_mm(go8, x8.t(), scale_a=scale_a, scale_b=scale_b,
                            out_dtype=torch.bfloat16)


def _scaled_mm_bias(x8, w8, scale_a, scale_b, bias):
    """fprop GEMM：bias 走 _scaled_mm 的 epilogue（省一次全张量读写）；不支持时
    退回 GEMM 外 add（首次失败即永久退回，_BIAS_FUSION_OK 三态缓存）。"""
    global _BIAS_FUSION_OK
    if _BIAS_FUSION_OK is not False and bias is not None \
            and bias.dtype == torch.bfloat16 and bias.dim() == 1:
        try:
            out = torch._scaled_mm(x8, w8.t(), scale_a=scale_a,
                                   scale_b=scale_b, bias=bias,
                                   out_dtype=torch.bfloat16)
            _BIAS_FUSION_OK = True
            return out
        except RuntimeError as exc:
            _BIAS_FUSION_OK = False
            logger.warning(
                "FP8 _scaled_mm(bias=) 不可用（%s），bias 退回 GEMM 外 add",
                exc)
    out = torch._scaled_mm(x8, w8.t(), scale_a=scale_a, scale_b=scale_b,
                           out_dtype=torch.bfloat16)
    return out if bias is None else out + bias


class _FP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, w8, scale_b, w8_t, scale_b_t,
                granularity):
        with torch.autocast(device_type="cuda", enabled=False):
            ctx.in_dtype = x.dtype
            if x.dtype != weight.dtype:
                x = x.to(weight.dtype)
            x2d = x.reshape(-1, x.shape[-1])
            x8, scale_a = _quantize_activation(x2d, granularity)
            out = _scaled_mm_bias(x8, w8, scale_a, scale_b, bias)
        ctx.save_for_backward(x, weight, w8_t, scale_b_t)
        ctx.granularity = granularity
        ctx.bias_dtype = None if bias is None else bias.dtype
        return out.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, w8_t, scale_b_t = ctx.saved_tensors
        granularity = ctx.granularity
        go2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        x2d = x.reshape(-1, x.shape[-1])
        with torch.autocast(device_type="cuda", enabled=False):
            go8, scale_go = _quantize_activation(go2d, granularity)
            grad_x = torch._scaled_mm(go8, w8_t.t(), scale_a=scale_go,
                                      scale_b=scale_b_t,
                                      out_dtype=torch.bfloat16)
            if _env_on("MOTUS_FP8_WGRAD"):
                grad_w = _wgrad_fp8(go2d, x2d, granularity)
            else:
                grad_w = torch.matmul(go2d.to(torch.bfloat16).t(),
                                      x2d.to(torch.bfloat16))
            grad_b = None
            if ctx.bias_dtype is not None:
                grad_b = go2d.sum(dim=0).to(ctx.bias_dtype)
        return (grad_x.reshape(x.shape).to(ctx.in_dtype),
                grad_w.to(weight.dtype), grad_b, None, None, None, None, None)


def _weight_checksum(weight: torch.Tensor) -> torch.Tensor:
    """权重原始位模式的采样校验和（int64 精确求和，8 个等距 512 元素块）。

    读写量与权重大小无关：覆盖整张权重的 8 个位置，用于判定缓存的量化权重
    是否仍与 master weight 一致（见模块文档的缓存失效说明）。
    """
    flat = weight.detach().reshape(-1)
    block = min(_GUARD_BLOCK, flat.numel())
    step = max(flat.numel() // _GUARD_BLOCKS, block)
    sample = flat.unfold(0, block, step)[:_GUARD_BLOCKS]
    return sample.to(torch.bfloat16).view(torch.int16).sum(dtype=torch.int64)


class _WeightCache:
    __slots__ = ("w8", "scale_b", "w8_t", "scale_b_t", "granularity", "version",
                 "ptr", "dtype", "device", "checksum")

    def __init__(self, weight, w8, scale_b, w8_t, scale_b_t, granularity):
        self.w8 = w8
        self.scale_b = scale_b
        self.w8_t = w8_t
        self.scale_b_t = scale_b_t
        self.granularity = granularity
        self.version = weight._version
        self.ptr = weight.data_ptr()
        self.dtype = weight.dtype
        self.device = weight.device
        self.checksum = _weight_checksum(weight)

    def matches(self, weight) -> bool:
        return (self.version == weight._version and self.ptr == weight.data_ptr()
                and self.dtype == weight.dtype and self.device == weight.device)


class _WeightCacheRegistry:
    """量化权重缓存的内容守卫（每"前向趟"至多一次校验和比较）。

    ``convert_to_fp8_linear`` 在根模块上注册 forward pre-hook 划定趟边界（本
    仓库 UniWorldModel 经 ``self.world_model(...)`` 调用，hook 必触发）；没有该
    hook 时退回逐次调用校验——正确但每次调用一次同步，仅测试/独立用法会走到。
    """

    def __init__(self):
        self._modules = []
        self._epoch = 0
        self._validated_epoch = -1
        self._hooked = False

    def register(self, module) -> None:
        for ref in self._modules:
            if ref() is module:
                return
        self._modules.append(weakref.ref(module))

    def begin_pass(self, *_args) -> None:
        self._epoch += 1
        self._hooked = True

    def validate(self) -> None:
        if self._hooked and self._validated_epoch == self._epoch:
            return
        self._validated_epoch = self._epoch
        entries = []
        for ref in list(self._modules):
            module = ref()
            if module is None:
                self._modules.remove(ref)
            elif module._weight_cache is not None:
                entries.append(module)
        if not entries:
            return
        checks = torch.stack(
            [_weight_checksum(m.weight) for m in entries]).cpu().tolist()
        for module, value in zip(entries, checks):
            cache = module._weight_cache
            if cache is not None and cache.checksum != value:
                module._weight_cache = None


_CACHE_REGISTRY = _WeightCacheRegistry()


@contextlib.contextmanager
def _preserved_rng_state():
    """构造期间保存并恢复全局 RNG 状态（见模块文档"构造不消耗 RNG"一节）。

    只保存**已初始化**的 CUDA 状态：未初始化时构造也碰不到 CUDA RNG，且此处不该
    为了读一份状态而把 CUDA context 拉起来（导入期/CPU-only 测试要求零副作用）。
    """
    cpu_state = torch.random.get_rng_state()
    cuda_states = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_initialized() else None)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class FP8Linear(nn.Linear):
    """Drop-in ``nn.Linear`` replacement（见模块文档；state_dict 键不变）。"""

    def __init__(self, in_features, out_features, bias=True, weight_only=False):
        # 构造不消耗全局 RNG（见模块文档）：reset_parameters() 本来会推进全局 RNG，
        # 而 from_linear 随后就用原 Parameter 顶掉这份随机权重——不平移的话
        # MOTUS_FP8_LINEAR=1 这个开关本身就会改变初始权重与噪声流，污染 A/B。
        with _preserved_rng_state():
            super().__init__(in_features, out_features, bias)
        self._weight_only = bool(weight_only)
        self._weight_cache = None
        self._fallback_warned = False
        if self._weight_only:
            self._install_weight_only(self.weight.detach())

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, weight_only: bool = False):
        module = cls(linear.in_features, linear.out_features,
                     bias=linear.bias is not None)
        if not weight_only:
            module.weight = linear.weight
            module.bias = linear.bias
        else:
            module._weight_only = True
            module._install_weight_only(linear.weight.detach())
            if linear.bias is not None:
                linear.bias.requires_grad_(False)
                module.bias = linear.bias
        return module

    def _install_weight_only(self, weight: torch.Tensor) -> None:
        max_val = _FP8_MAX[E4M3]
        w = weight.detach().to(torch.float32)
        scale = (w.abs().amax(dim=1, keepdim=True) / max_val).clamp(
            min=_SCALE_MIN)
        self.weight = nn.Parameter(
            (w / scale).clamp(-max_val, max_val).to(E4M3), requires_grad=False)
        self.register_buffer("weight_scale", scale)
        self._weight_cache = None

    def _apply(self, fn, recurse=True):
        """weight_only 模式的 fp8 存储/scale 不做 dtype 转换（只随设备搬移）。

        ``nn.Module.to(dtype=...)`` 会把浮点 Parameter 一律 cast（float8 也算
        浮点），UBT_WM 构造后即 ``.to(dtype=self.dtype)``（models/ubt_wm.py:649），
        不做保护会让 W8A16 的 fp8 权重被悄悄 cast 回 bf16。
        """
        if not self._weight_only or self.weight.dtype not in _FP8_DTYPES:
            return super()._apply(fn, recurse)
        fp8_dtype = self.weight.dtype
        weight = self._parameters.pop("weight")
        self._parameters["weight"] = nn.Parameter(weight.data.view(torch.int8),
                                                  requires_grad=False)
        scale = self._buffers.get("weight_scale")
        if scale is not None:
            self._buffers["weight_scale"] = scale.view(torch.int32)
        try:
            super()._apply(fn, recurse)
        finally:
            self._parameters["weight"] = nn.Parameter(
                self._parameters.pop("weight").data.view(fp8_dtype),
                requires_grad=False)
            if scale is not None:
                self._buffers["weight_scale"] = self._buffers[
                    "weight_scale"].view(torch.float32)
        return self

    def _dequantized_weight(self) -> torch.Tensor:
        return (self.weight.to(torch.float32)
                * self.get_buffer("weight_scale")).to(torch.bfloat16)

    def _quantized_weight(self, device: torch.device):
        granularity = resolve_granularity(device)
        if not _env_on("MOTUS_FP8_WEIGHT_CACHE"):
            self._weight_cache = None
            return (*self._quantize_weight(granularity), granularity)
        _CACHE_REGISTRY.validate()
        cache = self._weight_cache
        if cache is not None and cache.matches(self.weight):
            return (cache.w8, cache.scale_b, cache.w8_t, cache.scale_b_t,
                    cache.granularity)
        w8, scale_b, w8_t, scale_b_t = self._quantize_weight(granularity)
        self._weight_cache = _WeightCache(self.weight, w8, scale_b, w8_t,
                                          scale_b_t, granularity)
        _CACHE_REGISTRY.register(self)
        return w8, scale_b, w8_t, scale_b_t, granularity

    def _quantize_weight(self, granularity: str):
        weight = self.weight.detach()
        w8, scale_b = _quantize_weight_output_channel(weight, granularity,
                                                      E4M3)
        w8_t, scale_b_t = _quantize_weight_input_channel(weight, granularity,
                                                         E4M3)
        return w8, scale_b, w8_t, scale_b_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_only:
            return F.linear(x, self._dequantized_weight(), self.bias)
        if self.in_features % _MIN_FP8_DIM or self.out_features % _MIN_FP8_DIM:
            if not self._fallback_warned:
                self._fallback_warned = True
                logger.warning(
                    "FP8 Linear %dx%d: 维度不是 %d 的倍数，退回 bf16 F.linear",
                    self.out_features, self.in_features, _MIN_FP8_DIM)
            return F.linear(x, self.weight, self.bias)
        if not x.is_cuda:
            return F.linear(x, self.weight, self.bias)
        w8, scale_b, w8_t, scale_b_t, granularity = self._quantized_weight(
            x.device)
        return _FP8LinearFunction.apply(x, self.weight, self.bias, w8, scale_b,
                                        w8_t, scale_b_t, granularity)


def _convert_tree(module, prefix, *, weight_only, exclude, memo) -> int:
    count = 0
    # 用 _modules 直接遍历而非 named_children：后者对同一子模块只产出一次，
    # 会让"同一 Linear 挂两条路径"的第二次引用留在树上未被替换。
    for name, child in list(module._modules.items()):
        if child is None:
            continue
        full = f"{prefix}{name}"
        if isinstance(child, FP8Linear):
            continue
        if isinstance(child, nn.Linear):
            if exclude is not None and exclude(full, child):
                continue
            replaced = memo.get(id(child))
            if replaced is None:
                replaced = FP8Linear.from_linear(child, weight_only=weight_only)
                memo[id(child)] = replaced
                count += 1
            setattr(module, name, replaced)
        else:
            count += _convert_tree(child, full + ".", weight_only=weight_only,
                                   exclude=exclude, memo=memo)
    return count


def convert_to_fp8_linear(module, *, weight_only=False, exclude=None) -> int:
    """递归把 module 树里的 ``nn.Linear`` 就地替换为 ``FP8Linear``。

    exclude(name, module) -> bool 为 True 时跳过（name 为相对 module 的
    点分路径）。返回替换的 Linear 数量（共享同一 Linear 的多条路径复用同一个
    FP8Linear，只计一次）。非 weight_only 复用原 Parameter 对象，训练可继续；
    weight_only=True 时 master 权重被 fp8 + per-channel scale 取代（仅推理）。
    """
    memo = {}
    count = _convert_tree(module, "", weight_only=weight_only, exclude=exclude,
                          memo=memo)
    if any(not m._weight_only for m in memo.values()):
        module.register_forward_pre_hook(_CACHE_REGISTRY.begin_pass)
    return count
