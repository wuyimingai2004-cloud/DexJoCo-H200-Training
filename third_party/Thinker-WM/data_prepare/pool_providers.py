"""repo 级 pool provider 注册表：YAML `pool:` spec → pool 工厂（callable）。

索引选择策略归训练代码所有（本包）；motus_data 库只提供 SupportsSamplePool
注入点（open_training_data/IndexedTrainingSource 的 sample_pool 接受
callable，形式为 factory(repository=..., seed=...) -> pool）。

spec 形式：{"type": <provider 名>, 其余键作 options}。新增自定义选择方式：
写 Pool 类（实现 motus_data.SupportsSamplePool）+ register_pool_builder 注册，
YAML `pool: {type: <名>, ...}` 即可，管线零改动。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping


class PoolSpecError(ValueError):
    """Raised when a YAML pool spec cannot be resolved to a provider."""


_POOL_BUILDERS: dict[str, Callable[..., Any]] = {}
_POOL_KNOWN_OPTIONS: dict[str, tuple[str, ...]] = {}


def register_pool_builder(
    name: str,
    builder: Callable[..., Any],
    *,
    known_options: tuple[str, ...] = (),
) -> None:
    """注册 provider。builder 签名：f(repository, *, seed, **options) -> pool。"""
    key = str(name).strip()
    if not key:
        raise PoolSpecError("pool builder name must be non-empty")
    if key in _POOL_BUILDERS:
        raise PoolSpecError(f"pool builder already registered: {key!r}")
    if not callable(builder):
        raise PoolSpecError(f"pool builder must be callable: {name!r}")
    _POOL_BUILDERS[key] = builder
    _POOL_KNOWN_OPTIONS[key] = tuple(str(option) for option in known_options)


def pool_builders() -> tuple[str, ...]:
    return tuple(sorted(_POOL_BUILDERS))


def pool_factory_from_spec(spec: Mapping[str, Any]) -> Callable[..., Any]:
    """把 YAML pool spec 解析为 motus_data 注入点接受的 pool 工厂 callable。"""
    if not isinstance(spec, Mapping):
        raise PoolSpecError(
            f"pool spec must be a mapping with a 'type' key, got {type(spec).__name__}"
        )
    kind = spec.get("type")
    if not kind:
        raise PoolSpecError(
            f"pool spec missing 'type' (registered: {pool_builders()})"
        )
    kind = str(kind).strip()
    builder = _POOL_BUILDERS.get(kind)
    if builder is None:
        raise PoolSpecError(
            f"unknown pool type {kind!r} (registered: {pool_builders()})"
        )
    options = {str(key): value for key, value in spec.items() if key != "type"}
    unknown = sorted(set(options) - set(_POOL_KNOWN_OPTIONS[kind]))
    if unknown:
        raise PoolSpecError(
            f"pool type {kind!r} got unknown options {unknown} "
            f"(known: {sorted(_POOL_KNOWN_OPTIONS[kind])})"
        )

    def factory(*, repository: Any, seed: int) -> Any:
        return builder(repository, seed=seed, **options)

    return factory
