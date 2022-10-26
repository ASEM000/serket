from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.utils import _create_fields_from_container, _create_fields_from_mapping


@pytc.treeclass
class Lambda:
    func: Callable[[Any], jnp.ndarray] = pytc.nondiff_field()

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.func(x)


@pytc.treeclass
class Sequential:
    def __init__(self, layers: list[Any] | tuple[Any] | dict[[str], Any]):
        """Register layers in a sequential container.

        Args:
            layers: list of layers or dict of layers
        """
        if isinstance(layers, (list, tuple)):
            field_mapping = _create_fields_from_container(layers)

        elif isinstance(layers, dict):
            field_mapping = _create_fields_from_mapping(layers)
            layers = layers.values()

        else:
            raise TypeError(
                f"layers must be list, tuple or dict, but got {type(layers)}"
            )

        self._keys = tuple(field_mapping.keys())

        for name, layer in zip(self._keys, layers):
            setattr(self, name, layer)

        field_mapping = {**dict(self.__treeclass_fields__), **field_mapping}
        setattr(self, "__treeclass_fields__", field_mapping)

    def __getitem__(self, key: str | int | slice):
        if isinstance(key, str):
            return getattr(self, key)
        elif isinstance(key, int):
            return getattr(self, self._keys[key])
        elif isinstance(key, slice):
            return Sequential({name: getattr(self, name) for name in self._keys[key]})
        raise TypeError(f"key must be str or int, but got {type(key)}")

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter((getattr(self, name) for name in self._keys))

    def __reversed__(self):
        return (getattr(self, name) for name in reversed(self._keys))

    def items(self):
        return {name: getattr(self, name) for name in self._keys}

    def __contains__(self, item):
        return item in self.items()

    def __call__(
        self, x: jnp.ndarray, *, key: jr.PRNGKey | None = jr.PRNGKey(0)
    ) -> jnp.ndarray:
        keys = (
            jr.split(key, len(self.items()))
            if key is not None
            else [None] * len(self.items())
        )
        for ki, layer in zip(keys, self.items().values()):
            x = layer(x, key=ki)
        return x
