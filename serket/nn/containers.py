from __future__ import annotations

from dataclasses import field
from types import MappingProxyType
from typing import Any, Callable

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc


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

        Example:
            >>> model = Sequential([sk.nn.Linear(1,1), sk.nn.Dropout(0), sk.nn.Linear(1,1)])
            >>> model
            Sequential
                ├── Linear_0=Linear
                │   ├── weight=f32[1,1]
                │   ├── bias=f32[1]
                │   ├*─ in_features=1
                │   └*─ out_features=1
                ├── Dropout_1=Dropout
                │   ├── p=0
                │   └── eval=None
                └── Linear_2=Linear
                    ├── weight=f32[1,1]
                    ├── bias=f32[1]
                    ├*─ in_features=1
                    └*─ out_features=1
        """
        extra_fields = dict()
        if isinstance(layers, (list, tuple)):
            for i, layer in enumerate(layers):
                field_value = field(repr=True)
                field_name = f"{(layer.__class__.__name__)}_{i}"
                object.__setattr__(field_value, "name", field_name)
                object.__setattr__(field_value, "type", type(layer))
                extra_fields[field_name] = field_value
                setattr(self, field_name, layer)

        elif isinstance(layers, dict):
            for key, layer in layers.items():
                field_value = field(repr=True)
                object.__setattr__(field_value, "name", key)
                object.__setattr__(field_value, "type", type(layer))
                extra_fields[key] = field_value
                setattr(self, key, layer)
        else:
            raise TypeError(f"layers must be list, tuple or dict, but got {type(layers)}")  # fmt: skip

        self._keys = tuple(extra_fields.keys())

        object.__setattr__(
            self,
            "__undeclared_fields__",
            MappingProxyType({**dict(self.__undeclared_fields__), **extra_fields}),
        )

    def __getitem__(self, key: str | int):
        if isinstance(key, str) and key in self._keys:
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
