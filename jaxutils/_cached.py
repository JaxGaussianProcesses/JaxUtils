# Copyright 2023 The JaxGaussianProcesses Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This non-public module defines the caching utilities for the Module class's static properties."""

from __future__ import annotations
from typing import Callable


class _cached_static_property:
    """Decorator to cache result of static immutable properties of a PyTree.

    Example:

        This example shows us caching the result of sqauring a static float attribute of a Module.

        >>> import jaxutils as ju
        >>>
        >>> class MyModule(ju.Module):
        >>>     static_attribute: float = ju.static()

        >>>     @_cached_static_property
        >>>     def static_property(self):
        >>>         return self.static_attribute ** 2

    Note:
        The decorated property must *NOT* contain any dynamic attributes / PyTree leaves,
        i.e., any attributes referenced in the property must be marked as static.

        For example, the following will break durin tracing since `self.dynamic_attribute` is not static:

        >>> import jaxutils as ju
        >>>
        >>> class MyModule(ju.Module):
        >>>     static_attribute: float = ju.static()
        >>>     dynamic_attribute: float = ju.param(ju.Identity)
        >>>
        >>>     @_cached_static_property
        >>>     def static_property(self):
        >>>         return self.static_attribute ** 2 + self.dynamic_attribute
    """

    def __init__(self, static_property: Callable):
        """Here we store the name of the property and the function itself."""
        self.name = static_property.__name__
        self.func = static_property

    def __get__(self, instance, owner):
        """Here we cache the result of the property function, by overwriting the attribute with the result."""
        attr = self.func(instance)
        object.__setattr__(instance, self.name, attr)
        return attr
