# ===- dispatcher.py -----------------------------------------------------------
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
#
# ===---------------------------------------------------------------------------
#
# Template dispatcher module.
#
# A dispatcher is a function that can contains multiple behaviors.
# Its specific behavior is can be controlled by DispatchContext.
#
# DispatchContext is used in two ways, usually via different implementation
# of the DispatchContext base class.
#
# - During search, we can use it to pass the current proposal from tuner.
# - During evaluation, we can use it to set pick the best policy.
#
# ===---------------------------------------------------------------------------


class DispatchContext(object):
    """
    Base class of dispatch context.

    DispatchContext enables the target and workload
    specific dispatch mechanism for templates.
    """

    current = None
    # a set to prevent print duplicated message
    warning_messages = set()

    def __init__(self):
        self._old_ctx = DispatchContext.current

    def query(self, target, workload):
        """
        Query the context to get the specific config for a template.

        If cannot find the result inside this context, this function will query it
        from the upper contexts.
        """
        ret = self._query_inside(target, workload)
        if ret is None:
            ret = self._old_ctx.query(target, workload)
        return ret

    def update(self, target, workload, cfg):
        """Update context with a specific config."""
        raise NotImplementedError()

    def _query_inside(self, target, workload):
        """
        Query the context to get the specific config for a template.

        This function only query config inside this context.
        """
        raise NotImplementedError()

    def __enter__(self):
        self._old_ctx = DispatchContext.current
        DispatchContext.current = self
        return self

    def __exit__(self, ptype, value, trace):
        DispatchContext.current = self._old_ctx


class ApplyConfig(DispatchContext):
    """Apply a deterministic config entity for all queries."""

    def __init__(self, config):
        super(ApplyConfig, self).__init__()
        self._config = config
        self.workload = None

    def _query_inside(self, target, workload):
        """Override query"""
        self.workload = workload
        return self._config

    def update(self, target, workload, cfg):
        """Override update"""
        self.workload = workload
        self._config = cfg
