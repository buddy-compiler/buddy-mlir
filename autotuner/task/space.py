# ===- space.py -------------------------------------------------------------
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
# Template configuration space.
#
# Each template function can be parameterized by a ConfigSpace.
# The space is declared when we invoke the template function with ConfigSpace.
# During evaluation, we pass in a ConfigEntity, which contains a specific
# entity in the space. This entity contains deterministic parameters.
#
# ===---------------------------------------------------------------------------

import copy
import functools
import math
import re
from collections import namedtuple, OrderedDict
from random import randrange
import numpy as np


class InstantiationError(ValueError):
    """Actively detected error in instantiating a template with a config,
    raised by cfg.raise_error
    e.g. too many unrolling, too many threads in a block
    """


class TransformSpace(object):
    """Base class for transform space
    TransformSpace is the node in the computation graph of axes
    So all the combinations of the parameters of these op form our search space.

    Naming convention:
    We call the set of all possible values as XXXSpace. (XXX can be Split, Reorder, Config ...)
    We call a specific entity in a space as XXXEntity.
    """

    def __init__(self):
        self.entities = []

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):
        """Get an entity of the space by index"""
        return self.entities[index]


# TODO: helper function
def get_factors(n):
    """return all factors of an integer"""
    step = 2 if n % 2 else 1
    ret = list(
        set(
            functools.reduce(
                list.__add__,
                (
                    [i, n // i]
                    for i in range(1, int(math.sqrt(n)) + 1, step)
                    if n % i == 0
                ),
            )
        )
    )
    ret.sort()
    return ret


def pass_to_command(pass_name: str, pass_params: dict):
    """example
    -lower-gemmini="dim=4 acc_t=f32 elem_t=f32"
    """
    params = ""
    if len(pass_params) > 0:
        for k, v in pass_params.items():
            params += k + "=" + str(v) + " "
        params = params[:-1]
        return "-" + pass_name + "=" + '"' + params + '"'

    return "-" + pass_name


class GemminiPassSpace(TransformSpace):
    """Gemmini passes space"""

    def __init__(self, passes=None):
        super(GemminiPassSpace, self).__init__()

        self.pass_space = (
            {}
        )  # example: pass_space = {"pass1": [config1, config2], "pass2": [config1, config2]}
        self.entities = []
        # tunable passes
        default_passes = {
            "llvm-request-c-wrappers": {},
            "convert-linalg-to-gemmini": {
                "acc_t": ["f32"],
            },
            "expand-strided-metadata": {},
            "convert-linalg-to-loops": {},
            "lower-affine -convert-scf-to-cf": {},
            "convert-vector-to-llvm -finalize-memref-to-llvm": {},
            "convert-arith-to-llvm": {},
            "lower-gemmini": {
                "dim": [4, 8, 16],
                "acc_t": ["f32"],
                "elem_t": ["f32"],
            },
            "convert-func-to-llvm -reconcile-unrealized-casts": {},
        }

        if passes is None:
            self.passes = default_passes
        else:
            self.passes = passes
        # generate space for all given passes
        for pass_name, params in self.passes.items():
            self.pass_space[pass_name] = []
            self._generate_pass(pass_name, params, 0, {})
        # get search space
        self._generate_space(0, {})

        # convert entity from dict to list
        self._convert_entity()

    def _generate_space(self, now, pre):
        """Generate space by DFS"""
        keys = list(self.pass_space.keys())
        values = list(self.pass_space.values())
        if now == len(keys):
            self.entities.append(pre)
        else:
            if len(values[now]) == 0:
                cur = copy.deepcopy(pre)
                cur[keys[now]] = pass_to_command(keys[now], {})
                self._generate_space(now + 1, cur)
            else:
                for value in values[now]:
                    cur = copy.deepcopy(pre)
                    cur[keys[now]] = pass_to_command(keys[now], value)
                    self._generate_space(now + 1, cur)

    def _generate_pass(self, pass_name, params, now, pre):
        """Generate space of one specify pass with given params by DFS"""
        if len(params) == 0:
            return
        keys = list(params.keys())
        values = list(params.values())
        if now == len(keys):
            self.pass_space[pass_name].append(pre)
        else:
            for value in values[now]:
                cur = copy.deepcopy(pre)
                cur[keys[now]] = value
                self._generate_pass(pass_name, params, now + 1, cur)

    def _convert_entity(self):
        entities = []
        for entity in self.entities:
            new_entity = []
            for k, v in entity.items():
                new_entity.append(v)
            entities.append(GemminiPassEntity(new_entity))
        self.entities = entities

    def __repr__(self) -> str:
        output = "Gemmini PASS Space: (\n"
        for entity in self.entities:
            output += "     " + str(entity) + "\n"
        output += "   )"
        return output


class GemminiPassEntity(object):
    def __init__(self, config) -> None:
        self.pass_config = config
        self.features = self.get_features()

    def get_features(self):
        """
        ['-matmul-optimize="vec-size=32 kernel-m=2 kernel-n=2"', '-convert-linalg-to-loops']
        """
        features = []
        pattern = r'"([^"]+)"'
        for config in self.pass_config:
            match = re.search(pattern, config)
            if match:
                params = match.group(1).split(" ")
                for param in params:
                    key, value = param.split("=")
                    # TODO: We need modify here, about how can we get features for model_based_tuner.
                    if value.isdigit():
                        features.append(int(value))
                    elif value == "f16":
                        features.append(1)
                    elif value == "f32":
                        features.append(2)
                    elif value == "f64":
                        features.append(3)
                    else:
                        raise ValueError("Can't recognize this value %s", value)
            else:
                # For pass with no params, we set a zero.
                features.append(0)

        return features

    def apply(self, cmd):
        raise NotImplementedError()

    def __repr__(self) -> str:
        return str(self.pass_config)


class GemminiSpace(TransformSpace):
    """Gemmini hardware config"""

    def __init__(self):
        super(GemminiSpace, self).__init__()

        self.entities = []
        # tunable config parameters
        self.hardware_params = {
            # spatial array size option
            "tileRows": 1,
            "tileColumns": 1,
            "meshRows": 16,
            "meshColumns": 16,
            # TODO: Scratchpad and accumulator
            # "sp_banks": 4,
            # "acc_banks": 4,
        }
        # get config space
        self._generate_space(0, {})

    def _generate_space(self, now, pre):
        """Generate space by DFS"""
        keys = list(self.hardware_params.keys())
        values = list(self.hardware_params.values())
        if now == len(keys):
            self.entities.append(GemminiEntity(pre))
        else:
            for factor in get_factors(values[now]):
                cur = copy.deepcopy(pre)
                cur[keys[now]] = factor
                self._generate_space(now + 1, cur)

    def __repr__(self) -> str:
        return (
            "Gemmini HW Space(tileRows=%d, tileColumns=%d, meshRows=%d, meshColumns=%d) len=%d"
            % (
                self.hardware_params["tileRows"],
                self.hardware_params["tileColumns"],
                self.hardware_params["meshRows"],
                self.hardware_params["meshColumns"],
                len(self.entities),
            )
        )


class GemminiEntity(object):
    def __init__(self, config) -> None:
        self.hardware_config = config
        self.features = self.get_features()

    def get_features(self):
        """
        ['-matmul-optimize="vec-size=32 kernel-m=2 kernel-n=2"', '-convert-linalg-to-loops']
        """
        features = []
        pattern = r'"([^"]+)"'
        for config in self.pass_config:
            match = re.search(pattern, config)
            if match:
                params = match.group(1).split(" ")
                for param in params:
                    key, value = param.split("=")
                    # TODO: We need modify here, about how can we get features for model_based_tuner.
                    if value.isdigit():
                        features.append(int(value))
                    elif value == "f16":
                        features.append(1)
                    elif value == "f32":
                        features.append(2)
                    elif value == "f64":
                        features.append(3)
                    else:
                        raise ValueError("Can't recognize this value %s", value)
            else:
                # For pass with no params, we set a zero.
                features.append(0)

        return features

    def apply(
        self,
    ):
        # TODO: Modify config of Gemmini in Chipyard, and recompile spike.
        pass

    def __repr__(self) -> str:
        return str(self.hardware_config)


class LinalgPassSpace(TransformSpace):
    """Gemmini passes space"""

    def __init__(self, passes=None):
        super(LinalgPassSpace, self).__init__()

        self.pass_space = (
            {}
        )  # example: pass_space = {"pass1": [config1, config2], "pass2": [config1, config2]}
        self.entities = []
        # tunable passes
        default_passes = {
            "convert-linalg-to-loops": {},
            "expand-strided-metadata": {},
            "lower-affine": {},
            "convert-scf-to-cf": {},
            "convert-vector-to-llvm": {},
            "finalize-memref-to-llvm": {},
            "convert-arith-to-llvm": {},
            "convert-func-to-llvm": {},
            "reconcile-unrealized-casts": {},
        }

        if passes is None:
            self.passes = default_passes
        else:
            self.passes = passes
        # generate space for all given passes
        for pass_name, params in self.passes.items():
            self.pass_space[pass_name] = []
            self._generate_pass(pass_name, params, 0, {})
        # get search space
        self._generate_space(0, {})

        # convert entity from dict to list
        self._convert_entity()

    def _generate_space(self, now, pre):
        """Generate space by DFS"""
        keys = list(self.pass_space.keys())
        values = list(self.pass_space.values())
        if now == len(keys):
            self.entities.append(pre)
        else:
            if len(values[now]) == 0:
                cur = copy.deepcopy(pre)
                cur[keys[now]] = pass_to_command(keys[now], {})
                self._generate_space(now + 1, cur)
            else:
                for value in values[now]:
                    cur = copy.deepcopy(pre)
                    cur[keys[now]] = pass_to_command(keys[now], value)
                    self._generate_space(now + 1, cur)

    def _generate_pass(self, pass_name, params, now, pre):
        """Generate space of one specify pass with given params by DFS"""
        if len(params) == 0:
            return
        keys = list(params.keys())
        values = list(params.values())
        if now == len(keys):
            self.pass_space[pass_name].append(pre)
        else:
            for value in values[now]:
                cur = copy.deepcopy(pre)
                cur[keys[now]] = value
                self._generate_pass(pass_name, params, now + 1, cur)

    def _convert_entity(self):
        entities = []
        for entity in self.entities:
            new_entity = []
            for k, v in entity.items():
                new_entity.append(v)
            entities.append(LinalgPassEntity(new_entity))
        self.entities = entities

    def __repr__(self) -> str:
        output = "Linalg PASS Space: (\n"
        for entity in self.entities:
            output += "     " + str(entity) + "\n"
        output += "   )"
        return output


class LinalgPassEntity(object):
    def __init__(self, config) -> None:
        self.pass_config = config
        self.features = self.get_features()

    def get_features(self):
        """
        ['-matmul-optimize="vec-size=32 kernel-m=2 kernel-n=2"', '-convert-linalg-to-loops']
        """
        features = []
        pattern = r'"([^"]+)"'
        for config in self.pass_config:
            match = re.search(pattern, config)
            if match:
                params = match.group(1).split(" ")
                for param in params:
                    key, value = param.split("=")
                    # TODO: We need modify here, about how can we get features for model_based_tuner.
                    if value.isdigit():
                        features.append(int(value))
                    elif value == "f16":
                        features.append(1)
                    elif value == "f32":
                        features.append(2)
                    elif value == "f64":
                        features.append(3)
                    else:
                        raise ValueError("Can't recognize this value %s", value)
            else:
                # For pass with no params, we set a zero.
                features.append(0)

        return features

    def apply(self, cmd):
        raise NotImplementedError()

    def __repr__(self) -> str:
        return str(self.pass_config)


class OtherOptionSpace:
    """The parameter space for general option"""

    def __init__(self, axes, policy, **kwargs):
        super(OtherOptionSpace, self).__init__()

        candidate = kwargs["candidate"]
        self.entities = [OtherOptionEntity(x) for x in candidate]

    @staticmethod
    def get_num_output(axes, policy, **kwargs):
        return 0

    def __repr__(self):
        return f"OtherOption({self.entities}) len={len(self)}"


class OtherOptionEntity(object):
    """The parameter entity for general option, with a detailed value"""

    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return str(self.val)


class ConfigSpace(object):
    """The configuration space of a pass.Pass it as config to search space"""

    def __init__(self):
        # hardware config and software config
        self.space_map = OrderedDict()
        self._length = None
        self._dims = None
        self._range_length = None
        self._entity_map = OrderedDict()
        self._constraints = []
        self.errors = []
        self.code_hash = None
        self.cost = None

    # TODO: Specify konb for do the search.
    def _add_new_transform(self, space_class, name, args: dict):
        space = space_class(args)
        self.space_map[name] = space
        self._entity_map[name] = space[0]

    @property
    def range_length(self):
        """Length of the index range in the space"""
        if self._range_length is None:
            self._range_length = int(np.prod([len(x) for x in self.space_map.values()]))
        return self._range_length

    @property
    def dims(self):
        """Dimensions in the space"""
        if self._dims is None:
            self._dims = [len(x) for x in self.space_map.values()]
        return self._dims

    def is_index_valid(self, index):
        """Checks if the index satisfies the multi_filter condition"""
        return index >= 0 and index < self.range_length

    def subrange_length(self, start, end):
        """Returns the number of valid indexes within the limited range from [start, end]"""
        assert 0 <= start <= end <= self.range_length
        return end - start

    def get_rand_index(self, start=None, end=None, to_exclude=None):
        """Returns a random valid index unlisted to exclusion"""
        start = start or 0
        end = end or self.range_length
        while True:
            index = randrange(start, end)
            if self.is_index_valid(index) and index not in (to_exclude or []):
                return index

    def get_next_index(self, index, n=1, start=None, end=None):
        """Returns the nth valid next index or None if out of range"""
        assert n != 0
        start = start or 0
        end = end or self.range_length
        if self._shared_filter is None:
            index += n
            if start <= index < end:
                return index
            return None
        trend = 1 if n > 0 else -1
        counter = abs(n)
        while counter != 0:
            index += trend
            if index < start or index >= end:
                return None
            if self.is_index_valid(index):
                counter -= 1
        return index

    def define_gemmini(self, name, args: dict):
        """Define a new tunable knob which use gemmini passes"""
        self._add_new_transform(GemminiPassSpace, name, args)

    def define_gemmini_hardware(self, name, args: dict):
        """Define a new tunable knob which use different gemmini HW config"""
        self._add_new_transform(GemminiSpace, name, args)

    def define_linalg(self, name, args: dict):
        """Define a new tunable knob which use linalg passes"""
        self._add_new_transform(LinalgPassSpace, name, args)

    def point2knob(self, point):
        """Convert point form (single integer) to knob (vector)"""
        knob = []
        for dim in self.dims:
            knob.append(point % dim)
            point //= dim
        return knob

    def knob2point(self, knob):
        """Convert knob form (vector) to point form (single integer)"""
        point = 0
        for j, k in enumerate(knob):
            point += int(np.prod(self.dims[:j])) * k
        return point

    def sample_ints(self, m):
        """Sample m different integer numbers from [0, self.range_length) without replacement"""
        assert m <= len(self)
        vis = set()
        while len(vis) < m:
            new = randrange(0, self.range_length)
            if self.is_index_valid(new):
                vis.add(new)
        return np.fromiter(vis, int, len(vis))

    def random_walk(self, point):
        """random walk as local transition, given an index, returns new neighborhood index"""
        # transform to knob form
        old_knob = self.point2knob(point)
        new_knob = old_knob.copy()
        new_point = self.knob2point(new_knob)
        # mutate
        while new_knob == old_knob or not self.is_index_valid(new_point):
            from_i = np.random.randint(len(old_knob))
            to_v = np.random.randint(self.dims[from_i])
            new_knob[from_i] = to_v
            new_point = self.knob2point(new_knob)
        # transform to index form
        return new_point

    def get(self, index):
        """Get a config entity with detailed parameters from this space"""
        if not self.is_index_valid(index):
            raise IndexError(
                f"Index out of range: size {self.range_length}, got index {index}"
            )
        entities = OrderedDict()
        t = index
        for name, space in self.space_map.items():
            entities[name] = space[t % len(space)]
            t //= len(space)
        ret = ConfigEntity(index, self.code_hash, entities, self._constraints)
        return ret

    def __len__(self):
        """Returns the number of valid indexes in the space"""
        # TODO: how we define the length of config space
        self._length = len(self.space_map)
        return self._length

    def __iter__(self):
        return self._entity_map.__iter__()

    def __repr__(self):
        res = f"ConfigSpace (len={len(self)}, range_length={self._range_length}, space_map=\n"
        for i, (name, space) in enumerate(self.space_map.items()):
            res += f"  {i:2d} {name}: {space}\n"
        return res + ")"


class ConfigEntity(ConfigSpace):
    """A configuration with detailed parameters"""

    def __init__(self, index, code_hash, entity_map, constraints):
        super(ConfigEntity, self).__init__()
        self.index = index
        self._collect = False
        self._entity_map = entity_map
        self._space_map = None
        self._constraints = constraints
        self.code_hash = code_hash

    def get_flatten_feature(self):
        """flatten entities to a numerical one-dimensional feature vector

        Returns
        -------
        fea: np.array
            one dimensional float32 array
        """
        fea = []
        for _, v in self._entity_map.items():
            if isinstance(v, GemminiPassEntity):
                fea.extend(v.features)
            elif isinstance(v, LinalgPassEntity):
                fea.extend(v.features)
            elif isinstance(v, GemminiEntity):
                fea.extend(v.features)
            elif isinstance(v, OtherOptionEntity):
                fea.append(v.val)
        return np.array(fea, dtype=np.float32)

    def get_other_option(self):
        pass
        # return {x: x.val for x in self._entity_map.values() if isinstance(x, OtherOptionEntity)}

    def to_json_dict(self):
        """convert to a json serializable dictionary"""
        ret = {}
        ret["index"] = int(self.index)
        ret["code_hash"] = self.code_hash
        entity_map = []
        for k, v in self._entity_map.items():
            if isinstance(v, GemminiEntity):
                entity_map.append((k, v.hardware_config))
            elif isinstance(v, GemminiPassEntity):
                entity_map.append((k, v.pass_config))
            elif isinstance(v, LinalgPassEntity):
                entity_map.append((k, v.pass_config))
            elif isinstance(v, OtherOptionEntity):
                entity_map.append((k, v.val))
            else:
                raise RuntimeError("Invalid entity instance: " + str(v))
        ret["entity"] = entity_map
        return ret

    @staticmethod
    def from_json_dict(json_dict):
        """Build a ConfigEntity from json serializable dictionary"""
        index = json_dict["index"]
        code_hash = json_dict["code_hash"]
        constraints = []
        entity_map = OrderedDict()

        for item in json_dict["entity"]:
            key, knob_type, knob_args = item
            if knob_type == "gemminiHW":
                entity = GemminiEntity(knob_args)
            elif knob_type == "gemminiPASS":
                entity = GemminiPassEntity(knob_args)
            elif knob_type == "ot":
                entity = OtherOptionEntity(knob_args)
            else:
                raise RuntimeError("Invalid config knob type: " + knob_type)
            entity_map[str(key)] = entity

        return ConfigEntity(index, code_hash, entity_map, constraints)

    def __repr__(self):
        return f"{str(self._entity_map)[12:-1]},{self.code_hash},{self.index}"
