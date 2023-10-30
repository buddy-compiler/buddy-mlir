# ===- callback.py -------------------------------------------------------------
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
# Callback functions.
#
# ===---------------------------------------------------------------------------

import logging
import json
import pickle
import base64

logger = logging.getLogger("autotuner")


def log_to_file(file_out, protocol="json"):
    def _callback(_, inputs, results):
        """Callback implementation"""
        if isinstance(file_out, str):
            with open(file_out, "a") as f:
                for inp, result in zip(inputs, results):
                    f.write(encode(inp, result, protocol) + "\n")
        else:
            for inp, result in zip(inputs, results):
                file_out.write(encode(inp, result, protocol) + "\n")

    # pylint: disable=import-outside-toplevel
    from pathlib import Path

    if isinstance(file_out, Path):
        file_out = str(file_out)

    return _callback


def encode(inp, result, protocol="json"):
    if protocol == "json":
        json_dict = {
            "input": (str(inp.target), inp.task.name, inp.task.args),
            "config": inp.config.to_json_dict(),
            "result": (
                result.costs if result.error_no == 0 else (1e9,),
                result.error_no,
                result.all_cost,
                result.timestamp,
            ),
        }
        return json.dumps(json_dict)
    if protocol == "pickle":
        row = (
            str(inp.target),
            str(
                base64.b64encode(pickle.dumps([inp.task.name, inp.task.args])).decode()
            ),
            str(base64.b64encode(pickle.dumps(inp.config)).decode()),
            str(base64.b64encode(pickle.dumps(tuple(result))).decode()),
        )
        return "\t".join(row)

    raise RuntimeError("Invalid log protocol: " + protocol)
