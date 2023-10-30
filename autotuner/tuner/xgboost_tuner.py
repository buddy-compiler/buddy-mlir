# ===- xgboost_tuner.py --------------------------------------------------------
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
# Tuner that uses xgboost as cost model.
#
# ===---------------------------------------------------------------------------

from .model_based_tuner import ModelBasedTuner, ModelOptimizer
from .xgboost_cost_model import XGBoostCostModel
from .sa_model_optimizer import SimulatedAnnealingOptimizer


class XGBTuner(ModelBasedTuner):
    """Tuner that uses xgboost as cost model"""

    def __init__(
        self,
        task,
        plan_size=64,
        loss_type="reg",
        num_threads=None,
        optimizer="sa",
        diversity_filter_ratio=None,
        log_interval=25,
    ):
        cost_model = XGBoostCostModel(
            task,
            loss_type=loss_type,
            num_threads=num_threads,
            log_interval=log_interval // 2,
        )
        if optimizer == "sa":
            optimizer = SimulatedAnnealingOptimizer(task, log_interval=log_interval)
        else:
            assert isinstance(optimizer, ModelOptimizer), (
                "Optimizer must be "
                "a supported name string"
                "or a ModelOptimizer object."
            )

        super(XGBTuner, self).__init__(
            task, cost_model, optimizer, plan_size, diversity_filter_ratio
        )

    def tune(self, *args, **kwargs):
        super(XGBTuner, self).tune(*args, **kwargs)

        # manually close pool to avoid multiprocessing issues
        self.cost_model._close_pool()
