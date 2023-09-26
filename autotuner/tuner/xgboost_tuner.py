"""Tuner that uses xgboost as cost model"""

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
                "Optimizer must be " "a supported name string" "or a ModelOptimizer object."
            )

        super(XGBTuner, self).__init__(
            task, cost_model, optimizer, plan_size, diversity_filter_ratio
        )

    def tune(self, *args, **kwargs):
        super(XGBTuner, self).tune(*args, **kwargs)

        # manually close pool to avoid multiprocessing issues
        self.cost_model._close_pool()
