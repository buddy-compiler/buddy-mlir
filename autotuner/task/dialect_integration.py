# TODO: 做算子层级优化，暂时用不到

import logging

from .task import create, serialize_args


logger = logging.getLogger("autotuner")


# Task extractor for program
class TaskExtractEnv:
    """Global environment for extracting tuning tasks from graph"""

    current = None
    registered = None

    def __init__(self, allow_duplicate=False):
        self.allow_duplicate = allow_duplicate
        self.task_collection = []
        self.wanted_ops = None
        self.modified_funcs = []
        self.tracing = False

    def __enter__(self):
        self.task_collection = []
        self.tracing = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracing = False

    def reset(self, wanted_ops=None):
        """Reset task collections"""
        self.task_collection = []
        self.wanted_relay_ops = wanted_ops

    def add_task(self, task_name, args):
        """Add AutoTuner task"""
        key = (task_name, serialize_args(args))
        if self.allow_duplicate or key not in self.task_collection:
            self.task_collection.append(key)

    def get_tasks(self):
        """Get collected tasks"""
        return self.task_collection

    @staticmethod
    def get(allow_duplicate=False):
        """Get the single instance of TaskExtractEnv"""
        if not TaskExtractEnv.current:
            TaskExtractEnv.current = TaskExtractEnv(allow_duplicate)
        else:
            TaskExtractEnv.current.allow_duplicate = allow_duplicate
        return TaskExtractEnv.current


def extract_from_program(program, params, target="gemmini", ops=None):
    env = TaskExtractEnv.get()
    env.reset(ops)
    with env:
        # disable logger temporarily
        old_state = logger.disabled
        logger.disabled = True

        # TODO: lower code, maybe multiple passes here

        logger.disabled = old_state
    
    tasks = []
    for task_name, args in env.get_tasks():
        try:
            tsk = create(task_name, args, target=target)
            tasks.append(tsk)
        except:
            logger.warning("Invalid, error occurs during AutoTuner task creation")

    return tasks