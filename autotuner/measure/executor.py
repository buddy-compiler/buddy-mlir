""" Abstraction for asynchronous job execution """

class Executor(object):
    """
    Base abstract executor interface for asynchronous job submission.
    Allows submit asynchronous jobs and returns the Future object.
    """

    # TODO: Gemmini 使用 Spike 模拟运行，可能会非常慢，timeout 暂时先不定，后续看 cost model 的时延
    # timeout for jobs that may hang
    DEFAULT_TIMEOUT = 120

    def submit(self, func, *args, **kwargs):
        """
        Pass task (function, arguments) to the Executor.

        Returns
        -------
        future : Future
            Future object wrapping the task which can be used to
            collect the task's result.
        """
        raise NotImplementedError()


class Future(object):
    """
    Base class of the future object.
    The implementations can return object of subclass of this.
    This objects encapsulates the asynchronous execution of task
    submitted to another thread, or another worker for execution.

    Future objects store the state of tasks--can be polled for
    result or a blocking call to retrieve the result can be used.
    """

    def done(self):
        """Return True if job was successfully cancelled or finished running."""
        raise NotImplementedError()

    def get(self, timeout=None):
        """Get the result. This will block until the result is available."""
        raise NotImplementedError()


class FutureError(RuntimeError):
    """Base error class of all future events"""


# pylint:disable=redefined-builtin
class TimeoutError(FutureError):
    """Error raised when a task is timeout."""


class ExecutionError(FutureError):
    """Error raised when future execution crashes or failed."""