from keras_core.src import backend
from keras_core.src.api_export import keras_core_export
from keras_core.src.optimizers import base_optimizer

if backend.backend() == "tensorflow":
    from keras_core.src.backend.tensorflow.optimizer import TFOptimizer

    BackendOptimizer = TFOptimizer
elif backend.backend() == "torch":
    from keras_core.src.backend.torch.optimizers import TorchOptimizer

    BackendOptimizer = TorchOptimizer
else:
    BackendOptimizer = base_optimizer.BaseOptimizer


@keras_core_export(["keras_core.Optimizer", "keras_core.optimizers.Optimizer"])
class Optimizer(BackendOptimizer):
    pass


base_optimizer_keyword_args = base_optimizer.base_optimizer_keyword_args
Optimizer.__doc__ = base_optimizer.BaseOptimizer.__doc__

