from keras_core.src import optimizers
from keras_core.src.backend.torch.optimizers import torch_adam


class AdamW(torch_adam.Adam, optimizers.AdamW):
    pass

