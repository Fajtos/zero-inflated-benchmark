from .base import BaseModel
from .classical import CLASSICAL_MODELS
from .regularized import REGULARIZED_MODELS
from .tree import TREE_MODELS
from .neural import NEURAL_MODELS

ALL_MODELS = {}
ALL_MODELS.update(CLASSICAL_MODELS)
ALL_MODELS.update(REGULARIZED_MODELS)
ALL_MODELS.update(TREE_MODELS)
ALL_MODELS.update(NEURAL_MODELS)