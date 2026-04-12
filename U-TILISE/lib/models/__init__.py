from typing import Dict, Type, Union

from .ImageSeriesInterpolator import ImageSeriesInterpolator
from .utilise import UTILISE

MODELS: Dict[str, Union[Type[UTILISE], Type[ImageSeriesInterpolator]]] = {
    "utilise": UTILISE,
    "ImageSeriesInterpolator": ImageSeriesInterpolator
}