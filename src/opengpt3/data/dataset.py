import torch
from typing import Optional, Dict, Any


class Dataset(object):
    def skip(self, count: int):
        raise NotImplementedError()

    def where(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def assign(self, where: Dict[str, Any]):
        raise NotImplementedError()
