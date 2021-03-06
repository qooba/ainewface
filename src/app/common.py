import qdi
from typing import Any, Callable, Optional, Sequence
from functools import lru_cache, partial
from fastapi import Depends
from services.face import FaceService

class Bootstapper:

    def bootstrap(self):
        c = Bootstapper.container()
        c.register_instance(qdi.IContainer, c)
        c.register_instance(qdi.IFactory, qdi.Factory(c))

        # services
        face_service=FaceService()
        c.register_instance(FaceService,face_service)

        return c

    @staticmethod
    @lru_cache()
    def container():
        return qdi.Container()


def Injects(from_cls: Any, key: str='') -> Any:
    dependency=partial(Bootstapper.container().resolve, from_cls, key)
    return Depends(dependency)

