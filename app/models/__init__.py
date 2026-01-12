# Modèles de files d'attente selon la notation de Kendall
from .base_queue import BaseQueueModel, GenericQueue, ChainQueue
from .old.mm1 import MM1Queue
from .old.mmc import MMcQueue
from .old.mmck import MMcKQueue
from .old.md1 import MD1Queue
from .old.mg1 import MG1Queue
from .old.mdc import MDcQueue
from .old.mgc import MGcQueue

__all__ = [
    'BaseQueueModel',
    'MM1Queue',
    'MMcQueue', 
    'MMcKQueue',
    'MD1Queue',
    'MG1Queue',
    'MDcQueue',
    'MGcQueue',
    'GenericQueue',
    'ChainQueue'
]
