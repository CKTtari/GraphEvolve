"""GraphEvolve: graph-engineering autonomous experimentation."""

from ._version import __version__
from .orchestrator import GraphEvolve
from .schemas import TaskContract

__all__ = ["GraphEvolve", "TaskContract", "__version__"]
