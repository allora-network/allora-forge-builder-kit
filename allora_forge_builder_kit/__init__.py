__version__ = "3.0.0"

from .workflow import AlloraMLWorkflow
from .binance_data_manager import BinanceDataManager
from .atlas_data_manager import AtlasDataManager
from .base_data_manager import BaseDataManager
from .data_manager_factory import DataManager, list_data_sources
from .utils import get_api_key
from .evaluation import PerformanceEvaluator
from .topic_discovery import AlloraTopicDiscovery, TopicInfo

__all__ = [
    "__version__",
    "AlloraMLWorkflow",
    "BinanceDataManager",
    "AtlasDataManager",
    "BaseDataManager",
    "DataManager",
    "list_data_sources",
    "get_api_key",
    "PerformanceEvaluator",
    "AlloraTopicDiscovery",
    "TopicInfo",
]


def __getattr__(name):
    if name == "AlloraDataManager":
        import warnings
        warnings.warn(
            "AlloraDataManager was removed in v3.0. Use AtlasDataManager instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AtlasDataManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
