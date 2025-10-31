from typing import Any, Dict

from loguru import logger


class BaseRegistry:
    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, item: Any):
        """Register an item with a name"""
        cls._registry[name] = item
        logger.debug(f"Registered {cls.__name__}: {name}")

    @classmethod
    def get(cls, name: str, **kwargs) -> Any:
        """Get an instance by name with optional parameters"""
        if name not in cls._registry:
            raise ValueError(f"{name} not found in {cls.__name__}")
        try:
            return cls._registry[name](**kwargs)
        except Exception as e:
            logger.error(f"Error instantiating {name} with params {kwargs}: {str(e)}")
            raise

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all registered items"""
        return cls._registry.copy()
