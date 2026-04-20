"""
多数据源统一接口模块
"""
from .base import Paper, PaperSource
from .aggregator import MultiSourceAggregator

__all__ = ['Paper', 'PaperSource', 'MultiSourceAggregator']
