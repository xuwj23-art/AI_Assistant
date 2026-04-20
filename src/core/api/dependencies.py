"""
API 依赖

管理共享资源和依赖注入
"""
from pathlib import Path
from .services import PaperService


# 全局服务实例（单例）
_paper_service: PaperService = None


def get_paper_service() -> PaperService:
    """
    获取论文服务实例（依赖注入）
    
    使用单例模式，确保只创建一个实例
    """
    global _paper_service
    
    if _paper_service is None:
        # 构建数据文件路径
        project_root = Path(__file__).parent.parent.parent.parent
        data_path = project_root / "data" / "raw" / "arxiv_Transformer.csv"
        
        # 创建服务实例
        _paper_service = PaperService(str(data_path))
        print(f" 初始化 PaperService: {data_path}")
    
    return _paper_service


def reset_service():
    """
    重置服务（用于测试）
    """
    global _paper_service
    _paper_service = None