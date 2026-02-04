import pandas as pd
from pathlib import Path
from typing import List, Optional
from .models import PaperResponse

class PaperService:
    """
    负责论文数据的读取和处理
    """
    def __init__(self,data_path:str):
        self.data_path = Path(data_path)
        self._df = Optional[pd.DataFrame] = None

    def _load_data(self) -> pd.DataFrame:
        if self._df is None:
            if not self.data_path.exists():
                raise FileNotFoundError(f"数据文件不存在:{self.data_path}")
            self._df = pd.read_csv(self.data_path)
            print(f"加载了 {len(self._df)}篇论文")
        return self._df
    
    def _row_to_paper(self, row: pd.Series) -> PaperResponse:
        """
        将DataFrame 转换为 PaperResponse

        参数:
            row: DataFrame的一行
        返回:
            PaperResponse 对象
        """
        # 处理作者列表
        authors = row.get('authors', '[]') # []代表默认值,如果authors列不存在
        if isinstance(authors, str):
            # 如果是字符串，尝试解析
            import ast
            try:
                authors = ast.literal_eval(authors)
            except:
                authors = [authors]

        # 将str类型的数据转化为需要的List[str]
        categories = row.get('categories', '[]')
        if isinstance(categories, str):
            # 如果是字符串，尝试解析
            import ast
            try:
                categories = ast.literal_eval(categories)
            except:
                categories = [categories]  

        # 可选字段：空值转 None，避免 str(NaN) == "nan"
        pdf_url = row.get('pdf_url')
        if pd.isna(pdf_url):
            pdf_url = None
        else:
            pdf_url = str(pdf_url)

        # 必填 str 字段：空值转成 ""，避免 str(NaN) 得到 "nan"（不能转 None，模型要求必填）
        id_val = row.get('id', '')
        title_val = row.get('title', '')
        abstract_val = row.get('abstract', '')
        if pd.isna(id_val):
            id_val = ''
        if pd.isna(title_val):
            title_val = ''
        if pd.isna(abstract_val):
            abstract_val = ''

        return PaperResponse(
            id=str(id_val),
            title=str(title_val),
            authors=authors if isinstance(authors, list) else [],
            abstract=str(abstract_val),
            published=row.get('published'),
            pdf_url=pdf_url,
            categories=categories if isinstance(categories, list) else []
        ) 
    def get_all_papers(self, limit: int = 100) -> List[PaperResponse]:
        """
        获取所有论文
        limit:最多返回数量
        """
        df = self._load_data()
        df = df.head(limit)

        papers = []
        for _, row in df.iterrows():
            paper = self._row_to_paper(row)
            papers.append(paper)
        return papers

    def get_paper_by_id(self, paper_id: str) -> Optional[PaperResponse]:
        """
        根据ID获取论文
        """
        df = self._load_data()

        result = df[df['id'] == paper_id]

        if result.empty:
            return None
        
        row = result.iloc[0]
        return self._row_to_paper(row)

    def search_papers(
        self,
        query: str,
        fields: List[str] = None,
        max_results: int = 10
    ) -> List[PaperResponse]:
        """
        搜索论文,在指定字段fields中寻找关键词query
        """
        df = self._load_data()

        if fields is None:
            fields = ["title", "abstract"]

        mask = pd.Series([False] * len(df))

        for field in fields:
            if field in df.columns:
                mask |= df[field].str.contains(
                    query,
                    case=False,
                    na=False
                )
        
        result_df = df[mask].head(max_results)

        papers = []
        for _, row in result_df.iterrows():
            paper = self._row_to_paper(row)
            papers.append(paper)
        return papers
    
    def get_stats(self) -> dict:
        """ 获取统计信息 """
        df =self._load_data()

        return {
            "total_papers": len(df),
            "date_range": {
                "earliest": df['published'].min() if 'published' in df.columns else None,
                "latest": df['published'].max() if 'published' in df.columns else None
            }            
        }

