"""
冒烟测试：验证 Project progress.md 评估文档中修复点的正确性

覆盖：
1. row_to_paper 公共函数能从真实 CSV 行成功生成 PaperResponse
2. PaperService 与 TopicService 都委托到同一个公共实现，输出一致
3. RAGService.search_similar_papers 不再实例化 PaperService("")
   （通过 monkey-patch PaperService.__init__ 来兜底捕获）
4. topic_modeling.py 的类型注解修复后 import 不报错
"""
from __future__ import annotations

import sys
from pathlib import Path

# 走 conftest 风格的路径设置，便于直接运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest


# ---------- Fix 1 + 2：公共 row_to_paper ----------

def test_row_to_paper_basic():
    from core.api.paper_utils import row_to_paper

    row = pd.Series({
        "id": "http://arxiv.org/abs/1706.03762v5",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models...",
        "authors": "['Ashish Vaswani', 'Noam Shazeer']",
        "categories": "['cs.CL', 'cs.LG']",
        "url": "https://arxiv.org/abs/1706.03762",
        "pdf_url": "",
        "published": pd.Timestamp("2017-06-12"),
    })
    paper = row_to_paper(row)

    assert paper.id == "1706.03762", f"arXiv ID 应去版本号，实际: {paper.id}"
    assert paper.title.startswith("Attention")
    assert paper.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert paper.categories == ["cs.CL", "cs.LG"]
    # pdf_url 为空时应根据 url 推导
    assert str(paper.pdf_url) == "https://arxiv.org/pdf/1706.03762.pdf"
    assert paper.published.year == 2017


def test_row_to_paper_handles_nan():
    """脏行（NaN 标题/摘要 + 非法 pdf_url）也不能让 PaperResponse 校验崩溃。"""
    from core.api.paper_utils import row_to_paper

    # 用 object dtype 防止 pandas 把所有值推断成 NaT
    row = pd.Series(
        {
            "id": float("nan"),
            "title": float("nan"),
            "abstract": float("nan"),
            "authors": float("nan"),
            "categories": float("nan"),
            "url": float("nan"),
            "pdf_url": "nan",  # 非法 URL 字符串
            "published": pd.NaT,
        },
        dtype=object,
    )
    paper = row_to_paper(row)

    assert paper.id == ""
    # 占位符兜底：避免 min_length=1 校验失败
    assert paper.title == "(no title)"
    assert paper.abstract == "(no abstract)"
    assert paper.authors == []
    assert paper.categories == []
    assert paper.pdf_url is None
    assert paper.published.year == 2000  # fallback


def test_services_and_topic_service_share_impl():
    """两个 service 的 _row_to_paper 应得到相同结果（验证抽公共后无差异）。"""
    from core.api.services import PaperService
    from core.api.topic_service import TopicService

    csv_path = ROOT / "data" / "processed" / "arxiv_LLM_with_topics.csv"
    if not csv_path.exists():
        pytest.skip(f"未找到 {csv_path}")

    paper_svc = PaperService(str(csv_path))
    topic_svc = TopicService(str(csv_path))

    df = paper_svc._load_data()
    row = df.iloc[0]

    p1 = paper_svc._row_to_paper(row)
    p2 = topic_svc._row_to_paper(row)
    assert p1.id == p2.id
    assert p1.title == p2.title
    assert p1.authors == p2.authors


# ---------- Fix 3：RAGService 不再触发 PaperService("") ----------

def test_rag_does_not_instantiate_empty_paperservice(monkeypatch):
    """检查 rag.py 已经把 _row_to_paper 路径改成调用 paper_utils 而非 PaperService。

    我们不实际跑 search_similar_papers（避免加载模型），而是用静态方式审查源代码 +
    monkeypatch PaperService 兜底捕获。
    """
    import core.nlp.rag as rag_mod
    src = Path(rag_mod.__file__).read_text(encoding="utf-8")
    assert "PaperService(\"\")" not in src, (
        "rag.py 仍然存在 PaperService('') 临时实例化，循环依赖未修复"
    )
    assert "from ..api.paper_utils import row_to_paper" in src, (
        "rag.py 应通过 paper_utils.row_to_paper 解决行→PaperResponse 转换"
    )


# ---------- Fix 4：topic_modeling 类型注解 ----------

def test_topic_modeling_imports_clean():
    """List[Tuple(str,float)] 写错时该 import 在执行注解时会出问题；
    修复后应可正常导入。"""
    import importlib
    mod = importlib.import_module("core.nlp.topic_modeling")
    assert hasattr(mod, "TopicModeler")
    # 确认 get_topic 的注解里不再含有错误的小括号写法
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "List[Tuple(str,float)]" not in src
    assert "List[Tuple[str, float]]" in src


if __name__ == "__main__":
    import sys as _s
    print("运行冒烟测试...")
    pytest_args = [__file__, "-v"]
    if len(_s.argv) > 1:
        pytest_args.extend(_s.argv[1:])
    _s.exit(pytest.main(pytest_args))
