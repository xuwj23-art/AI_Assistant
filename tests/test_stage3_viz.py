# -*- coding: utf-8 -*-
"""
Stage 3 Integration Tests - Visualization API

测试内容:
  1. trend_analysis 模块单元测试（无需后端）
  2. /api/topics/sunburst 端点测试
  3. /api/topics/trends 端点测试
  4. /api/topics/{id}/papers 分页排序测试
  5. /api/topics/{id}/similar 相关推荐测试

运行方式:
  # 仅运行单元测试（无需后端）:
  cd e:/AIassistant_v2
  python src/test_stage3_viz.py --unit

  # 运行全部测试（需要后端在线）:
  python src/test_stage3_viz.py

  # 或使用 pytest:
  pytest src/test_stage3_viz.py -v
"""
from __future__ import annotations

import sys
import io
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# 强制 stdout 使用 UTF-8（解决 Windows GBK 终端问题）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np

# 确保 src 目录在 Python 路径中（移动到 tests/ 后需指向 ../src）
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# 工具函数
# ============================================================

def _print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _ok(msg: str):
    print(f"  [OK]   {msg}")


def _fail(msg: str):
    print(f"  [FAIL] {msg}")


def _info(msg: str):
    print(f"  [INFO] {msg}")


# ============================================================
# 单元测试：trend_analysis 模块
# ============================================================

def test_trend_analysis_basic():
    """测试 compute_topic_trends 基本功能"""
    _print_section("单元测试 1: compute_topic_trends 基本功能")

    from core.nlp.trend_analysis import compute_topic_trends

    papers = [
        {"topic_id": 0, "published_date": "2021-03-01"},
        {"topic_id": 0, "published_date": "2021-07-15"},
        {"topic_id": 0, "published_date": "2022-01-10"},
        {"topic_id": 1, "published_date": "2021-05-20"},
        {"topic_id": 1, "published_date": "2022-08-30"},
        {"topic_id": 1, "published_date": "2022-11-01"},
        {"topic_id": -1, "published_date": "2021-01-01"},  # 噪声，应被过滤
    ]
    topic_names = {0: "Transformer", 1: "BERT/GPT"}

    df = compute_topic_trends(papers, topic_names=topic_names)

    assert not df.empty, "结果不应为空"
    assert "Transformer" in df.columns, "应包含 Transformer 列"
    assert "BERT/GPT" in df.columns, "应包含 BERT/GPT 列"
    assert 2021 in df.index, "应包含 2021 年"
    assert 2022 in df.index, "应包含 2022 年"
    assert df.loc[2021, "Transformer"] == 2, f"2021年Transformer应为2，实际为{df.loc[2021, 'Transformer']}"
    assert df.loc[2022, "BERT/GPT"] == 2, f"2022年BERT/GPT应为2，实际为{df.loc[2022, 'BERT/GPT']}"

    _ok(f"DataFrame 形状: {df.shape}")
    _ok(f"年份范围: {df.index.min()} - {df.index.max()}")
    _ok("compute_topic_trends 基本功能正常")
    return True


def test_trend_analysis_empty():
    """测试空输入处理"""
    _print_section("单元测试 2: 空输入处理")

    from core.nlp.trend_analysis import compute_topic_trends, get_trending_topics

    df_empty = compute_topic_trends([])
    assert df_empty.empty, "空输入应返回空 DataFrame"
    _ok("空论文列表 → 空 DataFrame")

    trending = get_trending_topics(pd.DataFrame())
    assert trending == [], "空 DataFrame 应返回空列表"
    _ok("空 DataFrame → 空趋势列表")

    return True


def test_get_trending_topics():
    """测试 get_trending_topics 增长率计算"""
    _print_section("单元测试 3: get_trending_topics 增长率计算")

    from core.nlp.trend_analysis import get_trending_topics

    # 构造一个明显增长的主题
    data = {
        "Transformer": [2, 3, 5, 10, 25, 60],   # 快速增长
        "Old Method":  [20, 18, 15, 12, 10, 8],  # 下降
        "Stable":      [10, 10, 11, 10, 11, 10],  # 稳定
    }
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    df = pd.DataFrame(data, index=years)
    df.index.name = "year"

    trending = get_trending_topics(df, top_n=3)

    assert len(trending) > 0, "应返回至少一个主题"
    assert trending[0][0] == "Transformer", f"增长最快应为 Transformer，实际为 {trending[0][0]}"
    assert trending[0][1] > 1.0, "增长率应大于 1"

    _ok(f"增长最快: {trending[0][0]} (增长率 {trending[0][1]:.2f}x)")
    _ok(f"增长最慢: {trending[-1][0]} (增长率 {trending[-1][1]:.2f}x)")
    _ok("get_trending_topics 增长率计算正常")
    return True


def test_extract_year_formats():
    """测试 _extract_year 对各种日期格式的处理"""
    _print_section("单元测试 4: 日期格式解析")

    from core.nlp.trend_analysis import _extract_year

    test_cases = [
        (datetime(2023, 6, 15), 2023),
        (pd.Timestamp("2022-03-01"), 2022),
        ("2021-01-15", 2021),
        ("2020-12-31T00:00:00", 2020),
        ("2019", 2019),
        (2018, 2018),
        (2018.0, 2018),
        (None, None),
        ("", None),
        (pd.NaT, None),
    ]

    all_pass = True
    for val, expected in test_cases:
        result = _extract_year(val)
        if result == expected:
            _ok(f"_extract_year({repr(val)!s:40s}) = {result}")
        else:
            _fail(f"_extract_year({repr(val)!s:40s}) = {result}，期望 {expected}")
            all_pass = False

    return all_pass


def test_trend_df_to_api_format():
    """测试 trend_df_to_api_format 转换"""
    _print_section("单元测试 5: trend_df_to_api_format 格式转换")

    from core.nlp.trend_analysis import trend_df_to_api_format

    data = {"TopicA": [5, 10, 20], "TopicB": [3, 8, 15]}
    df = pd.DataFrame(data, index=[2021, 2022, 2023])
    df.index.name = "year"

    result = trend_df_to_api_format(df)

    assert "years" in result, "应包含 years 字段"
    assert "topics" in result, "应包含 topics 字段"
    assert result["years"] == [2021, 2022, 2023], f"years 不正确: {result['years']}"
    assert len(result["topics"]) == 2, f"应有 2 个主题，实际 {len(result['topics'])}"

    topic_names = [t["name"] for t in result["topics"]]
    assert "TopicA" in topic_names, "应包含 TopicA"
    assert "TopicB" in topic_names, "应包含 TopicB"

    _ok(f"years: {result['years']}")
    _ok(f"topics: {[t['name'] for t in result['topics']]}")
    _ok("trend_df_to_api_format 格式转换正常")
    return True


# ============================================================
# API 集成测试（需要后端在线）
# ============================================================

API_BASE = "http://127.0.0.1:8000"


def _check_backend() -> bool:
    """检查后端是否在线"""
    try:
        import requests
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def test_api_sunburst():
    """测试 /api/topics/sunburst 端点"""
    _print_section("API 测试 1: GET /api/topics/sunburst")

    import requests

    r = requests.get(f"{API_BASE}/api/topics/sunburst", timeout=30)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"

    data = r.json()
    required_fields = ["labels", "parents", "values", "ids", "topic_ids"]
    for field in required_fields:
        assert field in data, f"响应缺少字段: {field}"

    assert len(data["labels"]) == len(data["parents"]), "labels 和 parents 长度不一致"
    assert len(data["labels"]) == len(data["values"]), "labels 和 values 长度不一致"
    assert len(data["labels"]) == len(data["ids"]), "labels 和 ids 长度不一致"

    # 根节点检查
    assert data["labels"][0] == "All Topics", f"第一个标签应为 All Topics，实际为 {data['labels'][0]}"
    assert data["parents"][0] == "", f"根节点 parent 应为空字符串"

    n_topics = len(data["labels"]) - 1  # 减去根节点
    _ok(f"返回 {n_topics} 个主题节点")
    _ok(f"根节点: {data['labels'][0]}")
    if n_topics > 0:
        _ok(f"第一个主题: {data['labels'][1]} (论文数: {data['values'][1]})")
    _ok("/api/topics/sunburst 端点正常")
    return True


def test_api_trends():
    """测试 /api/topics/trends 端点"""
    _print_section("API 测试 2: GET /api/topics/trends")

    import requests

    r = requests.get(f"{API_BASE}/api/topics/trends", params={"top_n": 5}, timeout=30)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"

    data = r.json()
    assert "years" in data, "响应缺少 years 字段"
    assert "topics" in data, "响应缺少 topics 字段"
    assert "trending" in data, "响应缺少 trending 字段"

    if data["years"]:
        _ok(f"年份范围: {data['years'][0]} - {data['years'][-1]}")
        _ok(f"主题数量: {len(data['topics'])}")
        if data["trending"]:
            top = data["trending"][0]
            _ok(f"增长最快: {top['name']} (增长率 {top['growth_rate']:.2f}x)")
    else:
        _info("暂无趋势数据（可能尚未训练主题模型）")

    _ok("/api/topics/trends 端点正常")
    return True


def test_api_topics_list():
    """测试 /api/topics 端点"""
    _print_section("API 测试 3: GET /api/topics")

    import requests

    r = requests.get(f"{API_BASE}/api/topics", timeout=30)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"

    data = r.json()
    assert "total" in data, "响应缺少 total 字段"
    assert "topics" in data, "响应缺少 topics 字段"

    _ok(f"主题总数: {data['total']}")
    if data["topics"]:
        first = data["topics"][0]
        _ok(f"最大主题: {first['topic_name']} ({first['paper_count']} 篇)")
    _ok("/api/topics 端点正常")
    return data


def test_api_topic_papers_paged(topics_data=None):
    """测试 /api/topics/{id}/papers 分页排序"""
    _print_section("API 测试 4: GET /api/topics/{id}/papers（分页排序）")

    import requests

    # 获取第一个主题 ID
    if topics_data is None:
        r = requests.get(f"{API_BASE}/api/topics", timeout=30)
        topics_data = r.json()

    if not topics_data.get("topics"):
        _info("无主题数据，跳过此测试")
        return True

    topic_id = topics_data["topics"][0]["topic_id"]

    # 测试按时间排序
    r = requests.get(
        f"{API_BASE}/api/topics/{topic_id}/papers",
        params={"page": 1, "page_size": 5, "sort_by": "date"},
        timeout=30
    )
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()

    assert "total" in data, "响应缺少 total 字段"
    assert "papers" in data, "响应缺少 papers 字段"
    assert "page" in data, "响应缺少 page 字段"
    assert "page_size" in data, "响应缺少 page_size 字段"
    assert data["page"] == 1, f"页码应为 1，实际为 {data['page']}"
    assert len(data["papers"]) <= 5, f"每页不超过 5 篇，实际 {len(data['papers'])}"

    _ok(f"主题 {topic_id}: 共 {data['total']} 篇论文")
    _ok(f"第 1 页（每页5篇，按时间排序）: 返回 {len(data['papers'])} 篇")

    # 测试第 2 页
    if data["total"] > 5:
        r2 = requests.get(
            f"{API_BASE}/api/topics/{topic_id}/papers",
            params={"page": 2, "page_size": 5, "sort_by": "date"},
            timeout=30
        )
        assert r2.status_code == 200
        data2 = r2.json()
        _ok(f"第 2 页: 返回 {len(data2['papers'])} 篇")

    # 测试 404
    r_404 = requests.get(f"{API_BASE}/api/topics/99999/papers", timeout=10)
    assert r_404.status_code == 404, f"不存在的主题应返回 404，实际 {r_404.status_code}"
    _ok("不存在的主题 → 404 正确")

    _ok("/api/topics/{id}/papers 分页排序正常")
    return True


def test_api_similar_topics(topics_data=None):
    """测试 /api/topics/{id}/similar 相关推荐"""
    _print_section("API 测试 5: GET /api/topics/{id}/similar")

    import requests

    if topics_data is None:
        r = requests.get(f"{API_BASE}/api/topics", timeout=30)
        topics_data = r.json()

    if not topics_data.get("topics") or len(topics_data["topics"]) < 2:
        _info("主题数量不足，跳过相关推荐测试")
        return True

    topic_id = topics_data["topics"][0]["topic_id"]

    r = requests.get(
        f"{API_BASE}/api/topics/{topic_id}/similar",
        params={"top_n": 3},
        timeout=30
    )
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"

    data = r.json()
    assert "similar_topics" in data, "响应缺少 similar_topics 字段"
    assert len(data["similar_topics"]) <= 3, "返回数量不应超过 top_n"

    _ok(f"主题 {topic_id} 的相关推荐: {len(data['similar_topics'])} 个")
    for sim in data["similar_topics"]:
        _ok(f"  → {sim['topic_name']} (相似度: {sim['similarity']:.4f})")

    _ok("/api/topics/{id}/similar 端点正常")
    return True


# ============================================================
# 主函数
# ============================================================

def run_unit_tests() -> int:
    """运行所有单元测试，返回失败数"""
    print("\n" + "=" * 60)
    print("  Stage 3 单元测试（无需后端）")
    print("=" * 60)

    tests = [
        test_trend_analysis_basic,
        test_trend_analysis_empty,
        test_get_trending_topics,
        test_extract_year_formats,
        test_trend_df_to_api_format,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            _fail(f"{test_fn.__name__} 断言失败: {e}")
            failed += 1
        except Exception as e:
            _fail(f"{test_fn.__name__} 异常: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  单元测试结果: {passed} 通过 / {failed} 失败")
    print(f"{'='*60}")
    return failed


def run_api_tests() -> int:
    """运行所有 API 集成测试，返回失败数"""
    print("\n" + "=" * 60)
    print("  Stage 3 API 集成测试（需要后端在线）")
    print("=" * 60)

    if not _check_backend():
        print("\n  ⚠️  后端服务未启动，跳过 API 测试")
        print("  启动命令: cd src && python main.py")
        return 0

    _ok("后端服务在线")

    passed = 0
    failed = 0
    topics_data = None

    api_tests = [
        (test_api_sunburst, {}),
        (test_api_trends, {}),
        (test_api_topics_list, {}),
    ]

    for test_fn, kwargs in api_tests:
        try:
            result = test_fn(**kwargs)
            if test_fn.__name__ == "test_api_topics_list":
                topics_data = result  # 保存供后续测试使用
            passed += 1
        except AssertionError as e:
            _fail(f"{test_fn.__name__} 断言失败: {e}")
            failed += 1
        except Exception as e:
            _fail(f"{test_fn.__name__} 异常: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # 依赖 topics_data 的测试
    for test_fn in [test_api_topic_papers_paged, test_api_similar_topics]:
        try:
            test_fn(topics_data=topics_data)
            passed += 1
        except AssertionError as e:
            _fail(f"{test_fn.__name__} 断言失败: {e}")
            failed += 1
        except Exception as e:
            _fail(f"{test_fn.__name__} 异常: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  API 测试结果: {passed} 通过 / {failed} 失败")
    print(f"{'='*60}")
    return failed


def main():
    parser = argparse.ArgumentParser(description="Stage 3 可视化 API 集成测试")
    parser.add_argument(
        "--unit", action="store_true",
        help="仅运行单元测试（无需后端）"
    )
    args = parser.parse_args()

    total_failures = 0
    total_failures += run_unit_tests()

    if not args.unit:
        total_failures += run_api_tests()

    print(f"\n{'='*60}")
    if total_failures == 0:
        print("  [PASS] 所有测试通过！Stage 3 可视化功能就绪。")
    else:
        print(f"  [FAIL] 共 {total_failures} 个测试失败，请检查上方错误信息。")
    print(f"{'='*60}\n")

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
