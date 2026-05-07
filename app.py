"""
Streamlit 前端界面 — 搜索驱动版

交互流程:
  1. 用户在主搜索框输入关键词（如 "transformer attention"）
  2. 点击搜索 → 后端在线抓取论文 → 生成向量 → 训练主题模型
  3. 搜索完成后展示三个标签页：主题地图 / 研究趋势 / AI 对话
"""
import streamlit as st
import requests
import time

import plotly.graph_objects as go
import plotly.express as px

# ========== 配置 ==========

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI 论文助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 全局 CSS ==========

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .search-hint {
        text-align: center;
        color: #999;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    .paper-card {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .trending-badge {
        display: inline-block;
        background: #ff7f0e;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin-right: 6px;
    }
    .step-done   { color: #28a745; font-weight: 600; }
    .step-active { color: #1f77b4; font-weight: 600; }
    .step-wait   { color: #aaa; }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #999;
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)


# ========== 工具函数 ==========

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def api_get(path: str, params: dict = None, timeout: int = 30):
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 503:
            return None  # 无数据，静默处理
        else:
            st.error(f"API 错误 {r.status_code}: {r.text[:200]}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到后端服务，请先启动: `cd src && python main.py`")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ 请求超时，请稍后重试")
        return None
    except Exception as e:
        st.error(f"请求失败: {e}")
        return None


@st.cache_data(ttl=60, show_spinner=False)
def cached_api_get(path: str, params_str: str = "", timeout: int = 30):
    """缓存版 API GET 请求，避免 rerun 时重复调用"""
    import json
    params = json.loads(params_str) if params_str else None
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def api_post(path: str, payload: dict, timeout: int = 300):
    try:
        r = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 404:
            st.error("会话已过期，请刷新页面重新开始")
            return None
        elif r.status_code == 409:
            st.warning("正在处理中，请稍候...")
            return None
        else:
            st.error(f"请求失败: HTTP {r.status_code} — {r.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        st.error("请求超时（论文抓取可能需要较长时间，请稍后重试）")
        return None
    except Exception as e:
        st.error(f"请求失败: {e}")
        return None


def do_search(query: str, max_results: int, sources: list = None) -> bool:
    """执行搜索流水线，返回是否成功"""
    steps = [
        "🌐 在线抓取论文...",
        "💾 保存数据...",
        "🔢 生成向量（embedding）...",
        "🧠 训练主题模型（BERTopic）...",
        "✅ 完成！",
    ]

    progress_bar = st.progress(0, text="准备中...")
    status_text = st.empty()

    # 显示步骤预览
    steps_placeholder = st.empty()
    steps_placeholder.markdown(
        " → ".join([f'<span class="step-wait">{s}</span>' for s in steps]),
        unsafe_allow_html=True
    )

    # 更新第一步
    progress_bar.progress(10, text=steps[0])
    status_text.info(f"⏳ {steps[0]}")

    payload = {"query": query, "max_results": max_results}
    if sources:
        payload["sources"] = sources

    result = api_post(
        "/api/search",
        payload,
        timeout=600
    )

    if result:
        progress_bar.progress(100, text="完成！")

        # 显示来源统计
        source_stats = result.get("source_stats")
        source_info = ""
        if source_stats:
            parts = [f"{name}: {count}篇" for name, count in source_stats.items()]
            source_info = f" | 来源: {', '.join(parts)}"
            st.session_state["source_stats"] = source_stats

        status_text.success(
            f"✅ {result.get('message', '搜索完成')} "
            f"（{result.get('paper_count', 0)} 篇论文，"
            f"{result.get('topic_count', 0)} 个主题{source_info}）"
        )
        steps_placeholder.empty()
        return True
    else:
        progress_bar.empty()
        status_text.empty()
        steps_placeholder.empty()
        return False


def init_chat_session() -> str | None:
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/chat/init",
            json={"session_id": None, "topic_id": None},
            timeout=30
        )
        if r.status_code == 200:
            return r.json().get("session_id")
        else:
            return None
    except Exception:
        return None


# ========== 侧边栏 ==========

with st.sidebar:
    st.title("📚 AI 论文助手")
    st.markdown("---")

    api_online = check_api_health()
    if api_online:
        st.success("🟢 后端服务在线")
    else:
        st.error("🔴 后端服务离线")
        st.info("启动: `cd src && python main.py`")

    st.markdown("---")

    # 检查是否已有数据（使用缓存减少 rerun 延迟）
    status_data = cached_api_get("/api/search/status") if api_online else None
    has_data = status_data.get("has_data", False) if status_data else False

    if has_data:
        current_query = status_data.get("current_query", "")
        paper_count = status_data.get("paper_count", 0)
        topic_count = status_data.get("topic_count", 0)
        st.success(f"📊 已有数据")
        if current_query and current_query != "(历史数据)":
            st.caption(f"关键词: `{current_query}`")
        if paper_count:
            st.caption(f"论文: {paper_count} 篇 | 主题: {topic_count} 个")

        # 显示各来源论文统计（使用缓存）
        if api_online:
            source_stats_data = cached_api_get("/api/search/source-stats")
            if source_stats_data and source_stats_data.get("source_stats"):
                stats = source_stats_data["source_stats"]
                if stats:
                    source_parts = [f"{name}: {count}篇" for name, count in stats.items()]
                    st.caption(f"📡 来源: {' | '.join(source_parts)}")

        if st.button("🔄 重新搜索", use_container_width=True):
            st.session_state.pop("search_done", None)
            st.session_state.pop("session_id", None)
            st.session_state.pop("messages", None)
            st.session_state.pop("sunburst_data", None)
            st.session_state.pop("trend_data", None)
            st.rerun()
    else:
        st.info("💡 请在搜索框中输入关键词开始")

    st.markdown("---")
    st.markdown("### 📖 使用说明")
    st.markdown("""
1. 在搜索框输入研究关键词
2. 等待系统在线抓取并处理
3. 探索主题地图和趋势分析
4. 与 AI 对话深入了解
    """)

    st.markdown("---")
    st.caption("Powered by arXiv + BERTopic + Plotly")


# ========== 主界面 ==========

st.markdown('<div class="main-header">📚 AI 科研文献助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">输入研究关键词，系统自动在线抓取论文并分析</div>', unsafe_allow_html=True)

# ========== 搜索区域（始终显示在顶部）==========

search_col, btn_col = st.columns([5, 1])
with search_col:
    query_input = st.text_input(
        "搜索关键词",
        placeholder="例如: transformer attention mechanism / large language model / BERT",
        label_visibility="collapsed",
        key="query_input"
    )
with btn_col:
    max_results = st.session_state.get("max_results", 50)
    search_btn = st.button("🔍 搜索", use_container_width=True, type="primary")

# 高级选项（折叠）
with st.expander("⚙️ 高级选项"):
    max_results = st.slider("抓取论文数量", min_value=10, max_value=200, value=50, step=10,
                            help="数量越多，主题分析越准确，但耗时也越长")
    st.session_state["max_results"] = max_results

    # 数据源选择
    st.markdown("**数据源选择**")
    use_arxiv = st.checkbox("arXiv", value=True, key="use_arxiv", help="arXiv 预印本数据库")
    use_openalex = st.checkbox("OpenAlex", value=False, key="use_openalex", help="OpenAlex 开放学术数据库")
    selected_sources = []
    if use_arxiv:
        selected_sources.append("arxiv")
    if use_openalex:
        selected_sources.append("openalex")
    if not selected_sources:
        selected_sources = ["arxiv"]
    st.session_state["selected_sources"] = selected_sources

# 执行搜索
if search_btn and query_input.strip():
    with st.container():
        st.markdown("---")
        sources = st.session_state.get("selected_sources", ["arxiv"])
        st.markdown(f"**正在搜索**: `{query_input.strip()}` | 数据源: {', '.join(sources)}")
        success = do_search(query_input.strip(), max_results, sources=sources)
        if success:
            st.session_state["search_done"] = True
            st.session_state["current_query"] = query_input.strip()
            # 重置依赖数据的状态
            st.session_state.pop("session_id", None)
            st.session_state.pop("messages", None)
            st.session_state.pop("sunburst_data", None)
            st.session_state.pop("trend_data", None)
            # 清除 API 缓存，确保侧边栏立即显示最新数据
            cached_api_get.clear()
            time.sleep(1)
            st.rerun()
elif search_btn and not query_input.strip():
    st.warning("请输入搜索关键词")

# ========== 数据展示区域（搜索完成后显示）==========

search_done = st.session_state.get("search_done", False) or has_data

if not search_done:
    # 未搜索时显示引导界面
    st.markdown("---")
    st.markdown(
        '<div class="search-hint">👆 在上方输入关键词，系统将自动从 arXiv 在线抓取论文并分析</div>',
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 主题地图")
        st.markdown("搜索后自动生成 Sunburst 旭日图，展示各研究主题的论文分布")
    with col2:
        st.markdown("### 📈 研究趋势")
        st.markdown("折线图展示各主题在不同年份的热度变化，识别新兴方向")
    with col3:
        st.markdown("### 💬 AI 对话")
        st.markdown("基于 RAG 的语义问答，每个回答都有论文来源支撑")

else:
    # 搜索完成后显示三个标签页
    st.markdown("---")
    tab_map, tab_trend, tab_chat = st.tabs([
        "📊 主题地图",
        "📈 研究趋势",
        "💬 AI 对话",
    ])

    # ============================================================
    # 标签页 1：主题地图
    # ============================================================
    with tab_map:
        st.subheader("📊 研究主题地图")
        st.caption("旭日图展示各研究主题的论文分布。选择主题可查看该主题下的论文列表。")

        col_chart, col_papers = st.columns([3, 2])

        with col_chart:
            if "sunburst_data" not in st.session_state:
                st.session_state.sunburst_data = None
            if "selected_topic_id" not in st.session_state:
                st.session_state.selected_topic_id = None
            if "topic_options_list" not in st.session_state:
                st.session_state.topic_options_list = []
                st.session_state.topic_options_map = {}

            load_btn = st.button("🔄 加载主题地图", key="load_sunburst", use_container_width=True)

            # 自动加载或手动加载
            if load_btn or st.session_state.sunburst_data is None:
                with st.spinner("正在加载主题数据..."):
                    data = api_get("/api/topics/sunburst")
                    if data:
                        st.session_state.sunburst_data = data
                        # 预计算主题选项，避免每次 rerun 重新构建
                        opts = {}
                        for label, tid in zip(data["labels"][1:], data["topic_ids"][1:]):
                            if tid != -1:
                                key = f"Topic {tid}: {label.split(': ', 1)[-1] if ': ' in label else label}"
                                opts[key] = tid
                        st.session_state.topic_options_list = list(opts.keys())
                        st.session_state.topic_options_map = opts

            # 显示已加载的数据（独立于加载逻辑）
            data = st.session_state.sunburst_data
            if data and data.get("labels") and len(data["labels"]) > 1:
                # 提取主题数据（跳过根节点）
                topic_labels = data["labels"][1:]
                topic_values = data["values"][1:]
                total_papers = sum(topic_values)
                st.caption(f"已加载 {len(topic_labels)} 个主题，{total_papers} 篇论文")

                # 保存主题数量到 session_state（供趋势分析使用）
                st.session_state["actual_topic_count"] = len(topic_labels)

                # 使用水平柱状图展示主题分布
                import pandas as pd
                df_topics = pd.DataFrame({
                    "主题": topic_labels,
                    "论文数": topic_values
                }).sort_values("论文数", ascending=True)

                fig = go.Figure(go.Bar(
                    x=df_topics["论文数"],
                    y=df_topics["主题"],
                    orientation='h',
                    marker=dict(
                        color=df_topics["论文数"],
                        colorscale="Blues",
                    ),
                    text=df_topics["论文数"],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="研究主题论文分布",
                    xaxis_title="论文数量",
                    yaxis_title="",
                    height=max(300, len(topic_labels) * 50),
                    margin=dict(t=40, l=10, r=40, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(248,249,250,1)",
                )
                st.plotly_chart(fig, use_container_width=True)

                # 使用缓存的主题选项和 on_change 回调
                topic_options_list = st.session_state.topic_options_list
                topic_options_map = st.session_state.topic_options_map

                if topic_options_list:
                    st.markdown("**🔍 选择主题查看论文**")

                    def on_topic_change():
                        """selectbox 值变化时的回调，直接更新 session_state"""
                        sel = st.session_state.topic_selector
                        if sel and sel in st.session_state.topic_options_map:
                            st.session_state.selected_topic_id = st.session_state.topic_options_map[sel]
                            st.session_state.selected_topic_name = sel

                    st.selectbox(
                        "选择主题",
                        options=topic_options_list,
                        key="topic_selector",
                        label_visibility="collapsed",
                        on_change=on_topic_change,
                    )
                    # 确保首次加载时也设置选中的主题
                    current_sel = st.session_state.get("topic_selector")
                    if current_sel and current_sel in topic_options_map:
                        if st.session_state.selected_topic_id != topic_options_map[current_sel]:
                            st.session_state.selected_topic_id = topic_options_map[current_sel]
                            st.session_state.selected_topic_name = current_sel
            elif data:
                st.info("主题数据为空，可能论文数量不足以形成主题。请尝试增加抓取数量后重新搜索。")

        with col_papers:
            st.markdown("### 📄 主题论文列表")

            if st.session_state.get("selected_topic_id") is not None:
                topic_id = st.session_state.selected_topic_id
                topic_name = st.session_state.get("selected_topic_name", f"Topic {topic_id}")

                st.markdown(f"**当前主题**: `{topic_name}`")

                sort_col, page_col = st.columns(2)
                with sort_col:
                    sort_by = st.selectbox(
                        "排序方式",
                        options=["date", "relevance"],
                        format_func=lambda x: "📅 时间降序" if x == "date" else "🎯 相关性",
                        key="paper_sort"
                    )
                with page_col:
                    page_num = st.number_input("页码", min_value=1, value=1, step=1, key="paper_page")

                import json
                papers_data = cached_api_get(
                    f"/api/topics/{topic_id}/papers",
                    params_str=json.dumps({"page": page_num, "page_size": 10, "sort_by": sort_by})
                )

                if papers_data:
                    total = papers_data.get("total", 0)
                    papers = papers_data.get("papers", [])
                    page_size = papers_data.get("page_size", 10)
                    total_pages = max(1, (total + page_size - 1) // page_size)

                    st.caption(f"共 {total} 篇论文 | 第 {page_num}/{total_pages} 页")

                    for paper in papers:
                        title = paper.get("title", "Unknown")
                        with st.expander(f"📄 {title[:70]}{'...' if len(title) > 70 else ''}"):
                            authors = paper.get("authors", [])
                            published = paper.get("published", "")
                            abstract = paper.get("abstract", "")
                            pdf_url = paper.get("pdf_url", "")

                            st.markdown(f"**{title}**")
                            if authors:
                                st.caption(f"👤 {', '.join(authors[:3])}" + (" et al." if len(authors) > 3 else ""))
                            if published:
                                st.caption(f"📅 {published[:4]}")
                            if abstract:
                                st.markdown(f"> {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
                            if pdf_url:
                                st.markdown(f"[📥 下载 PDF]({pdf_url})")

                    # 相关主题推荐
                    st.markdown("---")
                    st.markdown("**🔗 相关主题推荐**")
                    similar_data = cached_api_get(
                        f"/api/topics/{topic_id}/similar",
                        params_str=json.dumps({"top_n": 3})
                    )
                    if similar_data and similar_data.get("similar_topics"):
                        for sim in similar_data["similar_topics"]:
                            sim_name = sim.get("topic_name", "Unknown")
                            sim_count = sim.get("paper_count", 0)
                            sim_score = sim.get("similarity", 0)
                            if st.button(
                                f"→ {sim_name} ({sim_count}篇, 相似度:{sim_score:.2f})",
                                key=f"sim_{sim['topic_id']}"
                            ):
                                st.session_state.selected_topic_id = sim["topic_id"]
                                st.session_state.selected_topic_name = sim_name
                                st.rerun()
            else:
                st.info("👈 请先加载主题地图，然后选择一个主题查看论文")

    # ============================================================
    # 标签页 2：研究趋势
    # ============================================================
    with tab_trend:
        st.subheader("📈 研究趋势分析")
        st.caption("展示各研究主题在不同年份的论文数量变化，帮助识别新兴和成熟研究方向。")

        # 根据实际主题数量动态设置 slider 上限
        actual_topic_count = st.session_state.get("actual_topic_count", 20)
        slider_max = max(3, actual_topic_count)  # 上限不超过实际主题数
        slider_default = min(8, slider_max)  # 默认值不超过上限

        top_n_topics = st.slider(
            "显示主题数量", min_value=1, max_value=slider_max, value=slider_default, step=1,
            help=f"当前共有 {actual_topic_count} 个主题"
        )

        load_trend_btn = st.button("🔄 加载趋势数据", key="load_trends")

        if "trend_data" not in st.session_state:
            st.session_state.trend_data = None

        if load_trend_btn:
            with st.spinner("正在计算研究趋势..."):
                data = api_get("/api/topics/trends", params={"top_n": top_n_topics})
                if data:
                    st.session_state.trend_data = data

        trend_data = st.session_state.trend_data

        if trend_data:
            years = trend_data.get("years", [])
            topics_series = trend_data.get("topics", [])
            trending = trend_data.get("trending", [])

            if not years or not topics_series:
                st.info("暂无趋势数据。论文数量可能不足，请尝试增加抓取数量后重新搜索。")
            else:
                if trending:
                    st.markdown("**🔥 近期增长最快的研究方向**")
                    badges_html = ""
                    for item in trending[:5]:
                        name = item.get("name", "")
                        rate = item.get("growth_rate", 1.0)
                        badges_html += f'<span class="trending-badge">↑{rate:.1f}x {name[:20]}</span>'
                    st.markdown(badges_html, unsafe_allow_html=True)
                    st.markdown("")

                fig = go.Figure()
                colors = px.colors.qualitative.Plotly

                for i, series in enumerate(topics_series):
                    name = series.get("name", f"Topic {i}")
                    counts = series.get("counts", [])
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=years, y=counts,
                        mode="lines+markers",
                        name=name[:30],
                        line=dict(width=2, color=color),
                        marker=dict(size=6, color=color),
                    ))

                fig.update_layout(
                    title="各研究主题论文数量年度变化",
                    xaxis_title="年份",
                    yaxis_title="论文数量",
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                    hovermode="x unified",
                    height=480,
                    margin=dict(t=50, r=200, b=50, l=60),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(248,249,250,1)",
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 查看原始数据表格"):
                    import pandas as pd
                    table_data = {"年份": years}
                    for series in topics_series:
                        table_data[series.get("name", "Unknown")[:25]] = series.get("counts", [])
                    df_display = pd.DataFrame(table_data).set_index("年份")
                    st.dataframe(df_display, use_container_width=True)
        else:
            st.info("点击「加载趋势数据」按钮开始分析。")

    # ============================================================
    # 标签页 3：AI 对话
    # ============================================================
    with tab_chat:
        st.subheader("💬 AI 文献问答")
        st.caption("基于 RAG 的语义问答，每个回答都有论文来源支撑。")

        # 初始化会话
        if "session_id" not in st.session_state:
            with st.spinner("正在初始化对话..."):
                session_id = init_chat_session()
                if session_id:
                    st.session_state.session_id = session_id
                    st.session_state.messages = []
                else:
                    st.warning("对话服务初始化失败，请确认搜索已完成。")

        # 显示对话历史
        if "messages" in st.session_state:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        content = msg["content"]
                        st.markdown(content.get("answer", "无回答"))
                        sources = content.get("sources", [])
                        if sources:
                            with st.expander(f"📚 查看 {len(sources)} 篇参考论文"):
                                for src in sources[:5]:
                                    st.markdown(f"**{src.get('title', 'Unknown')}**")
                                    if src.get("ai_summary"):
                                        st.markdown(f"> {src['ai_summary'][:150]}...")
                                    if src.get("pdf_url"):
                                        st.markdown(f"[PDF]({src['pdf_url']})")
                                    st.markdown("---")

        # 输入框
        if "session_id" in st.session_state:
            if prompt := st.chat_input("输入你的问题（支持中英文）..."):
                with st.chat_message("user"):
                    st.write(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()

                    progress_placeholder.progress(25, text="阶段 1/4: 理解问题...")
                    status_placeholder.markdown("⏳ 正在检索相关论文...")

                    result = None
                    try:
                        progress_placeholder.progress(50, text="阶段 2/4: 检索文档...")
                        r = requests.post(
                            f"{API_BASE_URL}/api/chat/message",
                            json={"session_id": st.session_state.session_id, "message": prompt},
                            timeout=90
                        )
                        progress_placeholder.progress(75, text="阶段 3/4: 生成回答...")
                        if r.status_code == 200:
                            result = r.json()
                        elif r.status_code == 404:
                            status_placeholder.error("会话已过期，请刷新页面")
                        else:
                            status_placeholder.error(f"请求失败: HTTP {r.status_code}")
                    except requests.exceptions.Timeout:
                        status_placeholder.error("请求超时，请稍后重试")
                    except Exception as e:
                        status_placeholder.error(f"请求失败: {e}")

                    progress_placeholder.empty()
                    status_placeholder.empty()

                    if result:
                        answer = result.get("answer", "无回答")
                        sources = result.get("sources", [])
                        st.markdown(answer)
                        if sources:
                            with st.expander(f"📚 查看 {len(sources)} 篇参考论文"):
                                for src in sources[:5]:
                                    st.markdown(f"**{src.get('title', 'Unknown')}**")
                                    authors = src.get("authors", [])
                                    if authors:
                                        st.caption(f"👤 {', '.join(authors[:2])}")
                                    if src.get("ai_summary"):
                                        st.markdown(f"> {src['ai_summary'][:200]}...")
                                    if src.get("pdf_url"):
                                        st.markdown(f"[📥 PDF]({src['pdf_url']})")
                                    st.markdown("---")
                        st.session_state.messages.append({"role": "assistant", "content": result})
                    else:
                        st.error("未能获取回答，请稍后重试")

            if st.session_state.get("messages"):
                if st.button("🗑️ 清空对话历史", key="clear_chat"):
                    st.session_state.messages = []
                    st.rerun()
        else:
            st.info("请先完成搜索以初始化对话功能。")

# ========== 底部 ==========
st.markdown(
    '<div class="footer">Powered by arXiv + BERTopic + Plotly + RAG</div>',
    unsafe_allow_html=True
)
