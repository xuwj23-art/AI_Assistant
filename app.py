"""
Streamlit frontend — search-driven version

Flow:
  1. User enters keywords in the search box
  2. Click Search → backend fetches papers → embeds → trains topic model
  3. Results displayed in three tabs: Topic Distribution / Trends / AI Chat
"""
import streamlit as st
import requests
import time
import urllib.parse

import plotly.graph_objects as go
import plotly.express as px

# ========== 配置 ==========

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Research Assistant",
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

@st.cache_data(ttl=8, show_spinner=False)
def check_api_health() -> bool:
    """Health check, cached 8 s to avoid hammering the API on every rerun."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def api_get(path: str, params: dict = None, timeout: int = 30):
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 503:
            return None  # no data yet, silent
        else:
            st.error(f"API error {r.status_code}: {r.text[:200]}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach backend. Start it with: `cd src && python main.py`")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out, please try again")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


@st.cache_data(ttl=10, show_spinner=False)
def cached_api_get(path: str, params_str: str = "", timeout: int = 30):
    """Cached API GET — avoids duplicate calls on Streamlit rerun."""
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
            st.error("Session expired — please refresh the page")
            return None
        elif r.status_code == 409:
            st.warning("Processing in progress, please wait...")
            return None
        else:
            st.error(f"Request failed: HTTP {r.status_code} — {r.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out (paper fetching can take a while, please retry)")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def do_search(query: str, max_results: int, sources: list = None) -> bool:
    """Run the search pipeline; return True on success."""
    steps = [
        "🌐 Fetching papers online...",
        "💾 Saving data...",
        "🔢 Generating embeddings...",
        "🧠 Training topic model (BERTopic)...",
        "✅ Done!",
    ]

    progress_bar = st.progress(0, text="Preparing...")
    status_text = st.empty()

    steps_placeholder = st.empty()
    steps_placeholder.markdown(
        " → ".join([f'<span class="step-wait">{s}</span>' for s in steps]),
        unsafe_allow_html=True
    )

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
        progress_bar.progress(100, text="Done!")

        source_stats = result.get("source_stats")
        source_info = ""
        if source_stats:
            parts = [f"{name}: {count}" for name, count in source_stats.items()]
            source_info = f" | Sources: {', '.join(parts)}"
            st.session_state["source_stats"] = source_stats

        status_text.success(
            f"✅ {result.get('message', 'Search complete')} "
            f"({result.get('paper_count', 0)} papers, "
            f"{result.get('topic_count', 0)} topics{source_info})"
        )
        steps_placeholder.empty()
        return True
    else:
        progress_bar.empty()
        status_text.empty()
        steps_placeholder.empty()
        return False


def init_chat_session(topic_id: int | None = None) -> str | None:
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/chat/init",
            json={"session_id": None, "topic_id": topic_id},
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
    st.title("📚 AI Research Assistant")
    st.markdown("---")

    api_online = check_api_health()
    if api_online:
        st.success("🟢 Backend Online")
    else:
        st.error("🔴 Backend Offline")
        st.info("Start: `cd src && python main.py`")

    st.markdown("---")

    status_data = cached_api_get("/api/search/status") if api_online else None
    has_data = status_data.get("has_data", False) if status_data else False

    if has_data:
        current_query = status_data.get("current_query", "")
        paper_count = status_data.get("paper_count", 0)
        topic_count = status_data.get("topic_count", 0)
        # Store total paper count in session state so the Topic Distribution
        # tab can reference it when computing the noise/outlier count.
        st.session_state["_total_paper_count"] = paper_count
        st.success("📊 Data Loaded")
        if current_query and current_query != "(historical data)":
            st.caption(f"Query: `{current_query}`")
        if paper_count:
            st.caption(f"Papers: {paper_count} | Topics: {topic_count}")

        if api_online:
            source_stats_data = cached_api_get("/api/search/source-stats")
            if source_stats_data and source_stats_data.get("source_stats"):
                stats = source_stats_data["source_stats"]
                if stats:
                    source_parts = [f"{name}: {count}" for name, count in stats.items()]
                    st.caption(f"📡 Sources: {' | '.join(source_parts)}")

        if st.button("🔄 New Search", use_container_width=True):
            st.session_state.pop("search_done", None)
            st.session_state.pop("session_id", None)
            st.session_state.pop("messages", None)
            st.session_state.pop("sunburst_data", None)
            st.session_state.pop("trend_data", None)
            st.rerun()
    else:
        st.info("💡 Enter keywords in the search box to begin")

    # ========== Search History ==========
    st.markdown("---")
    st.markdown("### 📂 Search History")

    if api_online:
        history_data = cached_api_get("/api/history")
        datasets = history_data.get("datasets", []) if history_data else []

        if datasets:
            options_map = {}
            for ds in datasets:
                label = (
                    f"{ds['query']}"
                    f"  ({ds['paper_count']} papers · {ds['topic_count']} topics"
                    f" · {ds['modified_str'][5:]})"
                )
                options_map[label] = ds["safe_query"]

            option_labels = list(options_map.keys())

            current_safe = (
                current_query.replace(" ", "_")[:30]
                if has_data and current_query and current_query != "(historical data)"
                else None
            )
            default_idx = 0
            if current_safe:
                for i, sq in enumerate(options_map.values()):
                    if sq == current_safe:
                        default_idx = i
                        break

            selected_label = st.selectbox(
                "Select dataset",
                options=option_labels,
                index=default_idx,
                key="history_selector",
                label_visibility="collapsed",
                help="Select a dataset then click Switch",
            )

            selected_safe_query = options_map[selected_label]

            if st.button("📂 Switch to This Dataset", use_container_width=True, key="btn_switch_history"):
                with st.spinner("Switching..."):
                    result = api_post(
                        "/api/history/switch",
                        {"safe_query": selected_safe_query},
                        timeout=15,
                    )
                if result:
                    st.success(result.get("message", "Switched successfully"))
                    for key in ["session_id", "messages", "sunburst_data", "trend_data"]:
                        st.session_state.pop(key, None)
                    st.session_state["search_done"] = True
                    st.session_state["current_query"] = ds["query"]
                    cached_api_get.clear()
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.caption("No history yet")

    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
1. Enter a research keyword in the search box
2. Wait for the system to fetch and process papers
3. Explore topic distribution and trend analysis
4. Chat with AI for deeper insights
    """)

    st.markdown("---")
    st.caption("Powered by arXiv + BERTopic + Plotly")


# ========== 主界面 ==========

st.markdown('<div class="main-header">📚 AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter research keywords and the system will fetch and analyze papers automatically</div>', unsafe_allow_html=True)

# ========== 搜索区域（始终显示在顶部）==========

search_col, btn_col = st.columns([5, 1])
with search_col:
    query_input = st.text_input(
        "Search keywords",
        placeholder="e.g. transformer attention mechanism / large language model / BERT",
        label_visibility="collapsed",
        key="query_input"
    )
with btn_col:
    max_results = st.session_state.get("max_results", 50)
    search_btn = st.button("🔍 Search", use_container_width=True, type="primary")

with st.expander("⚙️ Advanced Options"):
    max_results = st.slider("Max papers to fetch", min_value=10, max_value=200, value=50, step=10,
                            help="More papers improve topic quality but take longer")
    st.session_state["max_results"] = max_results

    st.markdown("**Data Sources**")
    use_arxiv = st.checkbox("arXiv", value=True, key="use_arxiv", help="arXiv preprint repository")
    use_openalex = st.checkbox("OpenAlex", value=False, key="use_openalex", help="OpenAlex open academic database")
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
        st.markdown(f"**Searching**: `{query_input.strip()}` | Sources: {', '.join(sources)}")
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
    st.warning("Please enter a search keyword")

# ========== 数据展示区域（搜索完成后显示）==========

search_done = st.session_state.get("search_done", False)

if not search_done:
    st.markdown("---")
    if has_data:
        st.info("💡 Local data detected — select a dataset in the sidebar and click Switch, or search for new keywords.")
    else:
        st.markdown(
            '<div class="search-hint">👆 Enter keywords above to fetch and analyze papers from arXiv</div>',
            unsafe_allow_html=True
        )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Topic Distribution")
        st.markdown("Automatically generates a topic distribution chart showing paper counts per research direction")
    with col2:
        st.markdown("### 📈 Research Trends")
        st.markdown("Line chart showing paper counts per topic over the years to identify emerging directions")
    with col3:
        st.markdown("### 💬 AI Chat")
        st.markdown("RAG-powered Q&A — every answer is backed by cited paper sources")

else:
    # 搜索完成后显示三个标签页
    st.markdown("---")
    tab_map, tab_trend, tab_chat = st.tabs([
        "📊 Topic Distribution",
        "📈 Research Trends",
        "💬 AI Chat",
    ])

    # ============================================================
    # 标签页 1：主题分布
    # ============================================================
    with tab_map:
        st.subheader("📊 Research Topic Distribution")
        st.caption("Bar chart showing paper counts per research topic. Select a topic below to browse papers.")

        col_chart, col_papers = st.columns([3, 2])

        with col_chart:
            if "sunburst_data" not in st.session_state:
                st.session_state.sunburst_data = None
            if "selected_topic_id" not in st.session_state:
                st.session_state.selected_topic_id = None
            if "topic_options_list" not in st.session_state:
                st.session_state.topic_options_list = []
                st.session_state.topic_options_map = {}

            load_btn = st.button("🔄 Load Topic Distribution", key="load_sunburst", use_container_width=True)

            # 校验缓存数据有效性（父节点引用必须在 ids 中）
            _cached = st.session_state.sunburst_data
            if _cached:
                _ids_set = set(_cached.get("ids", []))
                _bad = any(
                    p != "" and p not in _ids_set
                    for p in _cached.get("parents", [])
                )
                if _bad:
                    st.session_state.sunburst_data = None

            # 自动加载或手动加载
            if load_btn or st.session_state.sunburst_data is None:
                with st.spinner("Loading topic data..."):
                    data = api_get("/api/topics/sunburst")
                    if data:
                        st.session_state.sunburst_data = data
                        # 预计算主题选项：只保留叶子主题（topic_id >= 0），
                        # 排除根节点（-1）和内部聚类节点（-2）
                        opts = {}
                        for label, tid in zip(data["labels"][1:], data["topic_ids"][1:]):
                            if tid >= 0:  # 只取真实叶子主题
                                key = f"Topic {tid}: {label.split(': ', 1)[-1] if ': ' in label else label}"
                                opts[key] = tid
                        st.session_state.topic_options_list = list(opts.keys())
                        st.session_state.topic_options_map = opts

            # 显示已加载的数据（独立于加载逻辑）
            data = st.session_state.sunburst_data
            if data and data.get("labels") and len(data["labels"]) > 1:
                # Only count real leaf topics (topic_id >= 0); exclude root (-1) and
                # internal BERTopic cluster nodes (-2) so counts and Trends slider are accurate.
                leaf_count = sum(1 for tid in data["topic_ids"][1:] if tid >= 0)
                leaf_papers = sum(
                    v for v, tid in zip(data["values"][1:], data["topic_ids"][1:])
                    if tid >= 0
                )
                total_fetched = st.session_state.get("_total_paper_count", 0)
                noise_count = total_fetched - leaf_papers if total_fetched > leaf_papers else 0
                if noise_count > 0:
                    st.caption(
                        f"Loaded {leaf_count} topics, {leaf_papers}/{total_fetched} papers"
                        f" ({noise_count} outliers excluded by HDBSCAN)"
                    )
                else:
                    st.caption(f"Loaded {leaf_count} topics, {leaf_papers} papers")
                st.session_state["actual_topic_count"] = leaf_count

                # ── 水平柱状图 ────────────────────────────────────────
                # 只展示叶子主题（topic_id >= 0），过滤掉根节点和内部聚类节点
                leaf_labels = [
                    lbl for lbl, tid in zip(data["labels"][1:], data["topic_ids"][1:])
                    if tid >= 0
                ]
                leaf_values = [
                    v for v, tid in zip(data["values"][1:], data["topic_ids"][1:])
                    if tid >= 0
                ]

                palette = px.colors.qualitative.Plotly
                bar_colors = [palette[i % len(palette)] for i in range(len(leaf_labels))]

                fig = go.Figure(go.Bar(
                    x=leaf_values,
                    y=leaf_labels,
                    orientation="h",
                    marker=dict(color=bar_colors),
                    hovertemplate="<b>%{y}</b><br>Papers: %{x}<extra></extra>",
                ))
                fig.update_layout(
                    title=dict(
                        text="Research Topic Distribution",
                        x=0.5, xanchor="center", font=dict(size=15)
                    ),
                    xaxis_title="Paper Count",
                    yaxis=dict(autorange="reversed"),
                    height=max(300, len(leaf_labels) * 50 + 80),
                    margin=dict(t=50, l=10, r=20, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

                # 使用缓存的主题选项和 on_change 回调
                topic_options_list = st.session_state.topic_options_list
                topic_options_map = st.session_state.topic_options_map

                if topic_options_list:
                    st.markdown("**🔍 Select a Topic**")

                    def on_topic_change():
                        sel = st.session_state.topic_selector
                        if sel and sel in st.session_state.topic_options_map:
                            st.session_state.selected_topic_id = st.session_state.topic_options_map[sel]
                            st.session_state.selected_topic_name = sel

                    st.selectbox(
                        "Select topic",
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
                st.info("No topic data — there may not be enough papers. Try increasing the fetch count and searching again.")

        with col_papers:
            st.markdown("### 📄 Topic Papers")

            if st.session_state.get("selected_topic_id") is not None:
                topic_id = st.session_state.selected_topic_id
                topic_name = st.session_state.get("selected_topic_name", f"Topic {topic_id}")

                st.markdown(f"**Current Topic**: `{topic_name}`")

                # 初始化页码 session state（只设置一次，避免与 widget value= 参数冲突）
                if "paper_page" not in st.session_state:
                    st.session_state["paper_page"] = 1

                sort_col, page_col = st.columns(2)
                with sort_col:
                    def _on_sort_change():
                        st.session_state["paper_page"] = 1

                    sort_by = st.selectbox(
                        "Sort by",
                        options=["relevance", "date"],
                        format_func=lambda x: (
                            "🎯 Topic Relevance"
                            if x == "relevance"
                            else "📅 Date (Newest)"
                        ),
                        key="paper_sort",
                        on_change=_on_sort_change,
                        help="Topic Relevance = cosine similarity from paper embedding to topic centroid",
                    )
                with page_col:
                    page_num = st.number_input("Page", min_value=1, step=1, key="paper_page")

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

                    st.caption(f"{total} papers | Page {page_num}/{total_pages}")

                    for rank_idx, paper in enumerate(papers, 1):
                        title = paper.get("title", "Unknown")
                        rel = paper.get("relevance")
                        # 在折叠标题里加一个代表性徽章
                        rel_badge = f"  🎯{rel:.2f}" if (sort_by == "relevance" and rel is not None) else ""
                        with st.expander(
                            f"📄 #{rank_idx} {title[:65]}{'...' if len(title) > 65 else ''}{rel_badge}"
                        ):
                            authors = paper.get("authors", [])
                            published = paper.get("published", "")
                            abstract = paper.get("abstract", "")
                            pdf_url = paper.get("pdf_url", "")
                            paper_id = paper.get("id", "")

                            st.markdown(f"**{title}**")
                            meta_caps = []
                            if authors:
                                meta_caps.append(
                                    f"👤 {', '.join(authors[:3])}"
                                    + (" et al." if len(authors) > 3 else "")
                                )
                            if published:
                                meta_caps.append(f"📅 {published[:4]}")
                            if rel is not None:
                                meta_caps.append(f"🎯 Relevance {rel:.3f}")
                            if meta_caps:
                                st.caption(" · ".join(meta_caps))
                            if abstract:
                                st.markdown(f"> {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
                            if pdf_url:
                                col_pdf1, col_pdf2 = st.columns(2)
                                with col_pdf1:
                                    st.markdown(f"[🔗 View Online]({pdf_url})")
                                with col_pdf2:
                                    if paper_id:
                                        encoded_pid = urllib.parse.quote(paper_id, safe='')
                                        st.markdown(
                                            f"[📥 Download]"
                                            f"({API_BASE_URL}/api/papers/pdf?id={encoded_pid})"
                                        )

                    st.markdown("---")
                    st.markdown("**🔗 Related Topics**")
                    similar_data = cached_api_get(
                        f"/api/topics/{topic_id}/similar",
                        params_str=json.dumps({"top_n": 3})
                    )
                    if similar_data and similar_data.get("similar_topics"):
                        method = similar_data.get("method", "cosine")
                        method_tag = "🧭 Cosine" if method == "cosine" else "📝 Keyword"
                        st.caption(f"Similarity method: {method_tag}")
                        for sim in similar_data["similar_topics"]:
                            sim_name = sim.get("topic_name", "Unknown")
                            sim_count = sim.get("paper_count", 0)
                            sim_score = sim.get("similarity", 0)
                            if st.button(
                                f"→ {sim_name} ({sim_count} papers, sim: {sim_score:.2f})",
                                key=f"sim_{sim['topic_id']}"
                            ):
                                st.session_state.selected_topic_id = sim["topic_id"]
                                st.session_state.selected_topic_name = sim_name
                                st.rerun()
            else:
                st.info("👈 Load the topic distribution first, then select a topic to view papers")

    # ============================================================
    # 标签页 2：研究趋势
    # ============================================================
    with tab_trend:
        st.subheader("📈 Research Trend Analysis")
        st.caption("Shows paper counts per topic over the years to identify emerging and mature research directions.")

        # Prefer the leaf-topic count set when Topic Distribution is loaded;
        # fall back to topic_count from /api/search/status (available as soon as
        # search completes, even before the Topic Distribution tab is visited).
        actual_topic_count = (
            st.session_state.get("actual_topic_count")
            or (status_data.get("topic_count") if status_data else None)
            or 20
        )
        slider_max = max(3, actual_topic_count)
        slider_default = min(8, slider_max)

        top_n_topics = st.slider(
            "Number of topics to show", min_value=1, max_value=slider_max, value=slider_default, step=1,
            help=f"{actual_topic_count} topics available"
        )

        load_trend_btn = st.button("🔄 Load Trend Data", key="load_trends")

        if "trend_data" not in st.session_state:
            st.session_state.trend_data = None

        if load_trend_btn:
            with st.spinner("Computing research trends..."):
                data = api_get("/api/topics/trends", params={"top_n": top_n_topics})
                if data:
                    st.session_state.trend_data = data

        trend_data = st.session_state.trend_data

        if trend_data:
            years = trend_data.get("years", [])
            topics_series = trend_data.get("topics", [])
            trending = trend_data.get("trending", [])

            if not years or not topics_series:
                st.info("No trend data available. There may not be enough papers — try increasing the fetch count.")
            else:
                if trending:
                    st.markdown("**🔥 Fastest Growing Research Areas**")
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
                    title="Annual Paper Count by Research Topic",
                    xaxis_title="Year",
                    yaxis_title="Paper Count",
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                    hovermode="x unified",
                    height=480,
                    margin=dict(t=50, r=200, b=50, l=60),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(248,249,250,1)",
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 View Raw Data Table"):
                    import pandas as pd
                    table_data = {"Year": years}
                    for series in topics_series:
                        table_data[series.get("name", "Unknown")[:25]] = series.get("counts", [])
                    df_display = pd.DataFrame(table_data).set_index("Year")
                    st.dataframe(df_display, use_container_width=True)
        else:
            st.info('Click "Load Trend Data" to start analysis.')

    # ============================================================
    # 标签页 3：AI 对话
    # ============================================================
    with tab_chat:
        # ---------- 头部：标题 + 模型徽章 ----------
        head_col, badge_col = st.columns([5, 2])
        with head_col:
            st.subheader("💬 AI Literature Q&A")
            st.caption("RAG-powered multi-turn chat: retrieve related papers → generate cited answers with LLM.")
        with badge_col:
            llm_status = cached_api_get("/api/chat/llm-status") or {}
            rag_provider = llm_status.get("rag_provider", "local")
            rag_model = llm_status.get("rag_model", "t5-small")
            if rag_provider == "deepseek":
                st.markdown(
                    f"<div style='text-align:right;padding-top:0.3em'>"
                    f"<span style='background:#10b981;color:white;padding:4px 10px;"
                    f"border-radius:12px;font-size:0.8em'>🚀 {rag_model}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:right;padding-top:0.3em'>"
                    f"<span style='background:#94a3b8;color:white;padding:4px 10px;"
                    f"border-radius:12px;font-size:0.8em'>🔌 Local ({rag_model})</span></div>",
                    unsafe_allow_html=True,
                )

        # ---------- 初始化会话 ----------
        if "session_id" not in st.session_state:
            with st.spinner("Initializing chat session..."):
                _tid = st.session_state.get("selected_topic_id")
                session_id = init_chat_session(topic_id=_tid)
                if session_id:
                    st.session_state.session_id = session_id
                    st.session_state.messages = []
                else:
                    st.warning("Chat initialization failed — please ensure a search has been completed.")

        # ---------- 工具栏：清空 / 新会话 / 历史信息 ----------
        if "session_id" in st.session_state:
            tool_col1, tool_col2, tool_col3 = st.columns([2, 2, 3])
            with tool_col1:
                if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True,
                             help="Delete all messages in the current session (session ID is kept)"):
                    try:
                        requests.delete(
                            f"{API_BASE_URL}/api/chat/history",
                            params={"session_id": st.session_state.session_id},
                            timeout=10,
                        )
                    except Exception:
                        pass
                    st.session_state.messages = []
                    st.rerun()
            with tool_col2:
                if st.button("🔄 New Session", key="new_chat", use_container_width=True,
                             help="Create a new session ID and start a fresh conversation"):
                    new_sid = init_chat_session()
                    if new_sid:
                        st.session_state.session_id = new_sid
                        st.session_state.messages = []
                        st.rerun()
            with tool_col3:
                msg_count = len(st.session_state.get("messages", []))
                st.caption(
                    f"`session: {st.session_state.session_id[:8]}…` · {msg_count} messages"
                )

        # ---------- 示例问题（仅在空会话显示） ----------
        if "session_id" in st.session_state and not st.session_state.get("messages"):
            current_query = st.session_state.get("current_query", "this topic")
            example_questions = [
                f"What are the main approaches in {current_query}?",
                f"What are the key research directions in {current_query}?",
                "Compare the methods in the top 3 papers",
                "Which paper has the most novel contribution and why?",
            ]
            st.markdown("##### 💡 Try These Questions")
            ex_cols = st.columns(2)
            for i, q in enumerate(example_questions):
                with ex_cols[i % 2]:
                    if st.button(q, key=f"example_{i}", use_container_width=True):
                        st.session_state["pending_prompt"] = q
                        st.rerun()

        # ---------- 历史消息渲染 ----------
        import re as _re

        def _filter_cited_sources(answer: str, sources: list) -> list:
            """
            根据答案中出现的 [N] 编号筛选实际被引用的来源。
            - 没有任何 [N] 编号 → 返回空（隐藏来源面板）
            - 否则只保留答案中实际出现的那几个编号对应的论文
            """
            if not sources or not answer:
                return []
            cited_nums = set()
            for m in _re.finditer(r"\[(\d{1,2})\]", answer):
                try:
                    cited_nums.add(int(m.group(1)))
                except ValueError:
                    pass
            if not cited_nums:
                return []  # 答案里没有 [N] 引用 → 不展示
            # 只保留 1..len(sources) 范围内被引用的
            return [
                s for i, s in enumerate(sources, 1) if i in cited_nums
            ]

        def _render_sources(sources: list, label_prefix: str = "📚 View") -> None:
            if not sources:
                return
            with st.expander(f"{label_prefix} {len(sources)} cited paper(s)"):
                for idx, src in enumerate(sources, 1):
                    st.markdown(f"**[{idx}] {src.get('title', 'Unknown')}**")
                    authors = src.get("authors", [])
                    rel = src.get("relevance")
                    meta_parts = []
                    if authors:
                        meta_parts.append(
                            f"👤 {', '.join(authors[:2])}"
                            + (" et al." if len(authors) > 2 else "")
                        )
                    if rel is not None:
                        meta_parts.append(f"🎯 {rel:.3f}")
                    if meta_parts:
                        st.caption(" · ".join(meta_parts))
                    if src.get("ai_summary"):
                        st.markdown(f"> {src['ai_summary'][:280]}...")
                    pdf_link = src.get("pdf_url")
                    pid = src.get("id")
                    if pdf_link or pid:
                        link_parts = []
                        if pdf_link:
                            link_parts.append(f"[🔗 PDF]({pdf_link})")
                        if pid:
                            encoded_pid = urllib.parse.quote(pid, safe='')
                            link_parts.append(
                                f"[📥 Download]"
                                f"({API_BASE_URL}/api/papers/pdf?id={encoded_pid})"
                            )
                        st.markdown(" · ".join(link_parts))
                    st.markdown("---")

        def _model_badge(model_used: str) -> None:
            if model_used == "deepseek":
                st.caption(f"🚀 Generated by {rag_model}")
            elif model_used == "deepseek_meta":
                st.caption(f"💬 {rag_model} · Conversational mode (no retrieval)")
            elif model_used == "deepseek_history":
                st.caption(f"📜 {rag_model} · Follow-up answer based on history (no new retrieval)")
            elif model_used == "local":
                st.caption("🔌 Generated by local model")
            elif model_used == "local_meta":
                st.caption("💬 Local conversational mode (no LLM)")

        if "messages" in st.session_state:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        content = msg["content"]
                        used = content.get("model_used", "local")
                        _model_badge(used)
                        ans = content.get("answer", "No answer")
                        ts = content.get("topic_scope")
                        if ts:
                            st.caption(f"🎯 Search scope: {ts}")
                        rq = content.get("rewritten_query")
                        if rq:
                            st.caption(f"🔄 Query rewritten to: `{rq}`")
                        st.markdown(ans)
                        # 元问题 / 历史承上 / 本地 meta 不显示来源；其它按 [N] 筛选
                        if used not in ("deepseek_meta", "local_meta", "deepseek_history", "none"):
                            cited = _filter_cited_sources(ans, content.get("sources", []))
                            _render_sources(cited)

        # ---------- 输入框 + 处理 ----------
        if "session_id" in st.session_state:
            user_prompt = st.chat_input("Ask a question (follow-up supported)...")
            # 支持示例按钮触发
            pending = st.session_state.pop("pending_prompt", None)
            prompt = user_prompt or pending

            if prompt:
                # 立即把 user 消息加入历史并重绘，让用户立刻看到自己的提问
                st.session_state.messages.append({"role": "user", "content": prompt})
                # 标记"正在等待回复"，触发后续处理
                st.session_state["_pending_query"] = prompt
                st.rerun()

            # 处理等待中的请求（在 user 消息之后渲染，再触发一次 rerun 把 assistant 消息收编）
            pending_q = st.session_state.pop("_pending_query", None)
            if pending_q:
                with st.chat_message("assistant"):
                    status_placeholder = st.empty()
                    status_placeholder.markdown("🔍 Retrieving papers → 🧠 Generating answer...")

                    result = None
                    try:
                        r = requests.post(
                            f"{API_BASE_URL}/api/chat/message",
                            json={
                                "session_id": st.session_state.session_id,
                                "message": pending_q,
                            },
                            timeout=180,
                        )
                        if r.status_code == 200:
                            result = r.json()
                        elif r.status_code == 404:
                            status_placeholder.error("Session expired — click 🔄 New Session")
                        else:
                            status_placeholder.error(
                                f"Request failed: HTTP {r.status_code} — {r.text[:200]}"
                            )
                    except requests.exceptions.Timeout:
                        status_placeholder.error("Request timed out (>180s), please try again")
                    except Exception as e:
                        status_placeholder.error(f"Request failed: {e}")

                    status_placeholder.empty()

                    if result:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": result}
                        )
                        # 重新渲染：所有消息按顺序排列，chat_input 自动落到底部
                        st.rerun()
                    else:
                        st.error("Failed to get a response, please try again")
        else:
            st.info("Please complete a search first to enable the chat feature.")

# ========== 底部 ==========
st.markdown(
    '<div class="footer">Powered by arXiv + BERTopic + Plotly + RAG</div>',
    unsafe_allow_html=True
)
