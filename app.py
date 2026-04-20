"""
Streamlit 前端界面
位置：D:\dataScienceProjects\AI_Assistant\app.py
"""
import streamlit as st
import requests
import json
from datetime import datetime

# API 配置
API_BASE_URL = "http://127.0.0.1:8000"

# 页面配置
st.set_page_config(
    page_title="AI 论文助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 函数定义 ==========

def check_api_health():
    """检查 API 是否在线"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def init_session():
    """初始化对话会话"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat/init",
            json={"session_id": None, "topic_id": None},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("session_id")
        else:
            st.error(f"初始化失败: HTTP {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到后端服务！请确保后端已启动")
        st.info("启动命令: cd src && python main.py")
        return None
    except Exception as e:
        st.error(f"初始化失败: {e}")
        return None

def send_message(session_id, message):
    """发送消息"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat/message",
            json={"session_id": session_id, "message": message},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.error("会话已过期，请刷新页面重新开始")
            return None
        else:
            st.error(f"请求失败: HTTP {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.error("请求超时，请稍后重试")
        return None
    except Exception as e:
        st.error(f"请求失败: {e}")
        return None

def display_paper_card(paper_info):
    """显示论文卡片"""
    with st.container():
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #1f77b4;">
            <div style="font-size: 1.1rem; font-weight: bold; color: #1f77b4;">📄 {paper_info.get('title', 'Unknown')}</div>
        </div>
        """, unsafe_allow_html=True)

        if paper_info.get('ai_summary'):
            st.markdown(f"""
            <div style="background-color: #e8f4f8; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0;">
            🤖 <strong>AI 总结</strong><br>{paper_info['ai_summary']}
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            authors = paper_info.get('authors', [])
            if authors:
                st.markdown(f"**作者**: {', '.join(authors[:3])}")
        with col2:
            if paper_info.get('published'):
                st.markdown(f"**日期**: {paper_info['published'][:10]}")

        if paper_info.get('pdf_url'):
            st.markdown(f"[📥 下载PDF]({paper_info['pdf_url']})")

def display_answer(result):
    """显示回答"""
    if not result:
        return

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    st.markdown("### 💬 回答")

    # 显示回答文本
    st.markdown(answer)

    # 显示论文来源卡片
    if sources:
        st.markdown("---")
        st.markdown("### 📚 相关论文")

        for i, source in enumerate(sources[:5]):
            with st.expander(f"{i+1}. {source.get('title', 'Unknown')[:80]}..."):
                st.markdown(f"**标题**: {source.get('title', 'Unknown')}")

                authors = source.get('authors', [])
                if authors:
                    st.markdown(f"**作者**: {', '.join(authors[:3])}")

                if source.get('published'):
                    st.markdown(f"**发布日期**: {source['published'][:10]}")

                if source.get('relevance'):
                    st.markdown(f"**相关性**: {source['relevance']:.3f}")

                if source.get('ai_summary'):
                    st.markdown(f"**AI总结**: {source['ai_summary']}")

                if source.get('pdf_url'):
                    st.markdown(f"[📥 下载PDF]({source['pdf_url']})")

# ========== 自定义 CSS ==========

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.8rem;
    }
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========== 侧边栏 ==========

with st.sidebar:
    st.title("🤖 AI 论文助手")
    st.markdown("---")

    # 连接状态
    if check_api_health():
        st.success("🟢 后端服务在线")
    else:
        st.error("🔴 后端服务离线")
        st.info("请先启动后端: `cd src && python main.py`")

    st.markdown("---")
    st.markdown("### 功能介绍")
    st.markdown("""
    - 🔍 **智能搜索**：语义理解，不是简单关键词
    - 📝 **AI摘要**：每篇论文自动生成总结
    - 📚 **来源追溯**：每个回答都有论文出处
    - 💬 **多轮对话**：记住上下文
    """)
    st.markdown("---")
    st.markdown("### 使用技巧")
    st.markdown("""
    - 用英文提问效果更好
    - 可以追问细节
    - 尝试不同关键词
    """)
    st.markdown("---")
    st.markdown(f"**版本**: v1.0.0")

# ========== 主界面 ==========

def main():
    st.markdown('<div class="main-header">📚 AI 论文助手</div>', unsafe_allow_html=True)

    # 初始化会话
    if "session_id" not in st.session_state:
        with st.spinner("正在初始化..."):
            session_id = init_session()
            if session_id:
                st.session_state.session_id = session_id
                st.session_state.messages = []
                st.success("✅ 对话已就绪！")
            else:
                st.error("❌ 无法连接到后端服务")
                st.info("请先启动后端: `cd src && python main.py`")
                return

    # 显示对话历史
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"].get("answer", "无回答"))

                # 显示来源
                sources = msg["content"].get("sources", [])
                if sources:
                    with st.expander("📚 查看论文来源"):
                        for src in sources[:3]:
                            st.markdown(f"**{src.get('title', 'Unknown')}**")
                            if src.get('ai_summary'):
                                st.markdown(f"> {src['ai_summary'][:150]}...")
                            if src.get('pdf_url'):
                                st.markdown(f"[PDF链接]({src['pdf_url']})")
                            st.markdown("---")

    # 输入框
    if prompt := st.chat_input("输入你的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 发送请求
        with st.chat_message("assistant"):
            with st.spinner("正在搜索论文并生成回答..."):
                result = send_message(st.session_state.session_id, prompt)

                if result:
                    st.markdown(result.get("answer", "无回答"))

                    # 显示来源
                    sources = result.get("sources", [])
                    if sources:
                        with st.expander("📚 查看论文来源"):
                            for src in sources[:3]:
                                st.markdown(f"**{src.get('title', 'Unknown')}**")
                                if src.get('ai_summary'):
                                    st.markdown(f"> {src['ai_summary'][:150]}...")
                                if src.get('pdf_url'):
                                    st.markdown(f"[PDF链接]({src['pdf_url']})")
                                st.markdown("---")

                    st.session_state.messages.append({"role": "assistant", "content": result})
                else:
                    st.error("请求失败，请稍后重试")

    # 底部
    st.markdown('<div class="footer">Powered by RAG + AI Summary | 数据来源: arXiv</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()