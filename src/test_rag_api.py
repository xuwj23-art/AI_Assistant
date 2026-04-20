import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_rag():
    print("=" * 50)
    print("RAG模块测试开始")
    print("=" * 50)

    # 1. 健康检查
    print("\n1. 健康检查...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"   状态: {resp.json()}")

    # 2. 初始化对话
    print("\n2. 初始化对话...")
    resp = requests.post(f"{BASE_URL}/api/chat/init", json={})
    data = resp.json()
    session_id = data["session_id"]
    print(f"   会话ID: {session_id}")
    print(f"   消息: {data['message']}")

    # 3. 发送问题
    questions = [
        "什么是Transformer模型？",
        "机器学习有哪些应用？",
        "深度学习和神经网络有什么关系？"
    ]

    for q in questions:
        print(f"\n3. 提问: {q}")
        resp = requests.post(
            f"{BASE_URL}/api/chat/message",
            json={"session_id": session_id, "message": q}
        )
        data = resp.json()
        print(f"   回答: {data['answer'][:150]}...")
        print(f"   来源论文: {len(data['sources'])}篇")
        for src in data['sources']:
            print(f"     - {src['title']}")
        time.sleep(1)

    # 4. 获取历史
    print("\n4. 获取对话历史...")
    resp = requests.get(f"{BASE_URL}/api/chat/history", params={"session_id": session_id})
    data = resp.json()
    print(f"   历史记录数: {len(data['history'])}")

    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    test_rag()