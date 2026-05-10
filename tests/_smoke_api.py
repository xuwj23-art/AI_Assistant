"""手工 API 冒烟脚本（不属于 pytest 集合）"""
import requests

BASE = "http://127.0.0.1:8000"

print("--- /api/topics ---")
r = requests.get(f"{BASE}/api/topics", timeout=30)
d = r.json()
print(f"  total={d['total']}; first topic = {d['topics'][0]['topic_name']} ({d['topics'][0]['paper_count']} papers)")

print("--- /api/topics/trends ---")
r = requests.get(f"{BASE}/api/topics/trends?top_n=5", timeout=30)
d = r.json()
print(f"  years={d['years']}; n_topics={len(d['topics'])}; trending_count={len(d['trending'])}")

print("--- /api/topics/0/papers (paged, sort=date) ---")
r = requests.get(f"{BASE}/api/topics/0/papers?page=1&page_size=5&sort_by=date", timeout=30)
d = r.json()
print(f"  total={d['total']}, returned={len(d['papers'])}; sort_by={d['sort_by']}")
for p in d["papers"]:
    title = p["title"][:60] + ("..." if len(p["title"]) > 60 else "")
    print(f"    - {title} ({str(p['published'])[:10]})")

print("--- /api/topics/0/similar ---")
r = requests.get(f"{BASE}/api/topics/0/similar?top_n=2", timeout=30)
d = r.json()
sims = [(t["topic_name"], t["similarity"]) for t in d["similar_topics"]]
print(f"  target={d['topic_name']}; similar={sims}")
