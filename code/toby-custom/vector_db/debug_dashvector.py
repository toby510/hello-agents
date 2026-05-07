from dashvector import Client

client = Client(
    api_key="sk-i8bSXvL8AmwOQHb0fLonVn3S6KWfhFEBFF79C49F311F1B58F8282A7DF628D",
    endpoint="vrs-cn-n5m4rui5u0001c.dashvector.cn-hangzhou.aliyuncs.com",
)

coll = client.get("demo_collection")

# fetch 返回结构
print("=== fetch ===")
resp = coll.fetch(["1", "2"])
print(f"type: {type(resp)}")
print(f"output type: {type(resp.output)}")
print(f"output: {resp.output}")
print(f"output attrs: {[a for a in dir(resp.output) if not a.startswith('_')]}")

# query 返回结构
print("\n=== query ===")
resp2 = coll.query(vector=[0.82, 0.87, 0.11, 0.19], topk=3)
print(f"type: {type(resp2)}")
print(f"output type: {type(resp2.output)}")
print(f"output: {resp2.output}")

# stats 返回结构
print("\n=== stats ===")
resp3 = coll.stats()
print(f"type: {type(resp3)}")
print(f"output type: {type(resp3.output)}")
print(f"output: {resp3.output}")
print(f"output attrs: {[a for a in dir(resp3.output) if not a.startswith('_')]}")

# scan 替代方案：query 不带 vector
print("\n=== query without vector ===")
try:
    resp4 = coll.query(topk=100)
    print(f"output: {resp4.output}")
except Exception as e:
    print(f"error: {e}")
