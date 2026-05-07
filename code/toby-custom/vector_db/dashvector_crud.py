from dashvector import Client, Doc

COLLECTION_NAME = "demo_collection"
VECTOR_DIM = 4

client = Client(
    api_key="sk-i8bSXvL8AmwOQHb0fLonVn3S6KWfhFEBFF79C49F311F1B58F8282A7DF628D",
    endpoint="vrs-cn-n5m4rui5u0001c.dashvector.cn-hangzhou.aliyuncs.com",
)


def show_collections():
    print("=== 1. 查看所有 Collections ===")
    resp = client.list()
    for coll in resp:
        print(f"  {coll}")
    print()


def create_collection():
    print("=== 2. 创建 Collection ===")
    existing = client.get(COLLECTION_NAME)
    if existing:
        client.delete(COLLECTION_NAME)
        print(f"旧 Collection '{COLLECTION_NAME}' 已删除")

    client.create(
        name=COLLECTION_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
    )
    print(f"Collection '{COLLECTION_NAME}' 创建成功")
    print()


def insert_points():
    print("=== 3. 插入 Points (Create) ===")
    collection = client.get(COLLECTION_NAME)
    docs = [
        Doc(id="1", vector=[0.1, 0.2, 0.3, 0.4], fields={"name": "apple", "category": "fruit", "price": 5.5}),
        Doc(id="2", vector=[0.3, 0.3, 0.3, 0.3], fields={"name": "banana2", "category": "fruit", "price": 3.0}),
        Doc(id="3", vector=[0.8, 0.9, 0.1, 0.2], fields={"name": "carrot", "category": "vegetable", "price": 2.5}),
        Doc(id="4", vector=[0.85, 0.88, 0.12, 0.18], fields={"name": "broccoli", "category": "vegetable", "price": 4.0}),
    ]
    resp = collection.upsert(docs)
    print(f"插入结果: {resp}")
    print()


def _get_doc_by_id(collection, target_ids):
    """通过 query 全量获取后在内存中按 ID 筛选（规避 fetch 偶发返回 None 的问题）"""
    resp = collection.query(topk=100)
    if not resp.output:
        return []
    result = []
    target_set = set(target_ids)
    for doc in resp.output:
        if doc.id in target_set:
            result.append(doc)
    return result


def read_points():
    print("=== 4. 读取 Points (Read) ===")
    collection = client.get(COLLECTION_NAME)
    docs = _get_doc_by_id(collection, ["1", "2"])
    for doc in docs:
        print(f"  ID={doc.id}, fields={doc.fields}, vector={doc.vector}")
    print()


def update_point():
    print("=== 5. 更新 Point (Update) ===")
    collection = client.get(COLLECTION_NAME)
    collection.upsert([
        Doc(id="1", vector=[0.1, 0.2, 0.3, 0.4], fields={"name": "apple", "category": "fruit", "price": 6.0, "on_sale": True}),
    ])
    collection.upsert([
        Doc(id="2", vector=[0.25, 0.15, 0.35, 0.45], fields={"name": "banana3", "category": "fruit", "price": 3.5}),
    ])
    print("Point 1 payload 更新完成, Point 2 vector 和 payload 更新完成")

    docs = _get_doc_by_id(collection, ["1", "2"])
    for doc in docs:
        print(f"  ID={doc.id}, fields={doc.fields}")
    print()


def search_similar():
    print("=== 6. 向量搜索 (Search) ===")
    collection = client.get(COLLECTION_NAME)
    query_vector = [0.82, 0.87, 0.11, 0.19]
    resp = collection.query(
        vector=query_vector,
        topk=3,
    )
    for doc in (resp.output or []):
        print(f"  ID={doc.id}, score={doc.score:.4f}, fields={doc.fields}")
    print()


def search_with_filter():
    print("=== 7. 带过滤条件的搜索 (Search + Filter) ===")
    collection = client.get(COLLECTION_NAME)
    query_vector = [0.15, 0.15, 0.35, 0.35]
    resp = collection.query(
        vector=query_vector,
        topk=3,
        filter='category = "fruit"',
    )
    for doc in (resp.output or []):
        print(f"  ID={doc.id}, score={doc.score:.4f}, fields={doc.fields}")
    print()


def count_points():
    print("=== 8. 统计 Points ===")
    collection = client.get(COLLECTION_NAME)
    stats = collection.stats().output
    if stats:
        print(f"总点数: {stats.total_doc_count}")
    print()


def delete_points():
    print("=== 9. 删除 Points (Delete) ===")
    collection = client.get(COLLECTION_NAME)
    collection.delete(["1"])
    print("已删除 ID=1 的 point")

    stats = collection.stats().output
    if stats:
        print(f"删除后总点数: {stats.total_doc_count}")
    print()


def scroll_points():
    print("=== 10. 滚动遍历 (Scroll) ===")
    collection = client.get(COLLECTION_NAME)
    resp = collection.query(topk=100)
    for doc in (resp.output or []):
        print(f"  ID={doc.id}, fields={doc.fields}")
    print()


def cleanup():
    print("=== 11. 清理 Collection ===")
    client.delete(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' 已删除")
    print()


def main():
    show_collections()
    create_collection()
    insert_points()
    read_points()
    update_point()
    search_similar()
    search_with_filter()
    count_points()
    delete_points()
    scroll_points()
    # cleanup()
    print("所有操作执行完毕！")


if __name__ == "__main__":
    main()
