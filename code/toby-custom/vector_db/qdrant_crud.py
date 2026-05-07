from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

COLLECTION_NAME = "demo_collection"
VECTOR_SIZE = 4

qdrant_client = QdrantClient(
    url="https://95abf2e0-7725-4e71-a1c5-5c7deae88269.us-west-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MzQwYjZlNTUtNmQwYi00MzFjLWIzODQtYTU5YzE5NDc1ZDg0In0.evMEbWRPlSH8pM83HG88M0JGBoB3Ih_0nTGh3ZQSd80"
)


def show_collections():
    print("=== 1. 查看所有 Collections ===")
    collections = qdrant_client.get_collections()
    print(collections)
    print()


def create_collection():
    print("=== 2. 创建 Collection ===")
    # 如果已存在则先删除，保证可重复运行
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)
        print(f"旧 Collection '{COLLECTION_NAME}' 已删除")

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    # 为过滤字段创建索引，否则过滤搜索会报错
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_type=PayloadSchemaType.KEYWORD,
    )
    print(f"Collection '{COLLECTION_NAME}' 创建成功，并已为 'category' 字段创建索引")
    print()


def insert_points():
    print("=== 3. 插入 Points (Create) ===")
    points = [
        PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"name": "apple", "category": "fruit", "price": 5.5},
        ),
        PointStruct(
            id=2,
            vector=[0.3, 0.3, 0.3, 0.3],
            payload={"name": "banana2", "category": "fruit", "price": 3.0},
        ),
        PointStruct(
            id=3,
            vector=[0.8, 0.9, 0.1, 0.2],
            payload={"name": "carrot", "category": "vegetable", "price": 2.5},
        ),
        PointStruct(
            id=4,
            vector=[0.85, 0.88, 0.12, 0.18],
            payload={"name": "broccoli", "category": "vegetable", "price": 4.0},
        ),
    ]
    operation_info = qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points,
    )
    print(f"插入结果: {operation_info.status}")
    print()


def read_points():
    print("=== 4. 读取 Points (Read) ===")
    # 按 ID 查询
    points = qdrant_client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2],
        with_payload=True,
        with_vectors=True,
    )
    for p in points:
        print(f"  ID={p.id}, payload={p.payload}, vector={p.vector}")
    print()


def update_point():
    print("=== 5. 更新 Point (Update) ===")
    # 更新 payload
    qdrant_client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"price": 6.0, "on_sale": True},
        points=[1],
    )
    # 也可以覆盖整个 point（upsert）
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=2,
                vector=[0.25, 0.15, 0.35, 0.45],
                payload={"name": "banana3", "category": "fruit", "price": 3.5},
            )
        ],
    )
    print("Point 1 payload 更新完成, Point 2 vector 和 payload 更新完成")

    # 验证更新结果
    points = qdrant_client.retrieve(
        collection_name=COLLECTION_NAME, ids=[1, 2], with_payload=True
    )
    for p in points:
        print(f"  ID={p.id}, payload={p.payload}")
    print()


def search_similar():
    print("=== 6. 向量搜索 (Search) ===")
    query_vector = [0.82, 0.87, 0.11, 0.19]  # 接近 carrot / broccoli
    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3,
        with_payload=True,
    )
    for r in response.points:
        print(f"  ID={r.id}, score={r.score:.4f}, payload={r.payload}")
    print()


def search_with_filter():
    print("=== 7. 带过滤条件的搜索 (Search + Filter) ===")
    query_vector = [0.15, 0.15, 0.35, 0.35]
    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value="fruit"),
                )
            ]
        ),
        limit=3,
        with_payload=True,
    )
    for r in response.points:
        print(f"  ID={r.id}, score={r.score:.4f}, payload={r.payload}")
    print()


def count_points():
    print("=== 8. 统计 Points ===")
    count = qdrant_client.count(collection_name=COLLECTION_NAME)
    print(f"总点数: {count.count}")
    print()


def delete_points():
    print("=== 9. 删除 Points (Delete) ===")
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=[1],
    )
    print("已删除 ID=1 的 point")

    count = qdrant_client.count(collection_name=COLLECTION_NAME)
    print(f"删除后总点数: {count.count}")
    print()


def scroll_points():
    print("=== 10. 滚动遍历 (Scroll) ===")
    records, next_page_offset = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
    )
    for r in records:
        print(f"  ID={r.id}, payload={r.payload}")
    print()


def cleanup():
    print("=== 11. 清理 Collection ===")
    qdrant_client.delete_collection(COLLECTION_NAME)
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
