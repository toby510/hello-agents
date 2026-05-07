from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "demo_collection"
VECTOR_DIM = 4

client = MilvusClient(
    uri="https://in03-69cec781fbed1e6.serverless.aws-eu-central-1.cloud.zilliz.com",
    token="ecd5fc00cb9c97c547733c9d851039993c8ec9c38821ddcbdab7ef1f0458fd883906f0be740ebe9cfbe7baf4d0b7d5439b337e06",
)


def show_collections():
    print("=== 1. 查看所有 Collections ===")
    collections = client.list_collections()
    print(collections)
    print()


def create_collection():
    print("=== 2. 创建 Collection ===")
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"旧 Collection '{COLLECTION_NAME}' 已删除")

    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="price", datatype=DataType.FLOAT)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Collection '{COLLECTION_NAME}' 创建成功")
    print()


def insert_points():
    print("=== 3. 插入 Points (Create) ===")
    data = [
        {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "name": "apple", "category": "fruit", "price": 5.5},
        {"id": 2, "vector": [0.3, 0.3, 0.3, 0.3], "name": "banana2", "category": "fruit", "price": 3.0},
        {"id": 3, "vector": [0.8, 0.9, 0.1, 0.2], "name": "carrot", "category": "vegetable", "price": 2.5},
        {"id": 4, "vector": [0.85, 0.88, 0.12, 0.18], "name": "broccoli", "category": "vegetable", "price": 4.0},
    ]
    result = client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"插入结果: {result}")
    print()


def read_points():
    print("=== 4. 读取 Points (Read) ===")
    points = client.get(
        collection_name=COLLECTION_NAME,
        ids=[1, 2],
        output_fields=["*"],
    )
    for p in points:
        print(f"  ID={p['id']}, name={p.get('name')}, category={p.get('category')}, price={p.get('price')}, vector={p.get('vector')}")
    print()


def update_point():
    print("=== 5. 更新 Point (Update) ===")
    client.upsert(
        collection_name=COLLECTION_NAME,
        data=[
            {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "name": "apple", "category": "fruit", "price": 6.0},
        ],
    )
    client.upsert(
        collection_name=COLLECTION_NAME,
        data=[
            {"id": 2, "vector": [0.25, 0.15, 0.35, 0.45], "name": "banana3", "category": "fruit", "price": 3.5},
        ],
    )
    print("Point 1 payload 更新完成, Point 2 vector 和 payload 更新完成")

    points = client.get(
        collection_name=COLLECTION_NAME, ids=[1, 2], output_fields=["*"]
    )
    for p in points:
        print(f"  ID={p['id']}, name={p.get('name')}, category={p.get('category')}, price={p.get('price')}")
    print()


def search_similar():
    print("=== 6. 向量搜索 (Search) ===")
    query_vector = [[0.82, 0.87, 0.11, 0.19]]  # 接近 carrot / broccoli
    response = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vector,
        limit=3,
        output_fields=["*"],
    )
    for r in response[0]:
        print(f"  ID={r['id']}, score={r['distance']:.4f}, entity={r.get('entity', {})}")
    print()


def search_with_filter():
    print("=== 7. 带过滤条件的搜索 (Search + Filter) ===")
    query_vector = [[0.15, 0.15, 0.35, 0.35]]
    response = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vector,
        filter='category == "fruit"',
        limit=3,
        output_fields=["*"],
    )
    for r in response[0]:
        print(f"  ID={r['id']}, score={r['distance']:.4f}, entity={r.get('entity', {})}")
    print()


def count_points():
    print("=== 8. 统计 Points ===")
    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["count(*)"],
    )
    print(f"总点数: {result[0]['count(*)']}")
    print()


def delete_points():
    print("=== 9. 删除 Points (Delete) ===")
    client.delete(
        collection_name=COLLECTION_NAME,
        ids=[1],
    )
    print("已删除 ID=1 的 point")

    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["count(*)"],
    )
    print(f"删除后总点数: {result[0]['count(*)']}")
    print()


def scroll_points():
    print("=== 10. 滚动遍历 (Scroll) ===")
    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["*"],
        limit=10,
    )
    for r in result:
        print(f"  ID={r['id']}, name={r.get('name')}, category={r.get('category')}, price={r.get('price')}")
    print()


def cleanup():
    print("=== 11. 清理 Collection ===")
    client.drop_collection(COLLECTION_NAME)
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
