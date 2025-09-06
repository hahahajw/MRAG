"""
retriever - 根据 query 在向量数据库中进行搜索并返回相似文档
            对于搜索，可以选择：
            1. 单稀疏检索，仅在一个稀疏向量场上进行检索
            2. 单密集检索，仅在一个密集向量场上进行检索
            3. 混合检索（多路检索），在多个向量场上进行检索
            默认为单密集检索

            对于召回，可以选择：
            1. weighted reranker
            2. rrf
            3. TODO：子模优化
            默认为 rrf

Author - hahahajw
Date - 2025-07-29 
"""

# 这么写有点太复杂了，完全可以借助 langchain_milvus 封装好的 similarity_search + Weighted ReRanker 控制仅在一个或一些向量场上进行搜索
# 不过，借助 similarity_search 无法实现在仅某些向量场上进行搜索后重排，因为 langchain_milvus 在实现时就控制好了是在所有的 vector fields 上创建一个 AnnSearchRequest
# 这样也有些好处，你可以获得原始的分数了
# 在 langchain_milvus 的实现过程中，vector_fields 是很重要的，向量场与其对应的 搜索参数、Weighted ReRanker 的权重的分配都是与其顺序直接相关的
# _vector_fields_from_embedding 和 embedding_func 是一一对应的，即 `_vector_fields_from_embedding[i]` 是由 `embedding_func[i]` 生成的
# 这也是为什么 _collection_hybrid_search 中可以依靠字段找到对应的嵌入函数


from typing import (
    Literal,
    List,
    Dict,
    Optional,
    Tuple,
)
import dashscope
import os

from langchain_milvus import Milvus
from langchain_core.documents import Document

from pymilvus import AnnSearchRequest, WeightedRanker, RRFRanker


class Retriever:
    def __init__(
            self,
            vector_store: Milvus,
            topk: int = 3,
            filter_: str = '',  # 和 python 的内置函数重名了
            search_fields: Optional[List[str]] = None,  # Digg 因为将默认参数设置为可变对象导致一次事故
            search_type: Literal['sparse', 'dense', 'hybrid'] = 'dense',
            search_params: Optional[List[Dict]] = None,
            ranker_type: Literal['weighted', 'rrf', 'model'] = 'model',
            ranker_params: Optional[Dict] = None
    ):
        """
        初始化一个检索器
        Args:
            vector_store: 要查询的向量数据库。会通过它使用一些 langchain_milvus 封装好的函数
            topk: 返回的文档数
            filter_: 过滤条件，默认不进行过滤
            search_fields (List[str]): 要在哪些向量场上进行检索，默认仅在密集嵌入上进行检索
            search_type (Literal['sparse', 'dense', 'hybrid']): 搜索类型，默认为 dense，即仅在一个密集嵌入向量场上进行搜索
                         当在超过 2 个向量场上进行搜索时，应指定为 hybrid
            search_params (List[Dict]): 调用 client.search 或 client.hybrid_search 所需要的参数
                           需要为 search_fields 中的每个向量场都设置对应的搜索参数
            ranker_type (Literal['weighted', 'rrf', 'model']): 重排序多路召回结果的方式
            ranker_params: 定义 reranker 所需的参数
                           当 `ranker_type` 为 `weighted` 时，应为 `search_fields` 中的每个向量场按照顺序传入对应的权重。权重在 [0, 1] 之间，不要求权重和为 1
                           Weighted ReRanker 的默认参数为 {'weights': [1.0] * len(sear)}
        """
        self.vector_store = vector_store
        self.topk = topk  # 加一个过滤参数
        self.filter_ = filter_
        self.search_fields = search_fields if search_fields else ['dense']
        self.search_type = search_type
        self.search_params = search_params if search_params else [{'metric_type': 'L2', 'params': {'ef': 21}}]
        self.ranker_type = ranker_type
        self.ranker_params = ranker_params if ranker_params else {
            'model_name': 'gte-rerank-v2',
            'pre_topk': 10,
            'pre_ranker_type': 'rrf',
            'pre_ranker_params': {'k': 60}  # 定义粗排所使用的重排序参数，粗排只能使用 weighted 或 rrf
        }

    def get_similar_chunk_with_score(self, query: str) -> List[Tuple[Document, float]]:
        """
        根据指定的检索参数进行检索
        Args:
            query: 问题

        Returns:
            List[Tuple[Document, float]]: [(top1 chunk, score), ...]
        """

        # 判断传入的搜索参数是否正确
        self.check_args()

        # 根据搜索参数路由
        # 单稀疏、密集检索 | 在单个向量场上进行检索
        if self.search_type in {'sparse', 'dense'}:
            raw_res = self.sparse_search(query=query) if self.search_type == 'sparse' else self.dense_search(query=query)

        # 混合检索
        elif self.search_type == 'hybrid':
            raw_res = self.hybrid_search(query=query)
            # 如果使用 rerank model 可以直接返回结果
            if self.ranker_type == 'model':
                return raw_res

        else:
            raise ValueError(f'search_type={self.search_type} 不合法')

        # print(type(raw_res))  # 类型为 SearchResult，我想这是 Milvus_client 中 search 和 hybrid_search 的类型注解错误
        # https://github.com/milvus-io/pymilvus/pull/2740  这个链接似乎解释了这么修改的原因
        return self.vector_store._parse_documents_from_search_results(raw_res)  # type: ignore

    def hybrid_search(self, query: str):
        # 参考 API：
        # https://milvus.io/docs/zh/multi-vector-search.md#Perform-Hybrid-Search
        # https://milvus.io/docs/zh/hybrid_search_with_milvus.md#Hybrid-Search-with-Milvus
        # langchain_milvus 中 _collection_hybrid_search 的实现

        if self.ranker_type == 'model':
            return self.search_with_rerank_model(query)

        # 从 search_fields、search_params 中解析出各自向量场的 AnnSearchRequest
        search_reqs = []
        vector_fields_from_embedding = self.vector_store._vector_fields_from_embedding
        embedding_funcs = self.vector_store._as_list(self.vector_store.embedding_func)

        for field, params in zip(self.search_fields, self.search_params):
            search_data = query
            if field in vector_fields_from_embedding:
                embedding_func = embedding_funcs[
                    vector_fields_from_embedding.index(field)
                ]
                search_data = embedding_func.embed_query(query)

            req = AnnSearchRequest(
                data=[search_data],
                anns_field=field,
                param=params,
                limit=self.topk,
                expr=self.filter_  # 混合搜索要在每个 AnnSearchRequest 中添加过滤条件
            )
            search_reqs.append(req)

        # 创建 ReRanker
        if self.ranker_type == 'weighted':
            weights = self.ranker_params.get('weights', [1.0] * len(self.search_fields))
            reranker = WeightedRanker(*weights)
        elif self.ranker_type == 'rrf':
            reranker = RRFRanker(self.ranker_params.get('k', 60))
        else:
            raise ValueError('不支持的重排器')

        # 去除所有索引类型为 BM25 的字段
        if self.vector_store.enable_dynamic_field:
            output_fields = ["*"]
        else:
            output_fields = self.vector_store._remove_forbidden_fields(self.vector_store.fields[:])

        # 进行搜索
        return self.vector_store.client.hybrid_search(
            collection_name=self.vector_store.collection_name,
            reqs=search_reqs,
            ranker=reranker,
            limit=self.topk,
            output_fields=output_fields
        )

    def search_with_rerank_model(self, query: str) -> List[Tuple[Document, float]]:
        # 使用重排序模型处理结果

        # 新建一个 Retriever 实例用于粗排
        # 粗排默认使用 rrf 重排器
        pre_retriever = Retriever(
            vector_store=self.vector_store,
            topk=self.ranker_params.get('pre_topk', 10),  # 加一个过滤参数
            filter_=self.filter_,
            search_fields=self.search_fields,
            search_type='hybrid',
            search_params=self.search_params,  # 能路由到这里，search_fields 中的每个向量场一定都有对应的搜索参数
            ranker_type=self.ranker_params.get('pre_ranker_type', 'rrf'),
            ranker_params=self.ranker_params.get('pre_ranker_params', {'k': 60})
        )
        # 获得粗排结果
        pre_res = pre_retriever.get_similar_chunk_with_score(query)

        # 使用 rerank model 进行重排
        # https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2780056.html
        dashscope.api_key = os.getenv("BL_API_KEY")
        resp = dashscope.TextReRank.call(
            model=self.ranker_params.get('model_name', 'gte-rerank-v2'),
            query=query,
            documents=[res[0].page_content for res in pre_res],
            top_n=self.topk,
            return_documents=False
        )

        # 解析出最后的结果
        return [
            (pre_res[res['index']][0], res['relevance_score']) for res in resp['output']['results']
        ]

    def sparse_search(self, query: str):
        """
        在一个稀疏嵌入向量场上进行稀疏检索
        Args:
            query: 问题

        Returns:
            client.search 的原始结果
        """
        # 参考 API：
        # 执行全文搜索：https://milvus.io/docs/zh/full-text-search.md#Insert-text-data
        # BM25 函数生成的稀疏向量不能在全文检索中直接访问或输出：https://milvus.io/docs/zh/full-text-search.md#FAQ

        search_field = self.search_fields[0]  # 这里应当是 search_field 而不是 search_fields, 毕竟只有一个向量场参与搜索
        search_params = self.search_params[0]

        output_fields = self.vector_store.fields[:]
        output_fields.remove(search_field)
        # langchain_milvus 的 _remove_forbidden_fields 是移除所有索引类型为 BM25 的字段
        # 有多个索引为 BM25 的稀疏向量字段才可能会导致这个问题
        # output_fields = self.vector_store._remove_forbidden_fields(self.vector_store.fields[:])

        return self.vector_store.client.search(
            collection_name=self.vector_store.collection_name,
            data=[query],
            limit=self.topk,
            filter=self.filter_,
            anns_field=search_field,
            search_params=search_params,
            output_fields=output_fields
        )

    def dense_search(self, query: str):
        # 参考 API：
        # Search API：https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md
        # langchain_milvus 中 _collection_hybrid_search 的实现

        search_field = self.search_fields[0]
        search_params = self.search_params[0]

        # 选择建立 search_field 时所使用的 embedding function
        embedding_func = self.vector_store._as_list(self.vector_store.embedding_func)[
            self.vector_store._vector_fields_from_embedding.index(search_field)
        ]
        embedding = embedding_func.embed_query(query)
        return self.vector_store.client.search(
            collection_name=self.vector_store.collection_name,
            data=[embedding],
            limit=self.topk,
            filter=self.filter_,
            anns_field=search_field,
            search_params=search_params,
            output_fields=['*']
        )

    def check_args(self):
        # 判断传入的搜索参数是否正确

        # 判断 search_fields 是否在 vector_fields 中
        all_fields = set(self.vector_store.vector_fields)
        for field in self.search_fields:
            if field not in all_fields:
                raise ValueError(f'{field} 在向量数据库中不存在')

        # 判断参数间的数量关系是否合理
        if len(self.search_fields) > 1:
            if self.search_type != 'hybrid':
                raise ValueError('当在超过 2 个向量场上进行搜索时，search_type 应指定为 hybrid')
            # 当使用 weighted reranker 时，ranker_params 类似于 {'weights': [0.5, 0.5]}
            elif self.ranker_type == 'weighted' and len(self.ranker_params['weights']) != len(
                    self.search_fields):  # type: ignore
                raise ValueError('在使用 weighted reranker 时应为每个向量场都设置对应的权重')

            if len(self.search_params) != len(self.search_fields):
                raise ValueError('应为每个向量场都设置对应的搜索参数')

        return


if __name__ == '__main__':
    from langchain_milvus import Milvus, BM25BuiltInFunction
    from langchain_openai import OpenAIEmbeddings
    from dotenv import load_dotenv
    import pprint

    load_dotenv()

    e_m = OpenAIEmbeddings(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model="text-embedding-v3",
        dimensions=1024,
        check_embedding_ctx_length=False
    )

    vs = Milvus(
        collection_name='hybrid_hotpotqa500_hnsw',
        embedding_function=e_m,
        builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
        vector_field=['dense', 'sparse']
    )

    # 调用单稀疏检索
    sparse_r = Retriever(
        vector_store=vs,
        topk=3,
        search_fields=['sparse'],
        search_type='sparse',
        search_params=[{'params': {'drop_ratio_search': 0.2}}]
    )
    a = sparse_r.get_similar_chunk_with_score(query='hello')
    pprint.pprint(a)
    print('-' * 50)

    # 调用单密集检索
    dense_r = Retriever(
        vector_store=vs,
        topk=3,
        search_fields=['dense'],
        search_type='dense',
        search_params=[{'metric_type': 'L2', 'params': {'ef': 21}}]
    )
    a = dense_r.get_similar_chunk_with_score(query='hello')
    pprint.pprint(a)
    print('-' * 50)

    # 使用 Weighted ReRanker
    wr = Retriever(
        vector_store=vs,
        topk=3,
        search_fields=['dense', 'sparse'],
        search_type='hybrid',
        search_params=[
            {'metric_type': 'L2', 'params': {'ef': 21}},
            {'params': {'drop_ratio_search': 0.2}}
        ],
        ranker_type='weighted',
        ranker_params={'weights': [1.0, 1.0]}
    )
    a = wr.get_similar_chunk_with_score(query='hello')
    pprint.pprint(a)
    print('-' * 50)

    # 使用 RRF ReRanker
    rrr = Retriever(
        vector_store=vs,
        topk=3,
        search_fields=['dense', 'sparse'],
        search_type='hybrid',
        search_params=[
            {'metric_type': 'L2', 'params': {'ef': 21}},
            {'params': {'drop_ratio_search': 0.2}}
        ],
        ranker_type='rrf',
        ranker_params={'k': 60}
    )
    a = rrr.get_similar_chunk_with_score(query='hello')
    pprint.pprint(a)
    print('-' * 50)

    # 使用 rerank model
    model_reranker = Retriever(
        vector_store=vs,
        topk=3,
        search_fields=['dense', 'sparse'],
        search_type='hybrid',
        search_params=[
            {'metric_type': 'L2', 'params': {'ef': 21}},
            {'params': {'drop_ratio_search': 0.2}}
        ],
        ranker_type='model',
        ranker_params={
            'model_name': 'gte-rerank-v2',
            'pre_topk': 10,
            'pre_ranker_type': 'rrf',
            'pre_ranker_params': {'k': 60}  # 定义粗排所使用的重排序参数，粗排只能使用 weighted 或 rrf
        }
    )
    a = model_reranker.get_similar_chunk_with_score(query='hello')
    pprint.pprint(a)
    print('-' * 50)
