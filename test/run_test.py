"""
run_test - 运行测试数据，使用 Map-reducer 并行处理

Author - hahahajw
Date - 2025-08-25 
"""
from loguru import logger as log
from typing import (
    TypedDict,
    List
)
import os
import json
from tqdm import tqdm
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from langgraph.graph.state import START, StateGraph, END
from langgraph.types import Send

from naive_rag.workflow import get_naive_rag_workflow
from modules.retriever import Retriever


# 定义一个父图，包裹 Naive RAG
class SendState(TypedDict):
    queries: List[str]


def send_queries(state: SendState):
    return [
        Send(
            'naive_rag',
            {
                'query': query
            }
        ) for query in state['queries']
    ]


def get_send_workflow():
    wf = StateGraph(SendState)

    wf.add_node('naive_rag', get_naive_rag_workflow().compile())

    wf.add_conditional_edges(
        START,
        send_queries,
        ['naive_rag']
    )
    wf.add_edge('naive_rag', END)

    return wf


def main():
    import json
    from tqdm import tqdm
    from dotenv import load_dotenv

    load_dotenv()

    # 定义图的运行时配置
    # 定义 LLM、vector store、retriever
    llm = ChatOpenAI(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model='qwen-plus-latest',
        temperature=0.0
    )
    embed_model = OpenAIEmbeddings(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model="text-embedding-v4",
        dimensions=1024,
        check_embedding_ctx_length=False
    )
    # 定义向量场的索引参数
    dense_index_parma = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {
            "M": 64,
            "efConstruction": 400
        }
    }
    sparse_index_param = {
        "metric_type": "BM25",
        "index_type": "AUTOINDEX",
        "params": {}
    }
    vector_store = Milvus(
        collection_name='mrag',
        embedding_function=embed_model,
        builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
        vector_field=['dense', 'sparse'],
        index_params=[dense_index_parma, sparse_index_param]  # 顺序关系是对应的
    )
    rerank_model_retriever = Retriever(
        vector_store=vector_store,
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

    config = {
        'configurable': {
            'retriever': rerank_model_retriever,
            'llm': llm,
        }
    }

    # 定义图
    send_wf = get_send_workflow()
    graph = send_wf.compile()

    # 加载问题
    # test_file_path = './test.json'
    test_file_path = './train.json'
    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_datas = json.load(f)

    for i in tqdm(range(0, len(test_datas), 10), desc='回答'):
        batch_queries = [test_data['question'] for test_data in test_datas[i:i+10]]
        try:
            graph.invoke(
                input={'queries': batch_queries},
                config=config
            )
        except Exception as e:
            log.error('error', f'{e}')


def parse_answers():
    """把答案解析成符合提交的格式"""

    # answer_files = os.listdir('../answer')
    # output_file_path = './final_answer.json'
    answer_files = os.listdir('../train_answer')
    output_file_path = './train_final_answer.json'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    answers = []
    for cur_answer_file in answer_files:
        # cur_answer_file = f'../answer/{cur_answer_file}'
        cur_answer_file = f'../train_answer/{cur_answer_file}'
        with open(cur_answer_file, 'r', encoding='utf-8') as f:
            cur_answer = json.load(f)

        tmp = {
            'filename': cur_answer['filename'],
            'page': cur_answer['page'] + 1,
            'question': cur_answer['query'],
            'answer': cur_answer['answer']
        }
        answers.append(tmp)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)


def eval_answer():
    with open('./train_final_answer.json', 'r', encoding='utf-8') as f:
        prediction = json.load(f)
    with open('./train.json', 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    prediction.sort(key=lambda x: x['question'])
    ground_truth.sort(key=lambda x: x['question'])

    filename_score = page_score = answer_score = 0
    for pred, gt in zip(prediction, ground_truth):
        cur_f_s = 1 if pred['filename'] == gt['filename'] else 0
        cur_p_s = 1 if pred['page'] == gt['page'] else 0
        cur_a_s = get_answer_score(pred['answer'], gt['answer'])

        print(f'q: {pred["question"]}, {pred["question"] == gt["question"]}')
        print(f'p: {pred["filename"]}\ng: {gt["filename"]}\nscore: {cur_f_s}')
        print(f'p: {pred["page"]}\ng: {gt["page"]}\nscore: {cur_p_s}')
        print(f'p: {pred["answer"]}\ng: {gt["answer"]}\nscore: {cur_a_s}')
        print('-' * 20)

        pred['filename_score'] = cur_f_s
        pred['page_score'] = cur_p_s
        pred['answer_score'] = cur_a_s

        filename_score += cur_f_s
        page_score += cur_p_s
        answer_score += cur_a_s

    filename_score = (filename_score / len(prediction)) * 0.25
    page_score = (page_score / len(prediction)) * 0.25
    answer_score = (answer_score / len(prediction)) * 0.5

    prediction.append({
        'filename_score': filename_score,
        'page_score': page_score,
        'answer_score': answer_score,
    })

    print(f'filename_score: {filename_score}\npage_score: {page_score}\nanswer_score: {answer_score}\ntotal_score: {filename_score + page_score + answer_score}')


def get_answer_score(pred_answer: str, gt_answer: str) -> float:
    """
    计算预测答案与真实答案的Jaccard相似系数（字符级），先进行中文文本清洗

    清洗操作：
    - 移除所有标点符号（中英文标点）
    - 移除所有空格（包括全角/半角空格）
    - 保留汉字、字母和数字
    - 转换为小写（处理可能的英文答案）

    参数:
    pred_answer (str): 预测答案字符串
    gt_answer (str): 真实答案字符串

    返回:
    float: Jaccard相似系数，范围[0, 1]
    """

    # 文本清洗函数
    def clean_text(text):
        # 1. 转换为小写（处理可能的英文答案）
        text = text.lower()
        # 2. 移除所有标点符号和空格
        # 匹配所有非汉字、非字母、非数字的字符
        cleaned = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
        return cleaned

    # 清洗两个答案
    clean_pred = clean_text(pred_answer)
    clean_gt = clean_text(gt_answer)

    # 转换为字符集合（自动去重）
    set_pred = set(clean_pred)
    set_gt = set(clean_gt)

    # 计算交集和并集大小
    intersection_size = len(set_pred & set_gt)
    union_size = len(set_pred | set_gt)

    # 处理全空情况（两个空字符串相似度为1.0）
    if union_size == 0:
        return 1.0

    # 计算Jaccard相似系数
    return intersection_size / union_size


if __name__ == '__main__':
    # main()  # batch 的情况下 15 分钟
    # parse_answers()
    eval_answer()
