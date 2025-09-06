"""
nodes - 定义组成 NaiveRAG 所需的节点

Author - hahahajw
Date - 2025-05-26 
"""
from naive_rag.state import NaiveRagState
from naive_rag.prompt import get_sys_prompt

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

import os
import json
from pydantic import BaseModel, Field
from loguru import logger as log


def retriever_node(state: NaiveRagState,
                   config: RunnableConfig) -> NaiveRagState:
    """
    根据用户问题检索到相应的文档

    Args:
        state: 用户的输入，state['query']
        config: 检索器，config['configurable']['retriever']

    Returns:
        NaiveRagState['similar_chunks']: 更新 NaiveRagState 下的 similar_chunks 通道
    """
    retriever = config['configurable']['retriever']
    similar_chunks_with_scores = retriever.get_similar_chunk_with_score(query=state['query'])

    return {'similar_chunks': [similar_chunk for similar_chunk, score in similar_chunks_with_scores]}  # type: ignore


def augmented_node(state: NaiveRagState) -> NaiveRagState:
    """
    将检索到的内容经过处理后填入到准备好的 prompt 中

    Args:
        state: 检索到的文档，state['similar_chunks']
               用户问题，state['query']

    Returns:
        NaiveRagState['messages']: 将处理好的信息添加到消息 list 中
    """
    sys_prompt = get_sys_prompt()

    similar_chunks = state['similar_chunks']
    human_meg_content = f"""[检索结果]
文件：{similar_chunks[0].metadata['filename']}
页码：{similar_chunks[0].metadata['page']}
内容：{similar_chunks[0].page_content}

文件：{similar_chunks[1].metadata['filename']}
页码：{similar_chunks[1].metadata['page']}
内容：{similar_chunks[1].page_content}

文件：{similar_chunks[2].metadata['filename']}
页码：{similar_chunks[2].metadata['page']}
内容：{similar_chunks[2].page_content}

[问题]
{state['query']}

[回答]"""

    input_meg = [sys_prompt] + [HumanMessage(content=human_meg_content)]

    return {'messages': input_meg}  # type: ignore


def generate_node(state: NaiveRagState,
                  config: RunnableConfig) -> NaiveRagState:
    """
    回答最终的问题

    Args:
        state: 增强阶段更新的消息，state['messages']
        config: 要使用的 LLM，config['configurable']['llm']

    Returns:
        NaiveRagState['messages']: LLM 生成的完整消息
        NaiveRagState['answer']: LLM 生成的消息内容
    """
    class AnswerInf(BaseModel):
        filename: str = Field(description='所选页面的完整文件名（与输入一致）')
        page: int = Field(description='所选页面的页码（整数）')
        answer: str = Field(description='完整的问题答案，100 - 300字')

    llm = config['configurable']['llm']
    llm = llm.with_structured_output(AnswerInf)

    response = llm.invoke(state['messages'])

    # 把答案持久化
    tmp = {
        'query': state['query'],
        'answer': response.answer,
        'similar_chunks': [{'page_content': similar_chunk.page_content, 'metadata': similar_chunk.metadata} for similar_chunk in state['similar_chunks']],
        'filename': response.filename,
        'page': response.page,
    }
    # output_file_path = f'../answer/{state["query"]}.json'
    output_file_path = f'../train_answer/{state["query"]}.json'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(tmp, f, ensure_ascii=False, indent=4)

    log.info(f'{state["query"]}')

    return {'answer': response.answer, 'filename': response.filename, 'page': response.page}  # type: ignore
