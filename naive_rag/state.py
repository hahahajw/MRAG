"""
state - 定义 NaiveRAG 所使用的 state

Author - hahahajw
Date - 2025-05-26 
"""
from typing import List
from langchain_core.documents import Document
from langgraph.graph import MessagesState


class NaiveRagState(MessagesState):
    query: str  # 用户的问题
    similar_chunks: List[Document]  # 检索到的文档
    answer: str  # 最终的回答
    filename: str
    page: int
