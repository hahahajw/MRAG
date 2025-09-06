"""
workflow - Naive RAG 的 workflow

Author - hahahajw
Date - 2025-05-26 
"""
from naive_rag.state import NaiveRagState
import naive_rag.nodes as nodes
from langgraph.graph import START, StateGraph, END


def get_naive_rag_workflow() -> StateGraph:
    """
    获得 Naive RAG 的 workflow

    Returns:
        StateGraph: 利用 LangGraph 定义的 workflow
    """
    wf = StateGraph(NaiveRagState)

    wf.add_node('retrieve', nodes.retriever_node)
    wf.add_node('augmented', nodes.augmented_node)
    wf.add_node('generate', nodes.generate_node)

    wf.add_edge(START, 'retrieve')
    wf.add_edge('retrieve', 'augmented')
    wf.add_edge('augmented', 'generate')
    wf.add_edge('generate', END)

    return wf
