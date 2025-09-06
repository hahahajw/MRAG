"""
index - 完成索引过程

Author - hahahajw
Date - 2025-08-22 
"""
from typing import (
    List,
    Dict,
    Optional
)
import json
from loguru import logger as log
import os
from uuid import uuid4
from tqdm import tqdm
from dashscope import MultiModalConversation

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableConfig

from modules.index.prompt import get_vlm_prompt, get_llm_prompt


class Index:
    def __init__(
            self,
            files_path: List[str],
            embed_model: OpenAIEmbeddings,
            llm: ChatOpenAI,
            vlm: ChatOpenAI,
            vector_store_name: str = 'mrag',
    ):
        """
        Args:
            embed_model (必须): 使用的嵌入模型
            llm: 解析表格使用的 LLM
            vlm: 解析图片使用的 VLM
            files_path: 要解析文件的位置（指定到具体文件比较好，如果你仅指定到文件夹，那么对于现在这样嵌套文件夹的情况就不好处理了）
            vector_store_name: 创建或加载的向量数据库名称，默认 'mrag'
        """
        self.embed_model = embed_model
        self.llm = llm
        self.vlm = vlm
        self.files_path = files_path
        self.vector_store_name = vector_store_name

        self.vector_store: Optional[Milvus] = None

    def get_index_done(
            self,
            chunk_size: int = 700,
            chunk_overlap: int = 50
    ):
        """
        将 chunk 存入向量数据库中
        Args:
            chunk_size: chunk 中文本块的大致长度
            chunk_overlap: chunk 间重叠字符的数量

        Returns:
            None, 创建完成的 vector store 可以通过 Index 实例的 vector store 属性得到
        """
        from pymilvus.client.types import LoadState

        # 1. 看要创建的数据库是否已经存在，存在则直接返回向量数据库
        # 如果已经存在了一个重名的数据库但没被加载到内存中，则下面的代码会将其加载到内存中
        # 如果已经存在了一个重名的数据库并且已经被加载到内存中，则下面的代码只是为对应的向量数据库创建了一个新的引用

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
            collection_name=self.vector_store_name,
            embedding_function=self.embed_model,
            builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
            vector_field=['dense', 'sparse'],
            index_params=[dense_index_parma, sparse_index_param]  # 顺序关系是对应的
        )

        client = vector_store.client
        state = client.get_load_state(collection_name=self.vector_store_name)
        # 当前的向量数据库还没有被创建过
        if state['state'] == LoadState.NotExist:
            pass
        else:
            log.info(f'{self.vector_store_name} 向量数据库已存在，成功加载到内存中')
            self.vector_store = vector_store
            return

        # 2. 不存在则读取所有数据读、分块、嵌入，并返回向量数据库
        self.add_new_files(
            vector_store=vector_store,
            new_files_path=self.files_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.vector_store = vector_store

        return

    def add_new_files(
            self,
            vector_store: Milvus,
            new_files_path: List[str],
            chunk_size: int = 700,
            chunk_overlap: int = 50
    ):
        """
        新增文件到向量数据库中
        Args:
            vector_store: 存入的向量数据库
            new_files_path: 新增文件的具体路径
            chunk_size: 这次 chunk 的大小
            chunk_overlap: 这次 chunk 的重叠
        """
        # 使用 MinerU 解析新增文件
        self.parse_pdf_with_mineru()

        # 定义处理链
        chain = (
                RunnableLambda(Index.group_content_list_by_page)
                | RunnableLambda(self.page_content_list_to_documents_json)
                | RunnableLambda(Index.documents_json_to_documents)
                | RunnableLambda(Index.documents_to_chunks).bind(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )

        # 处理单个文件
        for cur_file_path in new_files_path:
            log.info(f'正在处理文件 {cur_file_path}')

            chunks = chain.invoke(cur_file_path)
            cur_file_name = chunks[0].metadata['filename']

            # 存入向量数据库
            for i in tqdm(range(0, len(chunks), 10), desc=f'处理文件 {cur_file_name}'):
                cur_chunks = chunks[i:i + 10]
                vector_store.add_documents(
                    documents=cur_chunks,
                    ids=[str(uuid4()) for _ in range(len(cur_chunks))]
                )

    def add_new_files_batch(
            self,
            vector_store: Milvus,
            new_files_path: List[str],
            chunk_size: int = 700,
            chunk_overlap: int = 50
    ):
        """
        以新增文件的页面为单位并行添加到向量数据库中
        Args:
            vector_store: 存入的向量数据库
            new_files_path: 新增文件的具体路径
            chunk_size: 这次 chunk 的大小
            chunk_overlap: 这次 chunk 的重叠
        """
        # 使用 MinerU 解析新增文件
        self.parse_pdf_with_mineru()

        # 将所有新文件的 content list 按页组织，item 中有后续处理所需的全部信息
        page_content_list = []
        for cur_file_path in new_files_path:
            page_content_list += Index.group_content_list_by_page(cur_file_path)
        log.info(f'所有文件以按页处理')

        # 定义处理链
        chain = (
                RunnableLambda(self.page_content_list_to_documents_json_batch)
                | RunnableLambda(Index.documents_json_to_documents)
                | RunnableLambda(Index.documents_to_chunks).bind(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )

        # 批处理页面
        # 返回值将是 List[List[Document]]，列表中的每个元素是某个页面的 chunk

        chunks = chain.batch(page_content_list, config=RunnableConfig(max_concurrency=21), return_exceptions=True)

        # 存入向量数据库
        for page_chunks in tqdm(chunks, desc='存入向量数据库'):
            if page_chunks:
                for i in range(0, len(page_chunks), 10):
                    cur_chunks = page_chunks[i:i + 10]
                    vector_store.add_documents(
                        documents=cur_chunks,
                        ids=[str(uuid4()) for _ in range(len(cur_chunks))]
                    )
            else:
                log.error(f'error: {page_chunks}')

        return chunks

    def page_content_list_to_documents_json_batch(
            self,
            page_content_list: List[Dict]
    ) -> List[Dict]:
        """
        这个函数只会处理某一页的内容，而不是处理整个文件的内容

        把 MinerU 的解析结果转化为可以保存到 Document 对象的 JSON 文件
        每个 JSON 对象都可以被加载到一个 Document 对象中，我们会将
        1. 每个页面所有的文字内容整合到一个 Document 中
        2. 每个图片内容整合到一个 Document 中
        3. 每个表格内容整合到一个 Document 中

        因为会使用 VLM、LLM 解析图片和表格，所以持久化更好些
        Args:
            self:   self.vlm: 解析图片使用的 VLM
                    self.llm: 解析表格使用的 LLM
            page_content_list (List[Dict]]): 某个页面的可读块列表
        Returns:
            List[Dict]，每个元素是一个 JSON 对象，包含了一个 Document 的信息
        """

        documents_json_list = []
        cur_file_name = page_content_list[0]['filename']
        page_content = ''
        page_idx = page_content_list[0]['page_idx']

        # 处理某个页面
        for item in page_content_list:
            if item['type'] == 'text':
                text_level = item.get('text_level', 0)
                # 正文
                if text_level == 0:
                    # MinerU 的解析结果可以保证一块完整的文字在一个 item 中
                    page_content += item['text'] + '\n'
                # 第 text_level 级标题
                else:
                    page_content += f'{"#" * text_level} {item["text"]}\n'

            elif item['type'] == 'image':
                # 图片有标题
                if item['image_caption'] and item['img_path'] != '':
                    documents_json_list.append(
                        self.image_to_text(item)
                    )
                    # log.info(f'已将图片 {item["image_caption"]} 转换为文本')  # 注释掉

            elif item['type'] == 'table':
                # 图表有标题
                if item['table_caption']:
                    documents_json_list.append(
                        self.table_to_text(item)
                    )
                    # log.info(f'已将表格 {item["table_caption"]} 转换为文本')  # 注释掉

            else:
                log.error(f'未知的可读块类型: {item["type"]}')
                pass

        # 把当前页面的文字内容整合到一个 Document 中
        documents_json_list.append({
            'page_content': page_content,
            'metadata': {
                'filename': cur_file_name,
                'page': page_idx,
                'type': 'text',
                'img_path': 'None',
                'table_body': 'None'
            }
        })
        # log.info(f'已将页面 {page_idx} 的文本内容整合到一个 Document 中')  # 注释掉
        # log.info(f'{cur_file_name}: {page_idx}')

        # 持久化
        output_file_path = f'../../res/{cur_file_name.split(".")[0]}/documents_{page_idx}.json'
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(documents_json_list, f, ensure_ascii=False, indent=4)
        # log.info(f'已将 {cur_file_name} 第 {page_idx} 的 Document 对象持久化到 {output_file_path}')
        log.info(f'{page_idx}: {output_file_path}')

        return documents_json_list

    def parse_pdf_with_mineru(self):
        """
        使用 MinerU 解析 PDF 文件
        Returns:
            None，最终的结果持久化到根目录下的 output 文件夹下
        """
        # 由于本地硬件资源有限，这一部分在云端完成
        # 具体思路和完成代码在 https://icnktcoe2mwu.feishu.cn/docx/JoOwdAnBToVzsLxWYW1cuOAOn2d#share-KpH8djyXOoawIfx5vPQcCkW2nYg
        pass

    @staticmethod
    def group_content_list_by_page(cur_file_path: str) -> List[List]:
        """
        将 MinerU 解析的 {文件名}_content_list.json 文件中的内容按页面分组
        Args:
            cur_file_path: 当前处理文件的路径
        Returns:
            List[List]，[[属于第一页的可读块], [], ...]
        """
        # log.info(f'--- group_content_list_by_page ---')

        # 读取当前文件的解析结果
        with open(cur_file_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)

        # 按页面分组
        cur_file_name = cur_file_path.split('/')[-3] + '.pdf'
        page_content_list, tmp = [], []
        cur_page_index = 0
        for item in content_list:
            # 为每个可读块添加文件来源
            item['filename'] = cur_file_name

            if item['page_idx'] == cur_page_index:
                tmp.append(item)
            else:
                page_content_list.append(tmp)
                tmp = [item]
                cur_page_index = item['page_idx']

        return page_content_list

    def page_content_list_to_documents_json(
            self,
            page_content_list: List[List[Dict]]
    ) -> List[Dict]:
        """
        把 MinerU 的解析结果转化为可以保存到 Document 对象的 JSON 文件
        每个 JSON 对象都可以被加载到一个 Document 对象中，我们会将
        1. 每个页面所有的文字内容整合到一个 Document 中
        2. 每个图片内容整合到一个 Document 中
        3. 每个表格内容整合到一个 Document 中

        因为会使用 VLM、LLM 解析图片和表格，所以持久化更好些
        Args:
            self:   self.vlm: 解析图片使用的 VLM
                    self.llm: 解析表格使用的 LLM
            page_content_list (List[List]]): 每个页面的可读块列表
        Returns:
            List[Dict]，每个元素是一个 JSON 对象，包含了一个 Document 的信息
        """
        log.info(f'--- page_content_list_to_documents_json ---')

        documents_json_list = []
        cur_file_name = page_content_list[0][0]['filename']

        for page_idx, items in enumerate(page_content_list):
            # 处理当前页面的可读块
            page_content = ''
            for item in items:
                if item['type'] == 'text':
                    text_level = item.get('text_level', 0)
                    # 正文
                    if text_level == 0:
                        # MinerU 的解析结果可以保证一块完整的文字在一个 item 中
                        page_content += item['text'] + '\n'
                    # 第 text_level 级标题
                    else:
                        page_content += f'{"#" * text_level} {item["text"]}\n'

                elif item['type'] == 'image':
                    # 图片有标题
                    if item['image_caption']:
                        documents_json_list.append(
                            self.image_to_text(item)
                        )
                        # log.info(f'已将图片 {item["image_caption"]} 转换为文本')  # 注释掉

                elif item['type'] == 'table':
                    # 图表有标题
                    if item['table_caption']:
                        documents_json_list.append(
                            self.table_to_text(item)
                        )
                        # log.info(f'已将表格 {item["table_caption"]} 转换为文本')  # 注释掉

                else:
                    log.error(f'未知的可读块类型: {item["type"]}')
                    pass

            # 把当前页面的文字内容整合到一个 Document 中
            documents_json_list.append({
                'page_content': page_content,
                'metadata': {
                    'filename': cur_file_name,
                    'page': page_idx,
                    'type': 'text',
                    'img_path': 'None',
                    'table_body': 'None'
                }
            })
            # log.info(f'已将页面 {page_idx} 的文本内容整合到一个 Document 中')  # 注释掉
            log.info(f'{page_idx}')

        # 持久化
        output_file_path = f'../../output/{cur_file_name.split(".")[0]}/documents.json'
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(documents_json_list, f, ensure_ascii=False, indent=4)
        log.info(f'已将 {cur_file_name} 的 Document 对象持久化到 {output_file_path}')

        return documents_json_list

    @staticmethod
    def documents_json_to_documents(documents_json_list: List[Dict]) -> List[Document]:
        """
        将 documents_json_list 中的 JSON 对象转化为 Document 对象
        Args:
            documents_json_list (List[Dict]): 每个元素是一个 JSON 对象，包含了一个 Document 的信息
        Returns:
            List[Document]，每个元素是一个 Document 对象
        """
        # log.info(f'--- documents_json_to_documents ---')

        return [
            Document(
                page_content=doc['page_content'],
                metadata=doc['metadata']
            ) for doc in documents_json_list
        ]

    @staticmethod
    def documents_to_chunks(
            documents: List[Document],
            chunk_size: int,
            chunk_overlap: int,
    ) -> List[Document]:
        """
        分块
        Args:
            documents: 较长的文档
            chunk_size: 每个 chunk 的大小
            chunk_overlap: 每个 chunk 之间的重叠大小

        Returns:
            List[Document]，分块后的文档
        """
        # log.info(f'--- documents_to_chunks ---')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            length_function=len,  # 计算文本长度的方式，最后划分出的 chunk 不严谨满足这个长度。因为不是以长度为标准进行的划分
            separators=[
                "\n\n",
                "\n",
                "。",
                ".",
                " ",
                ""
            ]  # 划分文本所使用的分隔符。添加了中文的句号和逗号，以便更好的划分中文文本。此外，分隔符的顺序也是有意义的
        )

        return text_splitter.split_documents(documents)

    def image_to_text(
            self,
            item: Dict,
    ) -> Dict:
        # 向 VLM 中传入本地文件：https://help.aliyun.com/zh/model-studio/vision/?mode=pure#d987f8de5395x
        # 调用 VLM
        prompt = get_vlm_prompt(item)
        try:
            response = MultiModalConversation.call(
                api_key=os.getenv('BL_API_KEY'),
                # model='qwen-vl-max-2025-08-13',
                model='qwen-vl-max-latest',
                messages=prompt,
                vl_high_resolution_images=True  # 启用高分辨率图像理解，图像将被编码为更多的 token
            )

            page_content = response["output"]["choices"][0]["message"].content[0]["text"]
        except Exception as e:
            log.error(f'图片解析错误：{e}\nitem: {item}')
            page_content = f'图片解析错误\nitem: {item}'

        return {
            'page_content': page_content,
            'metadata': {
                'filename': item['filename'],
                'page': item['page_idx'],
                'type': 'image',
                'img_path': item['img_path'],
                'table_body': 'None'
            }
        }

    def table_to_text(
            self,
            item: Dict,
    ) -> Dict:
        # 调用 LLM
        prompt = get_llm_prompt(item)
        try:
            response = self.llm.invoke(prompt)
            page_content = response.content
        except Exception as e:
            log.error(f'表格解析错误：{e}\nitem: {item}')
            page_content = f'表格解析错误\nitem: {item}'

        return {
            'page_content': page_content,
            'metadata': {
                'filename': item['filename'],
                'page': item['page_idx'],
                'type': 'table',
                'img_path': item['img_path'],
                'table_body': item['table_body']
            }
        }


if __name__ == '__main__':
    import time
    from dotenv import load_dotenv

    load_dotenv()

    _embed_model = OpenAIEmbeddings(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model="text-embedding-v4",
        dimensions=1024,
        check_embedding_ctx_length=False
    )
    _vlm = ChatOpenAI(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model='qwen-vl-max-2025-08-13',
        temperature=0.0
    )
    _llm = ChatOpenAI(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model='qwen-plus-latest',
        temperature=0.0
    )

    # file_name = '联邦制药-港股公司研究报告-创新突破三靶点战略联姻诺和诺德-25071225页'
    # all_files_path = [f'../../data/{file_name}/auto/{file_name}_content_list.json']
    # file_name = '极兔速递W-港股公司研究报告-系列一东南亚十年磨砺终成锋产业经营双拐点-25070834页'
    # all_files_path = [f'../../output/{file_name}/auto/{file_name}_content_list.json']
    # file_name = '凌云股份-公司深度研究报告热成型电池盒双轮驱动传感器加速布局-25071427页'
    # all_files_path = [f'../../output/{file_name}/auto/{file_name}_content_list.json']

    files_to_remove = [
        '联邦制药-港股公司研究报告-创新突破三靶点战略联姻诺和诺德-25071225页',
        '极兔速递W-港股公司研究报告-系列一东南亚十年磨砺终成锋产业经营双拐点-25070834页',
        '内蒙古伊利实业集团股份有限公司2022年年度报告264页',
        '内蒙古伊利实业集团股份有限公司2023年半年度报告212页',
        '内蒙古伊利实业集团股份有限公司2023年第三季度报告19页',
        '内蒙古伊利实业集团股份有限公司2023年第一季度报告19页',
        '凌云股份-公司深度研究报告热成型电池盒双轮驱动传感器加速布局-25071427页',
        '亚翔集成-公司研究报告-迎接海外业务重估-25071324页',
        '中恒电气-公司研究报告-HVDC方案领头羊AI浪潮下迎新机-25071124页',
        '亿欧中国企业出海沙特季度研究报告-AI专题2025Q267页',
        '伊利专业乳品2022中国现制茶饮渠道消费者与行业趋势报告40页',
        '伊利股份-产业周期叠加内生动力业绩增速向上-21093038页',
        '伊利股份-产品升级叠加费用率下降盈利能力持续提升-22041256页',
        '伊利股份-公司深度报告王者荣耀行稳致远-22021459页',
        '伊利股份-公司研究专题报告高股息铸盾景气度为矛当前重点推荐-23051519页',
        '伊利股份-公司研究报告-平台化的乳企龙头引领行业高质量转型-25071638页',
        '伊利股份-公司研究报告黑夜终将过去把握高股息低估值乳品龙头机会-24110830页',
        '伊利股份-大象起舞龙头远航-2020072627页',
        '伊利股份-格局之变提供发展机遇内生外延打造第二曲线-22052743页',
        '伊利股份-王者之路扶摇而上-21083136页',
        # '千味央厨-百味千寻餐饮供应链龙头正崛起-21091628页',
        # '千味央厨-公司深度报告深耕蓝海鹏程万里-22110953页'

        # '../../output/联邦制药-港股公司研究报告-创新突破三靶点战略联姻诺和诺德-25071225页/auto/联邦制药-港股公司研究报告-创新突破三靶点战略联姻诺和诺德-25071225页_content_list.json',
        # '../../output/极兔速递W-港股公司研究报告-系列一东南亚十年磨砺终成锋产业经营双拐点-25070834页/auto/极兔速递W-港股公司研究报告-系列一东南亚十年磨砺终成锋产业经营双拐点-25070834页_content_list.json',
        # '../../output/内蒙古伊利实业集团股份有限公司2022年年度报告264页/auto/内蒙古伊利实业集团股份有限公司2022年年度报告264页_content_list.json',
        # '../../output内蒙古伊利实业集团股份有限公司2023年半年度报告212页/auto/内蒙古伊利实业集团股份有限公司2023年半年度报告212页_content_list.json',
        # '../../output/内蒙古伊利实业集团股份有限公司2023年第三季度报告19页/auto/内蒙古伊利实业集团股份有限公司2023年第三季度报告19页_content_list.json',
        # '../../output/内蒙古伊利实业集团股份有限公司2023年第一季度报告19页/auto/内蒙古伊利实业集团股份有限公司2023年第一季度报告19页_content_list.json',
        # '../../output/凌云股份-公司深度研究报告热成型电池盒双轮驱动传感器加速布局-25071427页/auto/凌云股份-公司深度研究报告热成型电池盒双轮驱动传感器加速布局-25071427页_content_list.json'
    ]
    tmp = os.listdir('../../res')
    files_to_remove += tmp

    all_files_path = os.listdir('../../output')
    for file_to_remove in files_to_remove:
        all_files_path.remove(file_to_remove)

    for i, file_name in enumerate(all_files_path):
        all_files_path[i] = f'../../output/{file_name}/auto/{file_name}_content_list.json'

    print(len(all_files_path))

    mrag_index = Index(
        files_path=all_files_path,
        embed_model=_embed_model,
        vlm=_vlm,
        llm=_llm,
        vector_store_name='mrag'
    )

    mrag_index.get_index_done()

    # mrag_index.add_new_files(
    #     vector_store=mrag_index.vector_store,
    #     new_files_path=all_files_path,
    #     chunk_size=700,
    #     chunk_overlap=50
    # )

    # add_chain = RunnableLambda(mrag_index.add_new_files_batch)
    # for i in range(0,  len(all_files_path), 10):
    #     cur_files_path = all_files_path[i:i+10]
    #     add_chain.batch([cur_files_path])
    for file_path in all_files_path:
        log.info(f'--- {file_path} ---')
        mrag_index.add_new_files_batch(
            vector_store=mrag_index.vector_store,
            new_files_path=[file_path],
            chunk_size=700,
            chunk_overlap=50
        )

        time.sleep(12)
