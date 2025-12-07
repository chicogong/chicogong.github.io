---
title: "RAG系统从零搭建：让AI拥有「私人知识库」"
date: 2025-12-18T10:00:00+08:00
draft: false
tags: ["RAG", "Retrieval Augmented Generation", "向量数据库", "LangChain", "Embedding", "实用教程"]
categories: ["开发指南", "AI Agent"]
excerpt: "让ChatGPT读懂你的公司文档！本文手把手教你搭建RAG系统，从文档切块、向量嵌入到语义检索，完整代码+避坑指南，新手也能2小时上手。"
---

## 开场：老板的「灵魂拷问」

**老板**："小王，能不能让ChatGPT回答关于我们公司产品的问题？"

**小王**："额...ChatGPT不知道我们公司的事情啊..."

**老板**："那就教它！"

**小王**："..."

这个场景你熟悉吗？别慌，**RAG（Retrieval Augmented Generation）** 就是为了解决这个问题而生的。

简单说，RAG = **让AI读完你的文档后再回答问题**。

---

## 第一章：RAG是什么？一张图说清楚

![RAG架构图](/images/tutorials/rag-architecture.png)

### 核心流程

```
用户提问 → 语义搜索相关文档 → 把文档+问题一起发给LLM → 生成回答
```

### 类比理解

想象你是一个**开卷考试的学生**：
- **传统LLM**：闭卷考试，只能靠脑子里记住的知识
- **RAG**：开卷考试，可以翻书查资料再回答

**RAG的本质：用检索代替记忆。**

---

## 第二章：动手搭建 —— 完整代码

### Step 1: 环境准备

```bash
# 安装依赖
pip install langchain langchain-openai chromadb tiktoken
```

### Step 2: 文档加载与切块

```python
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加载文档
def load_documents(path: str):
    """加载指定目录下的所有文档"""
    loader = DirectoryLoader(
        path,
        glob="**/*.txt",  # 匹配所有txt文件
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"📚 加载了 {len(documents)} 个文档")
    return documents

# 文档切块
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """将长文档切分成小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # 每块500字符
        chunk_overlap=chunk_overlap, # 块之间重叠50字符
        separators=["\n\n", "\n", "。", "！", "？", "，", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️ 切分成 {len(chunks)} 个文本块")
    return chunks

# 使用示例
docs = load_documents("./my_documents")
chunks = split_documents(docs)
```

### 切块策略对比

| 策略 | 适用场景 | 推荐大小 |
|:---|:---|:---|
| **小块** (256-512) | 精确问答、FAQ | 追求精度 |
| **中块** (512-1000) | 通用场景 | 平衡选择 |
| **大块** (1000-2000) | 需要上下文 | 追求完整性 |

**黄金法则**：重叠10-20%，避免信息被切断。

### Step 3: 向量嵌入与存储

```python
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 推荐：便宜且效果好
    # model="text-embedding-3-large",  # 更强但更贵
)

# 创建向量数据库
def create_vector_store(chunks, persist_directory="./chroma_db"):
    """将文本块转换为向量并存储"""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()  # 持久化到磁盘
    print(f"💾 向量数据库已保存到 {persist_directory}")
    return vectorstore

# 加载已有的向量数据库
def load_vector_store(persist_directory="./chroma_db"):
    """加载已存在的向量数据库"""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

# 使用
vectorstore = create_vector_store(chunks)
```

### Step 4: 检索与生成

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 初始化LLM
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.2  # 降低随机性，增加准确性
)

# 自定义Prompt模板
PROMPT_TEMPLATE = """
你是一个专业的知识库助手。根据以下提供的上下文信息回答用户问题。

上下文信息：
{context}

用户问题：{question}

回答要求：
1. 只根据上下文信息回答，不要编造
2. 如果上下文中没有相关信息，请明确告知
3. 回答要简洁、专业

回答：
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# 创建检索链
def create_qa_chain(vectorstore):
    """创建问答链"""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # 检索最相关的3个文档块
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 把所有检索到的内容塞给LLM
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# 使用
qa_chain = create_qa_chain(vectorstore)
```

### Step 5: 开始问答！

```python
def ask(question: str):
    """提问并获取回答"""
    result = qa_chain({"query": question})
    
    print("🤖 回答：")
    print(result["result"])
    
    print("\n📄 参考来源：")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"  [{i}] {doc.metadata.get('source', '未知')} - {doc.page_content[:100]}...")
    
    return result

# 示例
ask("我们公司的退款政策是什么？")
ask("产品保修期是多久？")
```

---

## 第三章：进阶优化 —— 让RAG更聪明

### 优化1：混合检索（Hybrid Search）

语义搜索有时会错过关键词匹配，混合检索两者的优点：

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# BM25关键词检索
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# 向量语义检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 混合检索器
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 语义权重高一些
)
```

### 优化2：重排序（Reranking）

检索后，用更强的模型对结果重新排序：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# 使用Cross-Encoder重排序
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3
)

# 组合检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever
)
```

### 优化3：查询扩展（Query Expansion）

用LLM改写用户问题，提高检索效果：

```python
def expand_query(original_query: str) -> list:
    """用LLM扩展原始查询"""
    expansion_prompt = f"""
    原始问题：{original_query}
    
    请生成3个语义相似但表达不同的问题，用于搜索相关文档：
    1.
    2.
    3.
    """
    
    response = llm.invoke(expansion_prompt)
    expanded_queries = [original_query] + parse_queries(response)
    return expanded_queries
```

---

## 第四章：避坑指南

### 坑1：文档切块太大或太小

**症状**：回答不准确或缺少上下文

**解决**：
- 尝试不同的chunk_size（256/512/1024）
- 增加chunk_overlap防止信息切断
- 对不同类型文档用不同策略

### 坑2：检索到了但回答错了

**症状**：检索到了相关文档，但LLM理解错误

**解决**：
- 优化Prompt，明确告诉LLM只用上下文回答
- 增加检索数量（k=5）
- 使用更强的LLM（GPT-4 > GPT-3.5）

### 坑3：检索不到相关内容

**症状**：明明有相关文档，但检索不到

**解决**：
- 换更好的Embedding模型
- 使用混合检索
- 检查文档预处理（去掉噪音）

### 坑4：响应太慢

**症状**：等待时间过长

**解决**：
- 减少检索数量（k=3足够）
- 使用更快的Embedding（text-embedding-3-small）
- 考虑异步处理

---

## 第五章：生产部署清单

### ✅ 上线前检查

| 检查项 | 说明 |
|:---|:---|
| 文档更新机制 | 新文档如何自动入库？ |
| 向量库备份 | 定期备份ChromaDB |
| 错误处理 | 检索失败、LLM超时怎么办？ |
| 监控指标 | 检索准确率、响应时间、用户满意度 |
| 成本控制 | Embedding和LLM的API费用 |

### 推荐技术栈

| 组件 | 开发环境 | 生产环境 |
|:---|:---|:---|
| **向量数据库** | ChromaDB | Pinecone / Weaviate / Milvus |
| **Embedding** | OpenAI | OpenAI / Cohere / 自建 |
| **LLM** | GPT-4 | GPT-4 / Claude / 自建 |
| **框架** | LangChain | LangChain / LlamaIndex |

---

## 完整代码汇总

```python
"""
RAG系统完整实现
运行：python rag_demo.py
"""

import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 配置
os.environ["OPENAI_API_KEY"] = "your-api-key"
DOCS_PATH = "./documents"
DB_PATH = "./chroma_db"

def main():
    # 1. 加载文档
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"📚 加载 {len(docs)} 个文档")
    
    # 2. 切块
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✂️ 切分 {len(chunks)} 个文本块")
    
    # 3. 向量化存储
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    vectorstore.persist()
    print("💾 向量数据库已保存")
    
    # 4. 创建问答链
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2)
    
    prompt = PromptTemplate(
        template="根据以下上下文回答问题。\n上下文：{context}\n问题：{question}\n回答：",
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # 5. 交互式问答
    print("\n🤖 RAG系统就绪！输入问题开始对话（输入q退出）\n")
    while True:
        question = input("你: ")
        if question.lower() == 'q':
            break
        result = qa_chain({"query": question})
        print(f"AI: {result['result']}\n")

if __name__ == "__main__":
    main()
```

---

## 结语

RAG是让AI"私人定制"的最佳方案。不需要昂贵的微调，不需要重新训练模型，只需要把你的文档喂给它。

**3个要点带回去**：
1. **切块是关键**：500字符+50重叠是好的起点
2. **混合检索更强**：语义+关键词双保险
3. **Prompt很重要**：明确告诉LLM"只用上下文回答"

现在，去让你的ChatGPT变成"公司百科全书"吧！🚀

