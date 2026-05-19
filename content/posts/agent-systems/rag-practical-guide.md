---
title: "RAG实战：让AI不再胡说八道"
slug: rag-practical-guide
description: "RAG 实战指南:用检索增强生成让大模型「先查资料再回答」,有效减少幻觉,附向量数据库落地要点。"
date: 2026-01-12T10:00:00+08:00
draft: false
tags: ["RAG", "向量数据库", "LLM"]
categories: ["AI Agent"]
excerpt: "大模型最大的问题是会编造答案。RAG（检索增强生成）通过先搜索再回答的方式解决这个问题。这篇讲讲RAG怎么做、有什么坑。"
---

## RAG是什么

一句话：**先查资料，再回答问题**。

大模型直接回答问题容易编造内容。RAG让它先从你的知识库里找到相关内容，再基于这些内容回答。

```
用户问题 → 搜索知识库 → 找到相关文档 → 喂给LLM → 生成答案
```

---

## 最简实现

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# 1. 把文档切块并存入向量数据库
docs = load_and_split_documents("./docs")
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# 2. 检索相关内容
retriever = vectorstore.as_retriever(k=3)
relevant_docs = retriever.get_relevant_documents("什么是RAG？")

# 3. 生成答案
llm = ChatOpenAI()
answer = llm.invoke(f"""
根据以下内容回答问题：
{relevant_docs}

问题：什么是RAG？
""")
```

就这么简单。30行代码就能跑起来。

---

## 常见的坑

### 坑1：切块太大或太小

- **太大**：一块里混了好几个主题，检索不准
- **太小**：上下文断了，回答不完整

**建议**：500-1000字一块，重叠100-200字

### 坑2：只用向量检索

向量检索找语义相似的，但有时候用户就是要精确匹配。

**解决**：混合检索（向量 + 关键词BM25）

```python
# 向量检索 + 关键词检索，结果融合
vector_results = vector_search(query)
keyword_results = bm25_search(query)
final_results = fuse_results(vector_results, keyword_results)
```

### 坑3：检索结果不重排

检索出来的top5不一定按相关性排序。

**解决**：用CrossEncoder重排

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([[query, doc] for doc in docs])
```

### 坑4：塞太多上下文

上下文太长LLM反而会忽略关键信息。

**解决**：压缩上下文，只保留关键句子

---

## 评估RAG效果

两个维度：

1. **检索质量**：找到的内容对不对？（用Recall@K、MRR评估）
2. **生成质量**：回答是否忠实于检索内容？（人工评估或用LLM评判）

简单方法：准备100个问答对，跑一遍看效果。

---

## 什么时候用RAG

**适合：**
- 企业知识库问答
- 文档对话
- 客服系统
- 任何需要"查资料再回答"的场景

**不适合：**
- 通用聊天
- 创意写作
- 不需要外部知识的任务

---

## 工具推荐

| 场景 | 推荐 |
|------|------|
| 快速原型 | LangChain + ChromaDB |
| 生产部署 | LlamaIndex + Pinecone |
| 私有化部署 | Milvus / Qdrant |

---

## 最后

RAG不难，难的是调到好用。

建议：先跑起来，再一点点优化。别一开始就追求完美架构。

有问题留言。