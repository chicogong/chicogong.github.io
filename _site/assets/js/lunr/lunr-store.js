var store = [{
        "title": "欢迎来到我的博客",
        "excerpt":"欢迎来到我的个人博客   你好！欢迎来到我的个人博客。这是我在这个平台上发布的第一篇文章，我想借此机会介绍一下自己，以及创建这个博客的初衷。   关于我   我是 Chico Gong，一个热爱技术和分享的开发者。在技术的世界里，我专注于前端开发、后端架构，以及不断涌现的新兴技术。除了编程，我也对设计、产品思维和创业有着浓厚的兴趣。   我相信知识的力量，更相信分享知识的价值。在学习和工作的过程中，我深深体会到开源社区和技术博客对个人成长的巨大帮助。现在，我希望通过这个博客，将自己的学习心得、技术探索和生活感悟分享给更多的人。   为什么创建这个博客   创建这个博客有几个原因：      记录学习历程：技术发展日新月异，通过写博客可以帮助我整理思路，加深对知识的理解。   分享经验教训：在开发过程中遇到的问题和解决方案，希望能够帮助到其他遇到类似问题的朋友。   建立连接：希望通过博客认识更多志同道合的朋友，一起交流学习，共同成长。   提升表达能力：写作是一种很好的思维训练，可以帮助我更好地表达想法和观点。   博客内容规划   在这个博客里，我计划分享以下几类内容：   技术文章   包括但不限于：     前端开发技术（React, Vue, JavaScript, CSS等）   后端开发经验（Node.js, Python, 数据库设计等）   工具和效率（开发工具使用、工作流优化等）   新技术探索（AI/ML、Web3、移动开发等）   项目分享   我会分享一些自己做过的有趣项目，包括项目的创意来源、技术选型、开发过程中遇到的挑战和解决方案。   学习笔记   读书笔记、课程学习心得、会议参会感受等，希望通过分享促进更深入的思考。   生活感悟   技术之外的思考，关于工作、生活、成长的一些想法和体会。   写在最后   这个博客刚刚起步，内容和设计都还在不断完善中。如果你有任何建议或想法，非常欢迎与我交流。你可以通过邮件、GitHub 或其他社交媒体联系我。   感谢你花时间阅读这篇文章，希望在接下来的日子里，我们能在这个小小的数字空间里，一起学习、成长和探索。   期待与你的相遇！     2024年1月1日  ","categories": ["博客","介绍"],
        "tags": ["博客","介绍","开始"],
        "url": "/%E5%8D%9A%E5%AE%A2/%E4%BB%8B%E7%BB%8D/welcome-to-my-blog/",
        "teaser": null
      },{
        "title": "AI语音通话系统开发实战：从零构建智能语音交互平台",
        "excerpt":"前言 随着ChatGPT、Claude等大语言模型的兴起，AI语音交互已成为下一代人机交互的重要方向。本文将从零开始，带你构建一个完整的AI语音通话系统，实现人与AI的自然语音对话。 什么是AI语音通话系统 AI语音通话系统是一个集成了多种先进技术的智能交互平台，主要包括： 实时语音通信：基于WebRTC的低延迟音频传输 语音识别(ASR)：将语音转换为文本 自然语言理解(NLU)：理解用户意图和语义 对话管理：维护对话上下文和状态 语音合成(TTS)：将AI回复转换为自然语音 系统架构设计 整体架构 graph TB A[用户] --&gt; B[WebRTC客户端] B --&gt; C[信令服务器] C --&gt; D[媒体服务器] D --&gt; E[语音识别服务] E --&gt; F[AI对话引擎] F --&gt; G[语音合成服务] G --&gt; D D --&gt; B B --&gt; A 核心组件 前端WebRTC客户端 音频采集和播放 实时音频传输 用户界面交互 后端服务集群 信令服务器（WebSocket/Socket.io） 媒体处理服务器 AI对话引擎 语音处理服务...","categories": ["AI技术","语音通话"],
        "tags": ["WebRTC","语音识别","AI","实时通信"],
        "url": "/ai%E6%8A%80%E6%9C%AF/%E8%AF%AD%E9%9F%B3%E9%80%9A%E8%AF%9D/github-pages-setup-guide/",
        "teaser": "/assets/images/ai-voice-call-teaser.jpg"
      },{
        "title": "实时Agent系统技术演进与应用前景",
        "excerpt":"前言 随着人工智能技术的快速发展，实时Agent系统正在成为推动产业智能化升级的核心驱动力。从多模态感知到自主决策，从工作流协作到人机协同，Agent技术正在重新定义人机交互的未来。本文将深入探讨实时Agent系统的技术演进路径、应用场景及其对未来产业发展的深远影响。 一、实时Agent系统的技术架构与核心原理 1.1 多模态与自主决策技术突破 实时Agent系统的核心能力建立在多模态感知与自主决策的技术融合上。根据斯坦福大学的研究，多模态智能体通过整合视觉、听觉等传感器数据，实现了对物理和虚拟环境的交互式理解。 技术亮点： GPT-4多模态融合：通过插件系统调用外部工具时，需同步处理文本指令与图像数据，其多模态融合准确率较单模态提升40%以上 硬件加速优化：GPU/TPU集群将推理延迟从秒级降至毫秒级 算法优化：知识蒸馏技术使模型参数量减少70%的同时保持90%的原始性能 实时Agent系统架构图： graph TD A[用户输入] --&gt; B[语音识别ASR] B --&gt; C[自然语言理解NLU] C --&gt; D[Agent决策引擎] D --&gt; E[任务执行模块] D --&gt; F[知识库查询] E --&gt; G[自然语言生成NLG] F --&gt; G G --&gt; H[语音合成TTS] H --&gt; I[实时输出] subgraph \"核心处理层\" D E F end subgraph \"感知层\" A B...","categories": ["AI技术","Agent系统"],
        "tags": ["人工智能","多模态技术","实时系统","商业应用","技术架构"],
        "url": "/ai%E6%8A%80%E6%9C%AF/agent%E7%B3%BB%E7%BB%9F/realtime-agent-systems-evolution/",
        "teaser": "/assets/images/agent-systems-teaser.jpg"
      },{
        "title": "LangChain Graph 详解：构建智能知识图谱",
        "excerpt":"引言 在人工智能和大语言模型(LLM)的应用中，知识的表示与组织方式直接影响系统的推理能力和智能水平。LangChain Graph 作为LangChain生态系统中的重要组件，提供了一套强大的工具，使开发者能够轻松地从文本中提取结构化知识，构建知识图谱，并基于图进行复杂推理。本文将深入探讨LangChain Graph的概念、工作原理、应用场景以及实践技巧，帮助您全面理解和应用这一强大工具。 知识图谱与LangChain Graph基础 什么是知识图谱？ 知识图谱(Knowledge Graph)是一种结构化数据模型，用于表示实体(Entities)之间的关系(Relations)。它以图的形式组织信息，其中： 节点(Nodes)：代表实体或概念 边(Edges)：代表实体间的关系 graph LR A[艾伦·图灵] --&gt;|发明| B[图灵机] A --&gt;|出生于| C[英国] A --&gt;|被誉为| D[计算机科学之父] B --&gt;|是| E[理论计算模型] LangChain Graph的定义与价值 LangChain Graph是LangChain框架中专注于知识图谱构建、存储和查询的模块集合。它将LLM的自然语言处理能力与图数据库的结构化表示结合，实现了： 自动从文本中提取实体和关系 构建和维护知识图谱 基于图结构进行复杂查询和推理 增强LLM应用的上下文理解和回答质量 LangChain Graph架构 LangChain Graph的整体架构可以通过以下图示来理解： flowchart TB subgraph \"输入层\" A[文本文档] --&gt; B[网页内容] C[结构化数据] --&gt; D[用户查询] end subgraph...","categories": ["AI技术"],
        "tags": ["LangChain","知识图谱","Graph","大语言模型","LLM"],
        "url": "/ai%E6%8A%80%E6%9C%AF/langchain-graph-knowledge-construction/",
        "teaser": "/assets/images/langchain-graph-teaser.jpg"
      },{
        "title": "LangChain 与 LLM 的结合使用详解",
        "excerpt":"LangChain 是一个强大的框架，专为开发基于大语言模型(LLM)的应用而设计。本文将详细介绍 LangChain 与 LLM 的结合方式、核心组件以及常见应用场景。 LangChain 核心理念 LangChain 的核心理念是将 LLM 与外部资源(如数据源、工具、API等)连接起来，构建更强大、更实用的 AI 应用。它提供了一系列抽象和工具，使开发者能够轻松地: 与各种 LLM 服务进行标准化交互 构建复杂的处理流程 使 LLM 能够访问外部信息和工具 实现记忆和状态管理 LangChain 架构概览 flowchart TB subgraph \"应用层\" A1[智能问答系统] A2[对话机器人] A3[文档分析工具] A4[代码助手] end subgraph \"LangChain 核心层\" B1[链 Chains] B2[代理 Agents] B3[记忆 Memory] B4[工具 Tools] end subgraph \"模型层\" C1[OpenAI GPT] C2[Anthropic...","categories": ["AI技术"],
        "tags": ["LangChain","LLM","大语言模型","AI应用开发","提示工程"],
        "url": "/ai%E6%8A%80%E6%9C%AF/langchain-llm-integration-guide/",
        "teaser": "/assets/images/langchain-llm-teaser.jpg"
      }]
