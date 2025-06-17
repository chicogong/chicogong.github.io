var store = [{
        "title": "欢迎来到我的博客",
        "excerpt":"欢迎来到我的个人博客   你好！欢迎来到我的个人博客。这是我在这个平台上发布的第一篇文章，我想借此机会介绍一下自己，以及创建这个博客的初衷。   关于我   我是 Chico Gong，一个热爱技术和分享的开发者。在技术的世界里，我专注于前端开发、后端架构，以及不断涌现的新兴技术。除了编程，我也对设计、产品思维和创业有着浓厚的兴趣。   我相信知识的力量，更相信分享知识的价值。在学习和工作的过程中，我深深体会到开源社区和技术博客对个人成长的巨大帮助。现在，我希望通过这个博客，将自己的学习心得、技术探索和生活感悟分享给更多的人。   为什么创建这个博客   创建这个博客有几个原因：      记录学习历程：技术发展日新月异，通过写博客可以帮助我整理思路，加深对知识的理解。   分享经验教训：在开发过程中遇到的问题和解决方案，希望能够帮助到其他遇到类似问题的朋友。   建立连接：希望通过博客认识更多志同道合的朋友，一起交流学习，共同成长。   提升表达能力：写作是一种很好的思维训练，可以帮助我更好地表达想法和观点。   博客内容规划   在这个博客里，我计划分享以下几类内容：   技术文章   包括但不限于：     前端开发技术（React, Vue, JavaScript, CSS等）   后端开发经验（Node.js, Python, 数据库设计等）   工具和效率（开发工具使用、工作流优化等）   新技术探索（AI/ML、Web3、移动开发等）   项目分享   我会分享一些自己做过的有趣项目，包括项目的创意来源、技术选型、开发过程中遇到的挑战和解决方案。   学习笔记   读书笔记、课程学习心得、会议参会感受等，希望通过分享促进更深入的思考。   生活感悟   技术之外的思考，关于工作、生活、成长的一些想法和体会。   写在最后   这个博客刚刚起步，内容和设计都还在不断完善中。如果你有任何建议或想法，非常欢迎与我交流。你可以通过邮件、GitHub 或其他社交媒体联系我。   感谢你花时间阅读这篇文章，希望在接下来的日子里，我们能在这个小小的数字空间里，一起学习、成长和探索。   期待与你的相遇！     2024年1月1日  ","categories": ["博客","介绍"],
        "tags": ["博客","介绍","开始"],
        "url": "/%E5%8D%9A%E5%AE%A2/%E4%BB%8B%E7%BB%8D/welcome-to-my-blog/",
        "teaser": "/assets/images/tech-teaser.jpg"
      },{
        "title": "AI语音通话系统开发实战：从零构建智能语音交互平台",
        "excerpt":"前言 随着ChatGPT、Claude等大语言模型的兴起，AI语音交互已成为下一代人机交互的重要方向。本文将从零开始，带你构建一个完整的AI语音通话系统，实现人与AI的自然语音对话。 什么是AI语音通话系统 AI语音通话系统是一个集成了多种先进技术的智能交互平台，主要包括： 实时语音通信：基于WebRTC的低延迟音频传输 语音识别(ASR)：将语音转换为文本 自然语言理解(NLU)：理解用户意图和语义 对话管理：维护对话上下文和状态 语音合成(TTS)：将AI回复转换为自然语音 系统架构设计 整体架构 graph TB A[用户] --&gt; B[WebRTC客户端] B --&gt; C[信令服务器] C --&gt; D[媒体服务器] D --&gt; E[语音识别服务] E --&gt; F[AI对话引擎] F --&gt; G[语音合成服务] G --&gt; D D --&gt; B B --&gt; A 核心组件 前端WebRTC客户端 音频采集和播放 实时音频传输 用户界面交互 后端服务集群 信令服务器（WebSocket/Socket.io） 媒体处理服务器 AI对话引擎 语音处理服务...","categories": ["AI技术","语音通话"],
        "tags": ["WebRTC","语音识别","AI","实时通信"],
        "url": "/ai%E6%8A%80%E6%9C%AF/%E8%AF%AD%E9%9F%B3%E9%80%9A%E8%AF%9D/github-pages-setup-guide/",
        "teaser": "/assets/images/ai-voice-call-teaser.jpg"
      },{
        "title": "Python网页爬虫进阶教程：从基础到反爬虫对抗",
        "excerpt":"网页爬虫是数据科学和自动化领域的重要技能。本文将从基础概念开始，逐步深入到高级技巧和反爬虫对抗策略，帮助你构建稳定高效的爬虫系统。 爬虫基础概念 网页爬虫（Web Crawler）是一种自动获取网页内容的程序，主要用于： 数据收集：获取网站的公开信息 价格监控：跟踪商品价格变化 新闻聚合：收集多个新闻源的文章 SEO分析：分析网站结构和内容 基础爬虫实现 1. 使用requests和BeautifulSoup import requests from bs4 import BeautifulSoup import time import csv from urllib.parse import urljoin, urlparse import logging # 配置日志 logging.basicConfig(level=logging.INFO) logger = logging.getLogger(__name__) class BasicCrawler: def __init__(self, base_url, headers=None): self.base_url = base_url self.session = requests.Session() self.session.headers.update(headers or { 'User-Agent':...","categories": ["Python","数据采集"],
        "tags": ["Python","爬虫","Scrapy","反爬虫","数据挖掘"],
        "url": "/python/%E6%95%B0%E6%8D%AE%E9%87%87%E9%9B%86/python-web-scraping-advanced/",
        "teaser": "/assets/images/tech-teaser.jpg"
      },{
        "title": "Node.js微服务架构实战：从单体到分布式的演进之路",
        "excerpt":"随着业务的快速增长，传统的单体应用架构逐渐暴露出扩展性和维护性的问题。今天我将分享如何使用Node.js构建高效的微服务架构，以及在实际项目中的最佳实践。 什么是微服务架构？ 微服务架构是一种将单个应用程序拆分为一组小型、独立服务的架构模式。每个服务： 独立部署：可以独立发布和更新 单一职责：专注于特定的业务功能 轻量级通信：通过HTTP API或消息队列通信 技术多样性：可以使用不同的技术栈 架构设计图 ┌─────────────────┐ ┌─────────────────┐ │ Web Client │ │ Mobile Client │ └─────────┬───────┘ └─────────┬───────┘ │ │ └──────┬─────────────────┘ │ ┌───────▼────────┐ │ API Gateway │ └───────┬────────┘ │ ┌────────────┼────────────┐ │ │ │ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐ │ User │ │ Order │ │Payment │ │Service │ │Service...","categories": ["后端开发","架构设计"],
        "tags": ["Node.js","微服务","Docker","API Gateway","分布式系统"],
        "url": "/%E5%90%8E%E7%AB%AF%E5%BC%80%E5%8F%91/%E6%9E%B6%E6%9E%84%E8%AE%BE%E8%AE%A1/nodejs-microservices-architecture/",
        "teaser": "/assets/images/tech-teaser.jpg"
      },{
        "title": "React Hooks 最佳实践：从入门到精通",
        "excerpt":"React Hooks自2018年推出以来，已经彻底改变了我们编写React组件的方式。今天我将分享一些在实际项目中总结的Hooks最佳实践。 为什么使用Hooks？ Hooks让我们能够在函数组件中使用状态和其他React特性，相比类组件有以下优势： 更简洁的代码：避免了class的复杂语法 更好的逻辑复用：通过自定义Hooks 更容易测试：函数组件更容易进行单元测试 更好的性能优化：结合useMemo和useCallback 核心Hooks使用技巧 1. useState的优化使用 // ❌ 避免：频繁的状态更新 const [count, setCount] = useState(0); const [loading, setLoading] = useState(false); const [error, setError] = useState(null); // ✅ 推荐：使用useReducer管理复杂状态 const initialState = { count: 0, loading: false, error: null }; function reducer(state, action) { switch (action.type) {...","categories": ["前端开发","React"],
        "tags": ["React","Hooks","JavaScript","最佳实践"],
        "url": "/%E5%89%8D%E7%AB%AF%E5%BC%80%E5%8F%91/react/react-hooks-best-practices/",
        "teaser": "/assets/images/tech-teaser.jpg"
      },{
        "title": "WebRTC实时语音通信开发完整指南：构建高质量语音通话应用",
        "excerpt":"前言 WebRTC（Web Real-Time Communication）是一项革命性的技术，它使得浏览器之间可以直接进行实时音视频通信，无需安装任何插件。本文将带你从零开始构建一个完整的WebRTC语音通话应用。 WebRTC基础概念 什么是WebRTC WebRTC是一个开源项目，提供了浏览器和移动应用程序间实时通信的能力。它包含三个主要的API： MediaStream API：访问摄像头和麦克风 RTCPeerConnection API：建立P2P连接 RTCDataChannel API：传输任意数据 WebRTC通信流程 sequenceDiagram participant A as 用户A participant S as 信令服务器 participant B as 用户B A-&gt;&gt;S: 创建房间 B-&gt;&gt;S: 加入房间 A-&gt;&gt;S: 发送Offer S-&gt;&gt;B: 转发Offer B-&gt;&gt;S: 发送Answer S-&gt;&gt;A: 转发Answer A-&gt;&gt;B: ICE候选交换 B-&gt;&gt;A: ICE候选交换 A-&gt;&gt;B: 直接P2P通信 技术架构设计 整体架构 // 系统架构组件 const...","categories": ["WebRTC","实时通信"],
        "tags": ["WebRTC","语音通话","P2P","实时通信","JavaScript"],
        "url": "/webrtc/%E5%AE%9E%E6%97%B6%E9%80%9A%E4%BF%A1/ai-chatbot-development-guide/",
        "teaser": "/assets/images/webrtc-teaser.jpg"
      },{
        "title": "云原生Kubernetes实战指南：从入门到生产环境部署",
        "excerpt":"Kubernetes已经成为现代云原生应用部署的标准平台。本文将从基础概念开始，逐步深入到生产环境的最佳实践，帮助你全面掌握Kubernetes的核心技能。 什么是Kubernetes？ Kubernetes（简称K8s）是一个开源的容器编排系统，用于自动化应用程序的部署、扩展和管理。它提供了： 容器编排：管理大量容器的生命周期 服务发现：自动发现和连接服务 负载均衡：分发流量到健康的实例 自动扩缩容：根据负载自动调整实例数量 滚动更新：零停机时间的应用更新 核心概念详解 1. 基础对象 # pod.yaml - Pod是K8s中最小的部署单元 apiVersion: v1 kind: Pod metadata: name: my-app-pod labels: app: my-app version: v1.0 spec: containers: - name: app-container image: nginx:1.20 ports: - containerPort: 80 resources: requests: memory: \"64Mi\" cpu: \"250m\" limits: memory: \"128Mi\" cpu: \"500m\" env:...","categories": ["云原生","DevOps"],
        "tags": ["Kubernetes","Docker","容器编排","微服务","DevOps"],
        "url": "/%E4%BA%91%E5%8E%9F%E7%94%9F/devops/cloud-native-kubernetes-guide/",
        "teaser": "/assets/images/tech-teaser.jpg"
      }]
