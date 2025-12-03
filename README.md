# AI技术博客

基于Hugo构建的AI技术深度分享平台，专注于前沿AI技术的深度解析与实战经验分享。

## 📚 内容分类

### Agent系统技术
- AI Agent架构设计与实现
- 多Agent协作机制
- RAG（检索增强生成）实战
- LangChain框架应用
- Agent记忆系统设计

### 语音技术前沿
- ASR（语音识别）最新技术
- TTS（语音合成）系统实现
- Voice Agent架构设计
- Qwen-Audio多模态语音技术

### 大语言模型
- Qwen3-235B技术架构
- MoE（混合专家）模型详解
- 长上下文处理技术

### 开发指南
- LLM微调完全指南
- 模型部署与优化
- 工程最佳实践

## 🚀 快速开始

### 本地运行
```bash
# 安装Hugo
brew install hugo

# 克隆仓库
git clone https://github.com/chicogong/chicogong.github.io.git
cd chicogong.github.io

# 克隆 PaperMod 主题（本地开发需要）
git clone --depth=1 https://github.com/adityatelange/hugo-PaperMod themes/PaperMod

# 启动本地服务器
hugo server -D

# 访问 http://localhost:1313
```

### 发布文章
```bash
# 创建新文章
hugo new posts/category/article-name.md

# 编辑文章内容
# ...

# 提交更改
git add .
git commit -m "feat: add new article"
git push origin main
```

## 📝 文章格式

文章使用Markdown格式，支持以下Front Matter配置：

```yaml
---
title: "文章标题"
date: 2024-12-28T10:00:00+08:00
categories: [分类1, 分类2]
tags: [标签1, 标签2, 标签3]
excerpt: "文章摘要"
toc: true
---
```

## 🛠 技术栈

- **静态网站生成器**: Hugo
- **主题**: PaperMod
- **部署**: GitHub Pages
- **域名**: https://chicogong.github.io

## 📄 License

MIT License

## 👤 作者

Haoran Gong

---

*持续更新中，欢迎关注最新AI技术动态*