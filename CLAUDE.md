# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an AI technology blog built with Hugo and hosted on GitHub Pages at https://realtime-ai.chat/. The blog focuses on deep technical content about AI Agent systems, voice technology, LLMs, and development guides.

## Common Commands

### Local Development
```bash
# Clone PaperMod theme (required for local development)
git clone --depth=1 https://github.com/adityatelange/hugo-PaperMod themes/PaperMod

# Start local development server
hugo server -D

# Build the site
hugo --minify

# Create new article
hugo new posts/category/article-name.md
```

### Git Workflow
```bash
# Commit changes (auto-deploys via GitHub Actions)
git add .
git commit -m "feat: add new article"
git push origin main
```

## Architecture

### Technology Stack
- **Static Site Generator**: Hugo v0.149.1 (extended version)
- **Theme**: PaperMod (fetched during CI/CD)
- **Hosting**: GitHub Pages with custom domain
- **CI/CD**: GitHub Actions workflow at `.github/workflows/deploy-hugo.yml`

### Content Structure
```
content/
├── posts/
│   ├── agent-systems/      # AI Agent architecture, RAG, LangChain
│   ├── development-guides/  # LLM fine-tuning, deployment guides
│   ├── llm-models/         # Model architectures, Qwen series
│   └── voice-technology/   # ASR, TTS, Voice Agent systems
├── about/
└── search.md
```

### Article Format
Articles use Markdown with Hugo front matter:
```yaml
---
title: "Article Title"
date: 2024-12-28T10:00:00+08:00
categories: [Category1, Category2]
tags: [tag1, tag2, tag3]
excerpt: "Article summary"
toc: true
---
```

### Key Configuration
- Main config: `hugo.toml` - site settings, theme parameters, menu structure
- Base URL: https://realtime-ai.chat/
- Language: Chinese (zh-cn)
- Theme settings: Light theme default, code highlighting enabled
- Deployment: Automatic via GitHub Actions on push to main branch