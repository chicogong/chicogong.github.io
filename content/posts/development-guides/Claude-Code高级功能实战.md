---
title: "Claude Code 高级功能实战：MCP、Hooks、SubAgent 与自定义命令"
date: 2025-12-26T10:00:00+08:00
draft: false
weight: 1
tags: ["Claude Code", "MCP", "Hooks", "SubAgent", "AI编程", "高级配置"]
categories: ["开发工具", "AI"]
summary: "深入探索 Claude Code 的高级功能：MCP 协议扩展外部工具、Hooks 自动化工作流、SubAgent 多智能体并发、CLAUDE.md 项目规范配置。从原理到实战，让你真正掌握这个强大的 AI 编程工具。"
---

## 前言：不只是聊天机器人

大多数人使用 Claude Code 只是简单地"对话写代码"。但 Claude Code 的真正威力在于它的**可扩展性**和**自动化能力**。

本文将深入介绍 Claude Code 的四大高级功能：

1. **MCP（Model Context Protocol）**：让 Claude 连接外部工具和数据源
2. **Hooks**：在关键节点插入自动化脚本
3. **SubAgent 多智能体架构**：并发执行复杂任务
4. **CLAUDE.md 配置系统**：定义项目规范和工作流

这些功能组合起来，能让 Claude Code 从一个"AI助手"进化成"AI工程师团队"。

---

## 一、MCP：让 Claude 连接一切

### 1.1 什么是 MCP？

MCP（Model Context Protocol）是 Anthropic 推出的开放协议，让 AI 模型能够与外部工具和数据源进行标准化交互。

```
传统方式：Claude 只能看到你发给它的文本
MCP方式：Claude 可以主动调用工具获取信息

┌─────────────┐     MCP协议      ┌─────────────┐
│  Claude Code │ ◄────────────► │  外部服务    │
└─────────────┘                 └─────────────┘
                                    ├── 文件系统
                                    ├── 数据库
                                    ├── GitHub
                                    ├── Slack
                                    ├── Google Drive
                                    └── 自定义工具
```

### 1.2 配置 MCP 服务器

**方法一：命令行添加（推荐）**

```bash
# 添加文件系统访问
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem ~/Documents ~/Projects

# 添加 GitHub 集成
claude mcp add github -e GITHUB_TOKEN=your_token -- npx -y @modelcontextprotocol/server-github

# 添加内存服务（持久化记忆）
claude mcp add memory -- npx -y @modelcontextprotocol/server-memory

# 查看已配置的服务器
claude mcp list
```

**方法二：JSON 配置文件**

编辑 `~/.claude/settings.json`：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Projects"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxx"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/mydb"
      }
    }
  }
}
```

### 1.3 常用 MCP 服务器

| 服务器 | 用途 | 安装命令 |
|:---|:---|:---|
| filesystem | 读写本地文件 | `@modelcontextprotocol/server-filesystem` |
| github | GitHub 操作 | `@modelcontextprotocol/server-github` |
| postgres | 数据库查询 | `@modelcontextprotocol/server-postgres` |
| memory | 持久化记忆 | `@modelcontextprotocol/server-memory` |
| slack | Slack 消息 | `@modelcontextprotocol/server-slack` |
| puppeteer | 网页自动化 | `@modelcontextprotocol/server-puppeteer` |

### 1.4 实战：连接数据库查询

配置 PostgreSQL MCP 后，你可以这样使用：

```
You: 帮我查一下最近7天注册的用户数量，按天分组

Claude: 我来查询数据库...
[调用 postgres MCP 服务器]

查询结果：
| 日期 | 新增用户数 |
|------|-----------|
| 12-20 | 156 |
| 12-21 | 189 |
| 12-22 | 203 |
| 12-23 | 178 |
| 12-24 | 245 |
| 12-25 | 312 |
| 12-26 | 198 |

总计：1,481 位新用户
可以看出 12-25（圣诞节）有明显的注册高峰。
```

---

## 二、Hooks：自动化工作流的核心

### 2.1 什么是 Hooks？

Hooks 是 Claude Code 生命周期中的"钩子"，让你在特定事件发生时自动执行脚本。

```
用户输入 → [UserPromptSubmit Hook] → Claude 思考
                                          ↓
                              [PreToolUse Hook] → 工具执行
                                                      ↓
                                          [PostToolUse Hook] → 返回结果
                                                                   ↓
                                                          [Stop Hook] → 会话结束
```

### 2.2 官方 8 大事件

| 事件 | 触发时机 | 可用变量 | 能否阻断 |
|:---|:---|:---|:---|
| `SessionStart` | 会话新建/resume | 无 | ❌ |
| `UserPromptSubmit` | 用户按回车前 | `prompt` | ✅ |
| `PreToolUse` | 工具准备执行前 | `tool_name`, `tool_input` | ✅ |
| `PostToolUse` | 工具执行结束后 | `tool_name`, `tool_input`, `tool_output` | ❌ |
| `Notification` | Claude 需要用户输入 | `notification_text` | ❌ |
| `Stop` | 回答整体结束 | 无 | ❌ |
| `SubagentStop` | 子代理任务结束 | `subagent_name`, `result` | ❌ |
| `PreCompact` | 压缩对话缓存前 | 无 | ❌ |

### 2.3 配置 Hooks

**全局配置** `~/.claude/settings.json`：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "echo '[AUDIT] Bash command: $TOOL_INPUT' >> ~/.claude/audit.log"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command", 
            "command": "npx prettier --write $FILE_PATH 2>/dev/null || true"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/validate-prompt.sh"
          }
        ]
      }
    ]
  }
}
```

**项目级配置** `.claude/settings.local.json`：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "npm run lint:fix -- $FILE_PATH"
          }
        ]
      }
    ]
  }
}
```

### 2.4 实战示例

**示例1：自动格式化代码**

每次 Claude 写入文件后，自动运行 Prettier：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "npx prettier --write \"$TOOL_INPUT\" 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

**示例2：危险命令拦截**

阻止执行危险的 Bash 命令：

```bash
#!/bin/bash
# ~/.claude/scripts/validate-bash.sh

DANGEROUS_PATTERNS="rm -rf|DROP TABLE|DELETE FROM|format|mkfs"

if echo "$TOOL_INPUT" | grep -qE "$DANGEROUS_PATTERNS"; then
    echo "BLOCKED: 检测到危险命令"
    exit 1
fi

exit 0
```

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/scripts/validate-bash.sh"
          }
        ]
      }
    ]
  }
}
```

**示例3：Slack 通知**

任务完成后发送 Slack 通知：

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "curl -X POST -H 'Content-type: application/json' --data '{\"text\":\"Claude Code 任务完成!\"}' $SLACK_WEBHOOK_URL"
          }
        ]
      }
    ]
  }
}
```

---

## 三、SubAgent：多智能体并发架构

### 3.1 架构原理

Claude Code 采用分层多 Agent 架构，通过**主 Agent**和**SubAgent**协作处理复杂任务：

```
用户请求
    ↓
主 Agent (nO 函数)
    ↓
是否调用 Task 工具？
    ├── 否 → 直接处理 → 返回结果
    └── 是 → 创建 SubAgent
              ↓
         并发执行调度器 (UH1)
              ↓
         多个 SubAgent 并行执行
              ↓
         结果合成器 (KN5)
              ↓
         返回合成结果
```

### 3.2 核心特性

1. **完全隔离的执行环境**：每个 SubAgent 在独立上下文中运行
2. **智能并发调度**：支持最多 10 个 Agent 并发执行
3. **安全权限控制**：SubAgent 无法调用 Task 工具（防止递归）
4. **高效结果合成**：智能合并多个 Agent 的输出

### 3.3 Task 工具使用

当你给 Claude Code 一个复杂任务时，它会自动决定是否使用 SubAgent：

```
You: 分析这个项目的代码质量，包括：
1. 代码结构是否合理
2. 是否有潜在的性能问题
3. 测试覆盖率如何
4. 安全漏洞检查

Claude: 这是一个复杂的分析任务，我会启动多个 SubAgent 并行处理：

[启动 SubAgent 1] 分析代码结构...
[启动 SubAgent 2] 检查性能问题...
[启动 SubAgent 3] 分析测试覆盖率...
[启动 SubAgent 4] 扫描安全漏洞...

[等待所有 SubAgent 完成...]

[合成结果] 综合所有 Agent 的分析：
...
```

### 3.4 SubAgent 可用工具

SubAgent 可以使用以下工具：

```javascript
const SUBAGENT_ALLOWED_TOOLS = [
    'Read',       // 读取文件
    'Write',      // 写入文件
    'Edit',       // 编辑文件
    'MultiEdit',  // 多处编辑
    'Glob',       // 文件搜索
    'Grep',       // 内容搜索
    'Bash',       // 执行命令
    'WebFetch',   // 网页获取
    'WebSearch',  // 网络搜索
    'TodoRead',   // 读取任务
    'TodoWrite',  // 写入任务
];

// 被禁用的工具（防止递归）
const BLOCKED_TOOLS = ['Task'];
```

### 3.5 并发执行示例

```
You: 帮我同时搜索项目中所有的 TODO 注释、FIXME 标记和废弃的 API 调用

Claude: 我会并行启动 3 个搜索任务：

[SubAgent 1] 搜索 TODO 注释...
[SubAgent 2] 搜索 FIXME 标记...
[SubAgent 3] 搜索废弃 API 调用...

--- 3 秒后 ---

[SubAgent 1 完成] 找到 23 个 TODO
[SubAgent 2 完成] 找到 8 个 FIXME
[SubAgent 3 完成] 找到 5 个废弃 API

[合成结果]
## 代码待办事项汇总

### TODO (23处)
- src/api/user.ts:45 - TODO: 添加缓存
- src/components/Table.tsx:123 - TODO: 优化渲染性能
...

### FIXME (8处)
- src/utils/date.ts:67 - FIXME: 时区处理有问题
...

### 废弃 API (5处)
- src/legacy/auth.ts:12 - 使用了已废弃的 crypto.createCipher
...
```

---

## 四、CLAUDE.md 配置系统

### 4.1 三级配置文件

Claude Code 使用三类 `.md` 文件进行配置，优先级从高到低：

| 文件 | 位置 | 作用范围 | 版本控制 |
|:---|:---|:---|:---|
| `CLAUDE.local.md` | 项目根目录 | 当前用户 | ❌ 不提交 |
| `CLAUDE.md` | 项目根目录 | 当前项目 | ✅ 提交 |
| `~/.claude/CLAUDE.md` | 用户目录 | 所有项目 | ❌ 不提交 |

### 4.2 创建项目配置

使用 `/init` 命令自动生成：

```bash
claude
> /init
```

这会在项目根目录创建 `CLAUDE.md` 文件。

### 4.3 配置示例

**项目级 CLAUDE.md**（提交到 Git）：

```markdown
# 项目配置

## 项目概述
这是一个 Next.js 14 电商平台，使用 TypeScript + Prisma + PostgreSQL。

## 技术栈
- 框架：Next.js 14 (App Router)
- 语言：TypeScript 5.x (严格模式)
- 数据库：PostgreSQL + Prisma ORM
- 样式：Tailwind CSS
- 状态管理：Zustand
- 测试：Vitest + Testing Library

## 代码规范
- 使用 ESLint + Prettier 格式化
- 组件使用函数式写法
- 所有函数必须有 JSDoc 注释
- 禁止使用 any 类型
- 错误处理使用自定义 Error 类

## 目录结构
```
src/
├── app/           # Next.js App Router 页面
├── components/    # React 组件
├── lib/           # 工具函数和配置
├── hooks/         # 自定义 Hooks
├── types/         # TypeScript 类型定义
└── prisma/        # 数据库 Schema
```

## 常用命令
- `npm run dev` - 启动开发服务器
- `npm run build` - 构建生产版本
- `npm run test` - 运行测试
- `npm run lint` - 代码检查
- `npx prisma studio` - 打开数据库管理界面

## Git 提交规范
- feat: 新功能
- fix: 修复 Bug
- refactor: 重构
- docs: 文档
- test: 测试
- chore: 构建/工具

## 注意事项
- 数据库敏感操作需要人工确认
- 不要直接修改 prisma/migrations 目录
- API 路由必须有权限验证
```

**个人本地配置 CLAUDE.local.md**（不提交）：

```markdown
# 本地配置

## 个人偏好
- 我喜欢详细的注释
- 代码示例要完整可运行
- 解释时使用中文

## 本地环境
- Node.js: v20.10.0
- 数据库：本地 Docker 容器
- 端口：3000 (开发), 5432 (PostgreSQL)

## 调试配置
- 使用 VS Code 调试
- 断点调试端口: 9229
```

**全局配置 ~/.claude/CLAUDE.md**：

```markdown
# 全局配置

## 通用规范
- 使用 2 空格缩进
- 变量命名使用 camelCase
- 常量使用 UPPER_SNAKE_CASE
- 组件使用 PascalCase

## 默认工具
- 包管理器：pnpm
- 版本控制：Git
- 编辑器：VS Code

## 安全规则
- 不要在代码中硬编码密钥
- 敏感文件（.env）不要提交
- API 调用必须有超时设置
```

### 4.4 自定义斜杠命令

在 `.claude/commands/` 目录下创建 Markdown 文件，即可定义自定义命令：

**文件结构**：
```
.claude/
└── commands/
    ├── review.md      # /review 命令
    ├── test.md        # /test 命令
    └── deploy.md      # /deploy 命令
```

**示例：review.md**

```markdown
请对当前项目进行代码审查：

1. 检查代码质量
   - 是否有重复代码
   - 函数是否过长
   - 命名是否清晰

2. 检查潜在问题
   - 是否有未处理的错误
   - 是否有性能问题
   - 是否有安全漏洞

3. 检查最佳实践
   - 是否遵循项目规范
   - 是否有充分的测试
   - 是否有必要的注释

请提供详细的改进建议。
```

使用：
```
> /review
```

---

## 五、综合实战：打造个人 AI 工作流

### 5.1 完整配置示例

结合以上所有功能，配置一个完整的 AI 开发工作流：

**~/.claude/settings.json**：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "~/Projects"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "npx prettier --write \"$FILE_PATH\" 2>/dev/null || true"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "echo \"[$(date)] Task completed\" >> ~/.claude/activity.log"
          }
        ]
      }
    ]
  },
  "permissions": {
    "allow": [
      "Bash(npm run *)",
      "Bash(git *)",
      "Bash(npx prettier *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(sudo *)"
    ]
  }
}
```

### 5.2 工作流示例

```
1. 启动项目
   > claude
   > /init  # 生成 CLAUDE.md

2. 查看项目状态
   > /review  # 运行自定义代码审查命令

3. 开发新功能
   > 帮我实现用户头像上传功能，要支持裁剪和压缩
   
   [Claude 自动]:
   - 分析项目结构
   - 创建组件文件（自动格式化）
   - 编写测试用例
   - 更新文档

4. 提交代码
   > 提交这些改动，写一个清晰的 commit message
   
   [Claude 使用 GitHub MCP]:
   - 生成 commit message
   - 创建 commit
   - 可选：创建 PR
```

---

## 总结

Claude Code 的高级功能让它从一个简单的 AI 聊天工具，变成了一个真正的**AI 开发平台**：

| 功能 | 作用 | 收益 |
|:---|:---|:---|
| **MCP** | 连接外部工具和数据 | 扩展能力边界 |
| **Hooks** | 自动化工作流 | 减少重复操作 |
| **SubAgent** | 并发处理复杂任务 | 提升效率 |
| **CLAUDE.md** | 定义项目规范 | 保持一致性 |

掌握这些功能，你就能把 Claude Code 打造成专属的 AI 工程师团队。

---

## 参考资源

- [Claude Code 官方文档](https://docs.anthropic.com/claude-code)
- [MCP 协议规范](https://modelcontextprotocol.io)
- [MCP 服务器列表](https://github.com/modelcontextprotocol/servers)
