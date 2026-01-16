---
title: "AI Agent架构：想清楚再动手"
date: 2026-01-08T10:00:00+08:00
draft: false
tags: ["AI Agent", "架构设计", "LLM"]
categories: ["AI Agent"]
excerpt: "Agent不是简单的Prompt调用。想做好一个Agent，得想清楚几件事：怎么思考、怎么记忆、怎么用工具。这篇讲讲Agent的核心架构。"
---

## Agent的核心循环

一个Agent本质上在做这件事：

```
感知 → 思考 → 行动 → 反馈 → 继续思考...
```

用代码表示：

```python
while not done:
    # 1. 理解用户要什么
    intent = understand(user_input)
    
    # 2. 想想怎么做
    plan = think(intent, memory)
    
    # 3. 动手执行
    result = act(plan, tools)
    
    # 4. 看看结果对不对
    if verify(result):
        done = True
    else:
        memory.add(result)  # 记住失败，下次改进
```

---

## 三个关键模块

### 1. 记忆系统

Agent和普通LLM调用的区别：**Agent会记东西**。

```python
class Memory:
    short_term = []  # 当前对话历史
    long_term = {}   # 跨对话的知识
    
    def remember(self, key, value):
        self.long_term[key] = value
    
    def recall(self, query):
        return search(self.long_term, query)
```

**实际应用：**
- 记住用户的偏好
- 记住之前失败的尝试
- 记住成功的模式

### 2. 工具调用

Agent靠工具干活，不是靠瞎编。

```python
tools = {
    "search": lambda q: google_search(q),
    "calculate": lambda expr: eval(expr),
    "send_email": lambda to, content: send_email(to, content),
}

def use_tool(name, args):
    return tools[name](**args)
```

**关键点：**
- 工具描述要写清楚，LLM才知道什么时候用
- 工具要有错误处理
- 危险操作要二次确认

### 3. 任务规划

复杂任务要分解。

```python
def plan(task):
    if is_simple(task):
        return [task]
    else:
        return decompose(task)  # 拆成子任务

# 例如："写一篇技术博客"
# 拆成：
# 1. 确定主题
# 2. 列大纲
# 3. 写每一节
# 4. 润色
# 5. 发布
```

---

## ReAct模式

最常用的Agent思考模式：**边想边做**。

```
用户：北京明天天气怎么样？

Agent思考：需要查天气，我有天气工具
Agent行动：调用天气API
Agent观察：返回"晴，15-25度"
Agent思考：拿到结果了，可以回复
Agent输出：北京明天晴天，气温15-25度，适合出门。
```

代码实现：

```python
def react(query):
    thoughts = []
    for _ in range(max_steps):
        thought = llm.think(query, thoughts)
        thoughts.append(thought)
        
        if thought.type == "action":
            result = execute(thought.action)
            thoughts.append(f"观察: {result}")
        
        elif thought.type == "answer":
            return thought.content
    
    return "想不出来..."
```

---

## 常见坑

### 坑1：无限循环

Agent卡住了，一直在做同样的事。

**解决：** 设置最大步数，加入"放弃"逻辑

### 坑2：工具乱用

LLM选错了工具。

**解决：** 工具描述写清楚，提供使用示例

### 坑3：幻觉

Agent编造不存在的信息。

**解决：** 强制要求查证，不确定时说"不知道"

### 坑4：上下文超长

对话太长，超出token限制。

**解决：** 压缩历史记忆，只保留关键信息

---

## 实战建议

1. **从简单开始**。先做一个只有1个工具的Agent，跑通再加功能。

2. **日志要详细**。Agent做了什么、为什么做，都要记下来方便调试。

3. **人在环路**。关键操作需要人工确认，别让Agent自作主张。

4. **持续迭代**。根据实际使用反馈不断优化。

---

## 框架推荐

| 场景 | 推荐 |
|------|------|
| 快速原型 | LangChain |
| 生产级 | LangGraph |
| 轻量级 | 自己写（就几百行） |

---

有问题留言，下篇讲多Agent协作。