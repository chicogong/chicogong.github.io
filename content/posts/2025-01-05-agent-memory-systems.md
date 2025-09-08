---
title: "Agent记忆系统设计：构建有记忆的智能体"
date: 2025-01-05T10:00:00+08:00
draft: false
tags: ["记忆系统", "向量数据库", "认知架构", "长期记忆"]
categories: ["AI Agent", "认知系统"]
---

## 引言

记忆是智能的基石。一个没有记忆的Agent就像得了健忘症的助手，无法积累经验、学习模式或维持上下文。本文深入探讨如何为AI Agent构建高效的记忆系统。

## 1. 记忆系统架构

### 1.1 记忆类型分层

```python
from enum import Enum
from typing import Any, List, Dict, Optional
import time

class MemoryType(Enum):
    SENSORY = "sensory"          # 感官记忆（<1秒）
    WORKING = "working"          # 工作记忆（秒级）
    SHORT_TERM = "short_term"    # 短期记忆（分钟级）
    LONG_TERM = "long_term"      # 长期记忆（永久）
    EPISODIC = "episodic"        # 情景记忆
    SEMANTIC = "semantic"        # 语义记忆
    PROCEDURAL = "procedural"    # 程序记忆

class MemorySystem:
    def __init__(self):
        self.sensory_buffer = SensoryMemory(capacity=10, ttl=1)
        self.working_memory = WorkingMemory(capacity=7)
        self.short_term_memory = ShortTermMemory(capacity=100, ttl=300)
        self.long_term_memory = LongTermMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()
        
    def process_input(self, input_data: Any):
        """处理输入信息的记忆流程"""
        # 1. 感官记忆
        self.sensory_buffer.store(input_data)
        
        # 2. 注意力机制筛选
        if self.is_important(input_data):
            # 3. 进入工作记忆
            self.working_memory.add(input_data)
            
            # 4. 编码到短期记忆
            encoded = self.encode_memory(input_data)
            self.short_term_memory.store(encoded)
            
            # 5. 巩固到长期记忆
            if self.should_consolidate(encoded):
                self.consolidate_to_long_term(encoded)
```

### 1.2 工作记忆实现

```python
class WorkingMemory:
    """Miller's Magic Number: 7±2"""
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.buffer = []
        self.attention_weights = []
        
    def add(self, item: Any):
        """添加项目到工作记忆"""
        if len(self.buffer) >= self.capacity:
            # 移除注意力权重最低的项
            min_idx = self.attention_weights.index(min(self.attention_weights))
            self.buffer.pop(min_idx)
            self.attention_weights.pop(min_idx)
        
        self.buffer.append(item)
        self.attention_weights.append(1.0)
    
    def update_attention(self, idx: int, weight_delta: float):
        """更新注意力权重"""
        if 0 <= idx < len(self.attention_weights):
            self.attention_weights[idx] += weight_delta
            # 归一化
            total = sum(self.attention_weights)
            self.attention_weights = [w/total for w in self.attention_weights]
    
    def get_context(self) -> List[Any]:
        """获取当前工作记忆上下文"""
        # 按注意力权重排序
        sorted_items = sorted(
            zip(self.buffer, self.attention_weights),
            key=lambda x: x[1],
            reverse=True
        )
        return [item for item, _ in sorted_items]
```

## 2. 长期记忆管理

### 2.1 记忆编码与存储

```python
import hashlib
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Memory:
    id: str
    content: Any
    embedding: np.ndarray
    timestamp: float
    access_count: int = 0
    importance: float = 0.5
    last_accessed: float = None
    metadata: Dict = None
    
class LongTermMemory:
    def __init__(self, vector_dim: int = 768):
        self.memories = {}
        self.embeddings = []
        self.index = None  # FAISS索引
        self.vector_dim = vector_dim
        self._init_index()
        
    def _init_index(self):
        """初始化向量索引"""
        import faiss
        self.index = faiss.IndexFlatL2(self.vector_dim)
        
    def store(self, content: Any, importance: float = 0.5):
        """存储记忆"""
        # 生成唯一ID
        memory_id = self._generate_id(content)
        
        # 生成嵌入向量
        embedding = self._encode_content(content)
        
        # 创建记忆对象
        memory = Memory(
            id=memory_id,
            content=content,
            embedding=embedding,
            timestamp=time.time(),
            importance=importance,
            metadata=self._extract_metadata(content)
        )
        
        # 存储
        self.memories[memory_id] = memory
        self.index.add(np.array([embedding]))
        
        return memory_id
    
    def retrieve(self, query: Any, k: int = 5) -> List[Memory]:
        """检索相关记忆"""
        query_embedding = self._encode_content(query)
        
        # 向量检索
        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )
        
        # 获取记忆对象
        retrieved = []
        for idx in indices[0]:
            if idx < len(self.memories):
                memory_id = list(self.memories.keys())[idx]
                memory = self.memories[memory_id]
                
                # 更新访问信息
                memory.access_count += 1
                memory.last_accessed = time.time()
                
                retrieved.append(memory)
        
        # 按相关性和重要性重排
        return self._rerank_memories(retrieved, query)
    
    def _rerank_memories(self, memories: List[Memory], query: Any) -> List[Memory]:
        """重排记忆（考虑时间衰减、重要性等）"""
        current_time = time.time()
        
        scored_memories = []
        for memory in memories:
            # 时间衰减因子
            time_decay = np.exp(-(current_time - memory.timestamp) / 86400)  # 日衰减
            
            # 访问频率因子
            access_factor = np.log1p(memory.access_count) / 10
            
            # 综合得分
            score = (
                0.4 * memory.importance +
                0.3 * time_decay +
                0.3 * access_factor
            )
            
            scored_memories.append((memory, score))
        
        # 按得分排序
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, _ in scored_memories]
```

### 2.2 记忆巩固机制

```python
class MemoryConsolidation:
    def __init__(self, consolidation_threshold: float = 0.7):
        self.threshold = consolidation_threshold
        self.consolidation_queue = []
        
    def evaluate_for_consolidation(self, memory: Memory) -> bool:
        """评估是否需要巩固到长期记忆"""
        # 重要性评分
        importance_score = memory.importance
        
        # 重复接触评分
        repetition_score = min(1.0, memory.access_count / 5)
        
        # 情感强度评分
        emotion_score = self._evaluate_emotional_intensity(memory.content)
        
        # 新颖性评分
        novelty_score = self._evaluate_novelty(memory.content)
        
        # 综合评分
        consolidation_score = (
            0.3 * importance_score +
            0.2 * repetition_score +
            0.3 * emotion_score +
            0.2 * novelty_score
        )
        
        return consolidation_score >= self.threshold
    
    def consolidate(self, short_term_memories: List[Memory]) -> List[Memory]:
        """巩固短期记忆到长期记忆"""
        consolidated = []
        
        for memory in short_term_memories:
            if self.evaluate_for_consolidation(memory):
                # 增强记忆编码
                enhanced_memory = self._enhance_memory(memory)
                
                # 创建关联连接
                self._create_associations(enhanced_memory, consolidated)
                
                consolidated.append(enhanced_memory)
        
        return consolidated
    
    def _enhance_memory(self, memory: Memory) -> Memory:
        """增强记忆编码（添加更多细节和关联）"""
        # 提取关键概念
        concepts = self._extract_concepts(memory.content)
        
        # 生成记忆摘要
        summary = self._generate_summary(memory.content)
        
        # 更新元数据
        memory.metadata.update({
            "concepts": concepts,
            "summary": summary,
            "consolidation_time": time.time()
        })
        
        return memory
```

## 3. 情景记忆系统

### 3.1 情景记忆结构

```python
@dataclass
class Episode:
    id: str
    start_time: float
    end_time: float
    events: List[Dict]
    context: Dict
    outcome: Any
    emotional_valence: float  # -1到1，负面到正面
    
class EpisodicMemory:
    def __init__(self):
        self.episodes = {}
        self.current_episode = None
        self.episode_index = {}  # 用于快速检索
        
    def start_episode(self, context: Dict):
        """开始新情景"""
        episode_id = f"ep_{int(time.time() * 1000)}"
        self.current_episode = Episode(
            id=episode_id,
            start_time=time.time(),
            end_time=None,
            events=[],
            context=context,
            outcome=None,
            emotional_valence=0
        )
        return episode_id
    
    def add_event(self, event: Dict):
        """向当前情景添加事件"""
        if self.current_episode:
            event["timestamp"] = time.time()
            self.current_episode.events.append(event)
            
            # 更新情感效价
            if "emotion" in event:
                self._update_emotional_valence(event["emotion"])
    
    def end_episode(self, outcome: Any):
        """结束当前情景"""
        if self.current_episode:
            self.current_episode.end_time = time.time()
            self.current_episode.outcome = outcome
            
            # 存储情景
            self.episodes[self.current_episode.id] = self.current_episode
            
            # 建立索引
            self._index_episode(self.current_episode)
            
            # 重置当前情景
            self.current_episode = None
    
    def recall_similar_episodes(self, query_context: Dict, k: int = 3) -> List[Episode]:
        """回忆相似情景"""
        similar_episodes = []
        
        for episode in self.episodes.values():
            similarity = self._calculate_context_similarity(
                query_context,
                episode.context
            )
            similar_episodes.append((episode, similarity))
        
        # 排序并返回top-k
        similar_episodes.sort(key=lambda x: x[1], reverse=True)
        
        return [ep for ep, _ in similar_episodes[:k]]
    
    def extract_patterns(self) -> Dict:
        """从情景中提取行为模式"""
        patterns = {
            "successful_patterns": [],
            "failure_patterns": [],
            "emotional_triggers": []
        }
        
        for episode in self.episodes.values():
            # 分析成功模式
            if self._is_successful_outcome(episode.outcome):
                pattern = self._extract_action_sequence(episode)
                patterns["successful_patterns"].append(pattern)
            
            # 分析失败模式
            elif self._is_failure_outcome(episode.outcome):
                pattern = self._extract_action_sequence(episode)
                patterns["failure_patterns"].append(pattern)
            
            # 分析情感触发器
            if abs(episode.emotional_valence) > 0.5:
                trigger = self._identify_emotional_trigger(episode)
                patterns["emotional_triggers"].append(trigger)
        
        return patterns
```

### 3.2 情景压缩与摘要

```python
class EpisodeCompression:
    def __init__(self, compression_ratio: float = 0.3):
        self.compression_ratio = compression_ratio
        
    def compress_episode(self, episode: Episode) -> Dict:
        """压缩情景为摘要"""
        # 识别关键事件
        key_events = self._identify_key_events(episode.events)
        
        # 提取转折点
        turning_points = self._find_turning_points(episode.events)
        
        # 生成叙事摘要
        narrative = self._generate_narrative(
            key_events,
            turning_points,
            episode.outcome
        )
        
        compressed = {
            "id": episode.id,
            "duration": episode.end_time - episode.start_time,
            "key_events": key_events,
            "turning_points": turning_points,
            "narrative": narrative,
            "outcome": episode.outcome,
            "emotional_arc": self._extract_emotional_arc(episode)
        }
        
        return compressed
    
    def _identify_key_events(self, events: List[Dict]) -> List[Dict]:
        """识别关键事件"""
        if len(events) <= 5:
            return events
        
        # 计算事件重要性
        event_scores = []
        for event in events:
            score = self._calculate_event_importance(event)
            event_scores.append((event, score))
        
        # 选择top事件
        event_scores.sort(key=lambda x: x[1], reverse=True)
        num_key_events = max(3, int(len(events) * self.compression_ratio))
        
        key_events = [event for event, _ in event_scores[:num_key_events]]
        
        # 保持时间顺序
        key_events.sort(key=lambda x: x["timestamp"])
        
        return key_events
```

## 4. 语义记忆网络

### 4.1 概念网络构建

```python
import networkx as nx

class SemanticMemory:
    def __init__(self):
        self.concept_graph = nx.DiGraph()
        self.concept_embeddings = {}
        self.relation_types = [
            "is_a", "part_of", "has_property",
            "causes", "prevents", "related_to"
        ]
        
    def add_concept(self, concept: str, properties: Dict = None):
        """添加概念节点"""
        if concept not in self.concept_graph:
            self.concept_graph.add_node(
                concept,
                properties=properties or {},
                activation=0.0,
                last_activated=None
            )
            
            # 生成概念嵌入
            self.concept_embeddings[concept] = self._embed_concept(concept)
    
    def add_relation(self, concept1: str, relation: str, concept2: str,
                     strength: float = 1.0):
        """添加概念关系"""
        self.add_concept(concept1)
        self.add_concept(concept2)
        
        self.concept_graph.add_edge(
            concept1, concept2,
            relation=relation,
            strength=strength,
            created_at=time.time()
        )
    
    def activate_concept(self, concept: str, activation: float = 1.0):
        """激活概念（扩散激活）"""
        if concept not in self.concept_graph:
            return
        
        # 设置初始激活
        self.concept_graph.nodes[concept]["activation"] = activation
        self.concept_graph.nodes[concept]["last_activated"] = time.time()
        
        # 扩散激活
        self._spread_activation(concept, activation, decay=0.5, depth=3)
    
    def _spread_activation(self, source: str, activation: float,
                          decay: float, depth: int):
        """扩散激活算法"""
        if depth <= 0 or activation < 0.1:
            return
        
        # 激活相邻节点
        for neighbor in self.concept_graph.neighbors(source):
            edge_data = self.concept_graph[source][neighbor]
            spread_activation = activation * edge_data["strength"] * decay
            
            current_activation = self.concept_graph.nodes[neighbor].get("activation", 0)
            new_activation = current_activation + spread_activation
            
            self.concept_graph.nodes[neighbor]["activation"] = min(1.0, new_activation)
            
            # 递归扩散
            self._spread_activation(neighbor, spread_activation, decay, depth - 1)
    
    def query_concepts(self, query: str, k: int = 5) -> List[str]:
        """查询相关概念"""
        # 激活查询相关概念
        query_concepts = self._extract_concepts_from_text(query)
        for concept in query_concepts:
            self.activate_concept(concept)
        
        # 获取激活度最高的概念
        activated_concepts = [
            (node, data["activation"])
            for node, data in self.concept_graph.nodes(data=True)
            if data["activation"] > 0
        ]
        
        activated_concepts.sort(key=lambda x: x[1], reverse=True)
        
        return [concept for concept, _ in activated_concepts[:k]]
```

### 4.2 知识推理

```python
class SemanticReasoning:
    def __init__(self, semantic_memory: SemanticMemory):
        self.semantic_memory = semantic_memory
        
    def infer_relations(self, concept1: str, concept2: str) -> List[Dict]:
        """推理两个概念之间的关系"""
        inferences = []
        
        # 直接关系
        if self.semantic_memory.concept_graph.has_edge(concept1, concept2):
            edge_data = self.semantic_memory.concept_graph[concept1][concept2]
            inferences.append({
                "type": "direct",
                "relation": edge_data["relation"],
                "confidence": edge_data["strength"]
            })
        
        # 传递关系
        try:
            paths = list(nx.all_simple_paths(
                self.semantic_memory.concept_graph,
                concept1, concept2,
                cutoff=3
            ))
            
            for path in paths:
                if len(path) > 2:
                    inference = self._analyze_path(path)
                    inferences.append(inference)
        except nx.NetworkXNoPath:
            pass
        
        # 类比推理
        analogies = self._find_analogies(concept1, concept2)
        inferences.extend(analogies)
        
        return inferences
    
    def _find_analogies(self, concept1: str, concept2: str) -> List[Dict]:
        """查找类比关系"""
        analogies = []
        
        # 获取concept1的关系模式
        patterns1 = self._get_relation_patterns(concept1)
        
        # 查找相似模式
        for node in self.semantic_memory.concept_graph.nodes():
            if node != concept1:
                patterns = self._get_relation_patterns(node)
                similarity = self._pattern_similarity(patterns1, patterns)
                
                if similarity > 0.7:
                    analogies.append({
                        "type": "analogy",
                        "base": concept1,
                        "target": node,
                        "mapped_to": concept2,
                        "confidence": similarity
                    })
        
        return analogies
```

## 5. 程序记忆系统

### 5.1 技能学习

```python
@dataclass
class Skill:
    name: str
    steps: List[Dict]
    preconditions: List[str]
    postconditions: List[str]
    success_rate: float = 0.0
    execution_count: int = 0
    
class ProceduralMemory:
    def __init__(self):
        self.skills = {}
        self.skill_hierarchy = nx.DiGraph()
        self.execution_history = []
        
    def learn_skill(self, demonstration: List[Dict]) -> Skill:
        """从演示中学习技能"""
        # 提取动作序列
        action_sequence = self._extract_actions(demonstration)
        
        # 识别前置和后置条件
        preconditions = self._identify_preconditions(demonstration[0])
        postconditions = self._identify_postconditions(demonstration[-1])
        
        # 创建技能
        skill = Skill(
            name=self._generate_skill_name(action_sequence),
            steps=action_sequence,
            preconditions=preconditions,
            postconditions=postconditions
        )
        
        # 存储技能
        self.skills[skill.name] = skill
        
        # 更新技能层次
        self._update_skill_hierarchy(skill)
        
        return skill
    
    def execute_skill(self, skill_name: str, context: Dict) -> Dict:
        """执行技能"""
        if skill_name not in self.skills:
            return {"success": False, "error": "Skill not found"}
        
        skill = self.skills[skill_name]
        
        # 检查前置条件
        if not self._check_preconditions(skill.preconditions, context):
            return {"success": False, "error": "Preconditions not met"}
        
        # 执行步骤
        result = {"success": True, "steps_executed": []}
        for step in skill.steps:
            step_result = self._execute_step(step, context)
            result["steps_executed"].append(step_result)
            
            if not step_result["success"]:
                result["success"] = False
                result["error"] = f"Failed at step: {step}"
                break
        
        # 更新技能统计
        skill.execution_count += 1
        if result["success"]:
            skill.success_rate = (
                (skill.success_rate * (skill.execution_count - 1) + 1) /
                skill.execution_count
            )
        else:
            skill.success_rate = (
                (skill.success_rate * (skill.execution_count - 1)) /
                skill.execution_count
            )
        
        # 记录执行历史
        self.execution_history.append({
            "skill": skill_name,
            "context": context,
            "result": result,
            "timestamp": time.time()
        })
        
        return result
    
    def compose_skills(self, goal: str) -> List[str]:
        """组合技能以达成目标"""
        # 查找能达成目标的技能序列
        relevant_skills = self._find_relevant_skills(goal)
        
        # 规划技能执行顺序
        skill_plan = self._plan_skill_sequence(relevant_skills, goal)
        
        return skill_plan
```

### 5.2 技能优化

```python
class SkillOptimizer:
    def __init__(self, procedural_memory: ProceduralMemory):
        self.procedural_memory = procedural_memory
        
    def optimize_skill(self, skill_name: str):
        """优化技能执行"""
        skill = self.procedural_memory.skills.get(skill_name)
        if not skill:
            return
        
        # 分析执行历史
        history = [
            h for h in self.procedural_memory.execution_history
            if h["skill"] == skill_name
        ]
        
        # 识别失败模式
        failure_patterns = self._identify_failure_patterns(history)
        
        # 优化步骤
        optimized_steps = self._optimize_steps(
            skill.steps,
            failure_patterns
        )
        
        # 创建优化版本
        optimized_skill = Skill(
            name=f"{skill_name}_optimized",
            steps=optimized_steps,
            preconditions=skill.preconditions,
            postconditions=skill.postconditions
        )
        
        self.procedural_memory.skills[optimized_skill.name] = optimized_skill
        
        return optimized_skill
    
    def _identify_failure_patterns(self, history: List[Dict]) -> List[Dict]:
        """识别失败模式"""
        failures = [h for h in history if not h["result"]["success"]]
        
        patterns = []
        for failure in failures:
            failed_step = failure["result"].get("error", "")
            context = failure["context"]
            
            pattern = {
                "step": failed_step,
                "context_conditions": self._extract_conditions(context),
                "frequency": 1
            }
            
            # 合并相似模式
            merged = False
            for existing_pattern in patterns:
                if self._patterns_similar(pattern, existing_pattern):
                    existing_pattern["frequency"] += 1
                    merged = True
                    break
            
            if not merged:
                patterns.append(pattern)
        
        return patterns
```

## 6. 记忆检索优化

### 6.1 上下文感知检索

```python
class ContextAwareRetrieval:
    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system
        self.context_window = 10  # 考虑最近10个交互
        
    def retrieve(self, query: str, context: List[Dict]) -> List[Memory]:
        """上下文感知的记忆检索"""
        # 1. 提取上下文特征
        context_features = self._extract_context_features(context)
        
        # 2. 扩展查询
        expanded_query = self._expand_query_with_context(query, context_features)
        
        # 3. 多源检索
        candidates = []
        
        # 从长期记忆检索
        ltm_results = self.memory_system.long_term_memory.retrieve(
            expanded_query, k=10
        )
        candidates.extend(ltm_results)
        
        # 从情景记忆检索
        episodes = self.memory_system.episodic_memory.recall_similar_episodes(
            context_features, k=3
        )
        for episode in episodes:
            candidates.extend(self._extract_memories_from_episode(episode))
        
        # 从语义记忆检索
        concepts = self.memory_system.semantic_memory.query_concepts(
            query, k=5
        )
        for concept in concepts:
            candidates.extend(self._get_concept_memories(concept))
        
        # 4. 重排和去重
        unique_memories = self._deduplicate_memories(candidates)
        ranked_memories = self._rank_by_relevance(
            unique_memories,
            query,
            context_features
        )
        
        return ranked_memories[:5]
    
    def _rank_by_relevance(self, memories: List[Memory], query: str,
                          context: Dict) -> List[Memory]:
        """按相关性排序记忆"""
        scored_memories = []
        
        for memory in memories:
            # 查询相关性
            query_relevance = self._calculate_similarity(
                memory.content, query
            )
            
            # 上下文相关性
            context_relevance = self._calculate_context_relevance(
                memory, context
            )
            
            # 时间相关性
            time_relevance = self._calculate_time_relevance(memory)
            
            # 综合评分
            score = (
                0.4 * query_relevance +
                0.3 * context_relevance +
                0.2 * time_relevance +
                0.1 * memory.importance
            )
            
            scored_memories.append((memory, score))
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, _ in scored_memories]
```

### 6.2 记忆链构建

```python
class MemoryChain:
    def __init__(self):
        self.chain_graph = nx.DiGraph()
        
    def build_memory_chain(self, seed_memory: Memory,
                          memory_pool: List[Memory],
                          max_length: int = 5) -> List[Memory]:
        """构建记忆链"""
        chain = [seed_memory]
        current = seed_memory
        
        while len(chain) < max_length:
            # 找到最相关的下一个记忆
            next_memory = self._find_next_memory(
                current, memory_pool, chain
            )
            
            if next_memory is None:
                break
            
            chain.append(next_memory)
            current = next_memory
        
        return chain
    
    def _find_next_memory(self, current: Memory,
                         candidates: List[Memory],
                         chain: List[Memory]) -> Optional[Memory]:
        """找到链中的下一个记忆"""
        best_memory = None
        best_score = -1
        
        for candidate in candidates:
            if candidate in chain:
                continue
            
            # 计算连接强度
            connection_strength = self._calculate_connection_strength(
                current, candidate
            )
            
            # 计算多样性奖励
            diversity_bonus = self._calculate_diversity_bonus(
                candidate, chain
            )
            
            score = connection_strength + 0.2 * diversity_bonus
            
            if score > best_score:
                best_score = score
                best_memory = candidate
        
        return best_memory if best_score > 0.3 else None
```

## 7. 记忆更新与遗忘

### 7.1 自适应遗忘

```python
class AdaptiveForgetting:
    def __init__(self, base_decay_rate: float = 0.01):
        self.base_decay_rate = base_decay_rate
        
    def update_memories(self, memories: Dict[str, Memory]):
        """更新记忆（包括遗忘）"""
        current_time = time.time()
        to_forget = []
        
        for memory_id, memory in memories.items():
            # 计算遗忘曲线
            time_since_access = current_time - (memory.last_accessed or memory.timestamp)
            time_since_creation = current_time - memory.timestamp
            
            # Ebbinghaus遗忘曲线
            retention = self._calculate_retention(
                time_since_access,
                memory.access_count,
                memory.importance
            )
            
            # 更新记忆强度
            memory.strength = retention
            
            # 标记需要遗忘的记忆
            if retention < 0.1 and time_since_creation > 86400:  # 24小时
                to_forget.append(memory_id)
        
        # 执行遗忘
        for memory_id in to_forget:
            self._forget_memory(memories, memory_id)
    
    def _calculate_retention(self, time_elapsed: float,
                            access_count: int,
                            importance: float) -> float:
        """计算记忆保持率"""
        # 基础遗忘率
        base_retention = np.exp(-self.base_decay_rate * time_elapsed / 3600)
        
        # 重复强化因子
        repetition_factor = 1 + np.log1p(access_count) * 0.1
        
        # 重要性调节
        importance_factor = 1 + importance * 0.5
        
        # 最终保持率
        retention = min(1.0, base_retention * repetition_factor * importance_factor)
        
        return retention
    
    def _forget_memory(self, memories: Dict, memory_id: str):
        """遗忘记忆（不是删除，而是转为痕迹）"""
        memory = memories[memory_id]
        
        # 保留痕迹
        trace = {
            "id": memory_id,
            "summary": self._create_summary(memory.content),
            "timestamp": memory.timestamp,
            "importance": memory.importance * 0.1
        }
        
        # 存储痕迹（可以用于后续的重建）
        self._store_trace(trace)
        
        # 从活跃记忆中移除
        del memories[memory_id]
```

## 8. 记忆系统集成

### 8.1 统一记忆接口

```python
class UnifiedMemoryInterface:
    def __init__(self):
        self.memory_system = MemorySystem()
        self.retrieval = ContextAwareRetrieval(self.memory_system)
        self.forgetting = AdaptiveForgetting()
        
    async def remember(self, content: Any, memory_type: MemoryType = None):
        """记住信息"""
        # 自动判断记忆类型
        if memory_type is None:
            memory_type = self._infer_memory_type(content)
        
        if memory_type == MemoryType.EPISODIC:
            self.memory_system.episodic_memory.add_event(content)
        elif memory_type == MemoryType.SEMANTIC:
            concepts = self._extract_concepts(content)
            for concept in concepts:
                self.memory_system.semantic_memory.add_concept(concept)
        elif memory_type == MemoryType.PROCEDURAL:
            if self._is_skill_demonstration(content):
                self.memory_system.procedural_memory.learn_skill(content)
        else:
            # 默认存储到长期记忆
            self.memory_system.long_term_memory.store(content)
    
    async def recall(self, query: str, context: List[Dict] = None) -> List[Any]:
        """回忆信息"""
        # 并行从各种记忆类型检索
        tasks = [
            self._recall_from_ltm(query),
            self._recall_from_episodic(query, context),
            self._recall_from_semantic(query),
            self._recall_from_procedural(query)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 合并和排序结果
        all_memories = []
        for result in results:
            all_memories.extend(result)
        
        # 去重和排序
        unique_memories = self._deduplicate(all_memories)
        ranked_memories = self._rank_memories(unique_memories, query)
        
        return ranked_memories[:10]
    
    def reflect(self) -> Dict:
        """反思和总结记忆"""
        reflection = {
            "patterns": self.memory_system.episodic_memory.extract_patterns(),
            "important_concepts": self._get_important_concepts(),
            "skill_improvements": self._suggest_skill_improvements(),
            "memory_statistics": self._get_memory_stats()
        }
        
        return reflection
```

## 9. 最佳实践

1. **分层存储**：根据访问频率和重要性分层
2. **智能遗忘**：模拟人类遗忘曲线
3. **关联构建**：自动构建记忆间的关联
4. **上下文感知**：考虑当前上下文进行检索
5. **持续学习**：从交互中不断优化记忆系统
6. **压缩策略**：定期压缩和总结记忆

## 结论

一个优秀的Agent记忆系统不仅要能存储和检索信息，还要能够学习、关联、遗忘和总结。通过模拟人类记忆的多层次结构和处理机制，我们可以构建出更智能、更有"记忆"的AI Agent。

## 参考资源

- [Memory Networks](https://arxiv.org/abs/1410.3916)
- [Neural Turing Machines](https://arxiv.org/abs/1410.5401)
- [Generative Agents: Interactive Simulacra](https://arxiv.org/abs/2304.03442)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)