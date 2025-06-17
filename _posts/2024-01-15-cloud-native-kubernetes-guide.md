---
layout: post
title: "云原生Kubernetes实战指南：从入门到生产环境部署"
date: 2024-01-15 11:45:00 +0800
categories: [云原生, DevOps]
tags: [Kubernetes, Docker, 容器编排, 微服务, DevOps]
---

Kubernetes已经成为现代云原生应用部署的标准平台。本文将从基础概念开始，逐步深入到生产环境的最佳实践，帮助你全面掌握Kubernetes的核心技能。

## 什么是Kubernetes？

Kubernetes（简称K8s）是一个开源的容器编排系统，用于自动化应用程序的部署、扩展和管理。它提供了：

- **容器编排**：管理大量容器的生命周期
- **服务发现**：自动发现和连接服务
- **负载均衡**：分发流量到健康的实例
- **自动扩缩容**：根据负载自动调整实例数量
- **滚动更新**：零停机时间的应用更新

## 核心概念详解

### 1. 基础对象

```yaml
# pod.yaml - Pod是K8s中最小的部署单元
apiVersion: v1
kind: Pod
metadata:
  name: my-app-pod
  labels:
    app: my-app
    version: v1.0
spec:
  containers:
  - name: app-container
    image: nginx:1.20
    ports:
    - containerPort: 80
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"
    env:
    - name: ENV_VAR
      value: "production"
    livenessProbe:
      httpGet:
        path: /health
        port: 80
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /ready
        port: 80
      initialDelaySeconds: 5
      periodSeconds: 5

---
# service.yaml - Service提供稳定的网络端点
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP
  type: ClusterIP

---
# deployment.yaml - Deployment管理Pod的副本
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-deployment
  labels:
    app: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app-container
        image: nginx:1.20
        ports:
        - containerPort: 80
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
```

### 2. 配置管理

```yaml
# configmap.yaml - 配置数据
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_url: "postgresql://localhost:5432/mydb"
  debug_mode: "false"
  app.properties: |
    server.port=8080
    logging.level=INFO
    cache.enabled=true

---
# secret.yaml - 敏感信息
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  # 注意：这些值需要base64编码
  database-password: cGFzc3dvcmQxMjM=  # password123
  api-key: YWJjZGVmZ2hpams=            # abcdefghijk

---
# deployment-with-config.yaml - 使用配置的部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-with-config
spec:
  replicas: 2
  selector:
    matchLabels:
      app: configured-app
  template:
    metadata:
      labels:
        app: configured-app
    spec:
      containers:
      - name: app
        image: my-app:v1.0
        env:
        # 从ConfigMap获取环境变量
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: database_url
        # 从Secret获取敏感信息
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-password
        volumeMounts:
        # 挂载配置文件
        - name: config-volume
          mountPath: /etc/config
        - name: secret-volume
          mountPath: /etc/secrets
          readOnly: true
      volumes:
      - name: config-volume
        configMap:
          name: app-config
      - name: secret-volume
        secret:
          secretName: app-secrets
```

### 3. 持久化存储

```yaml
# pv.yaml - 持久卷
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mysql-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: standard
  hostPath:
    path: /data/mysql

---
# pvc.yaml - 持久卷声明
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mysql-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard

---
# mysql-deployment.yaml - 使用持久存储的数据库
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mysql
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8.0
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: root-password
        - name: MYSQL_DATABASE
          value: "myapp"
        ports:
        - containerPort: 3306
        volumeMounts:
        - name: mysql-storage
          mountPath: /var/lib/mysql
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
      volumes:
      - name: mysql-storage
        persistentVolumeClaim:
          claimName: mysql-pvc
```

## 高级功能实战

### 1. 自动扩缩容（HPA）

```yaml
# hpa.yaml - 水平Pod自动扩缩容
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 15
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15

---
# vpa.yaml - 垂直Pod自动扩缩容
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: my-app-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app-deployment
  updatePolicy:
    updateMode: "Auto"  # 或 "Off", "Initial"
  resourcePolicy:
    containerPolicies:
    - containerName: app-container
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 128Mi
```

### 2. 网络策略和安全

```yaml
# network-policy.yaml - 网络访问控制
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: app-network-policy
spec:
  podSelector:
    matchLabels:
      app: my-app
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: frontend
    - namespaceSelector:
        matchLabels:
          name: trusted
    ports:
    - protocol: TCP
      port: 80
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432

---
# rbac.yaml - 基于角色的访问控制
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-service-account
  namespace: default

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: app-role
  namespace: default
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "patch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: app-role-binding
  namespace: default
subjects:
- kind: ServiceAccount
  name: app-service-account
  namespace: default
roleRef:
  kind: Role
  name: app-role
  apiGroup: rbac.authorization.k8s.io

---
# pod-security-policy.yaml - Pod安全策略
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### 3. Ingress和负载均衡

```yaml
# ingress.yaml - Ingress资源
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - myapp.example.com
    secretName: app-tls-secret
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /api/v1
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80

---
# load-balancer-service.yaml - LoadBalancer类型服务
apiVersion: v1
kind: Service
metadata:
  name: app-loadbalancer
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
```

## CI/CD集成

### 1. GitLab CI配置

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy-staging
  - deploy-production

variables:
  DOCKER_REGISTRY: registry.gitlab.com/myproject
  KUBE_NAMESPACE_STAGING: myapp-staging
  KUBE_NAMESPACE_PRODUCTION: myapp-production

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $DOCKER_REGISTRY/myapp:$CI_COMMIT_SHA .
    - docker push $DOCKER_REGISTRY/myapp:$CI_COMMIT_SHA
  only:
    - main
    - develop

test:
  stage: test
  image: $DOCKER_REGISTRY/myapp:$CI_COMMIT_SHA
  script:
    - npm test
    - npm run lint
  coverage: '/Coverage: \d+\.\d+%/'

deploy-staging:
  stage: deploy-staging
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_STAGING
    - kubectl set image deployment/myapp-deployment app=$DOCKER_REGISTRY/myapp:$CI_COMMIT_SHA -n $KUBE_NAMESPACE_STAGING
    - kubectl rollout status deployment/myapp-deployment -n $KUBE_NAMESPACE_STAGING
  environment:
    name: staging
    url: https://staging.myapp.com
  only:
    - develop

deploy-production:
  stage: deploy-production
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_PRODUCTION
    - kubectl set image deployment/myapp-deployment app=$DOCKER_REGISTRY/myapp:$CI_COMMIT_SHA -n $KUBE_NAMESPACE_PRODUCTION
    - kubectl rollout status deployment/myapp-deployment -n $KUBE_NAMESPACE_PRODUCTION
  environment:
    name: production
    url: https://myapp.com
  when: manual
  only:
    - main
```

### 2. Helm Charts

```yaml
# Chart.yaml
apiVersion: v2
name: myapp
description: My Application Helm Chart
version: 0.1.0
appVersion: 1.0.0

# values.yaml
replicaCount: 3

image:
  repository: myapp
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
  hosts:
    - host: myapp.local
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPU: 70
  targetMemory: 80

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true

# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "myapp.fullname" . }}
  labels:
    {{- include "myapp.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "myapp.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "myapp.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.targetPort }}
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}

# 部署命令
# helm install myapp ./myapp-chart
# helm upgrade myapp ./myapp-chart --set image.tag=v2.0.0
```

## 监控和日志

### 1. Prometheus监控

```yaml
# monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "alert_rules.yml"
    
    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093
    
    scrape_configs:
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
      
      - job_name: 'kubernetes-services'
        kubernetes_sd_configs:
          - role: service
        relabel_configs:
          - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
            action: keep
            regex: true

---
# monitoring/grafana-dashboard.json
{
  "dashboard": {
    "id": null,
    "title": "Kubernetes Application Monitoring",
    "panels": [
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total{namespace=\"default\"}[5m])",
            "legendFormat": "{{pod}}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_working_set_bytes{namespace=\"default\"}",
            "legendFormat": "{{pod}}"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      }
    ]
  }
}
```

### 2. 日志收集

```yaml
# logging/fluentd-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: fluentd
  template:
    metadata:
      labels:
        name: fluentd
    spec:
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc.cluster.local"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: config-volume
          mountPath: /fluentd/etc
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: config-volume
        configMap:
          name: fluentd-config

---
# logging/fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: kube-system
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name kubernetes
      type_name fluentd
    </match>
```

## 生产环境最佳实践

### 1. 集群安全加固

```bash
#!/bin/bash
# security-hardening.sh

# 1. 启用RBAC
kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch", "list"]
EOF

# 2. 网络策略
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
EOF

# 3. Pod安全标准
kubectl label namespace default pod-security.kubernetes.io/enforce=restricted
kubectl label namespace default pod-security.kubernetes.io/audit=restricted
kubectl label namespace default pod-security.kubernetes.io/warn=restricted

# 4. 启用审计日志
cat << EOF > /etc/kubernetes/audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
- level: Metadata
  resources:
  - group: ""
    resources: ["secrets", "configmaps"]
- level: RequestResponse
  resources:
  - group: ""
    resources: [""]
  - group: "apps"
    resources: ["deployments"]
EOF

# 5. 定期更新和补丁
kubectl get nodes -o wide
kubectl get pods --all-namespaces -o wide | grep -v Running
```

### 2. 备份和灾难恢复

```yaml
# backup/velero-backup.yaml
apiVersion: velero.io/v1
kind: Backup
metadata:
  name: daily-backup
spec:
  includedNamespaces:
  - default
  - production
  excludedResources:
  - events
  - events.events.k8s.io
  storageLocation: default
  ttl: 720h0m0s  # 30 days

---
# backup/schedule-backup.yaml
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: daily-backup-schedule
spec:
  schedule: "0 2 * * *"  # 每天凌晨2点
  template:
    includedNamespaces:
    - default
    - production
    storageLocation: default
    ttl: 720h0m0s

# 恢复命令示例
# velero restore create --from-backup daily-backup-20240115
```

### 3. 性能优化脚本

```python
#!/usr/bin/env python3
# k8s-optimizer.py

import subprocess
import json
import yaml
from datetime import datetime, timedelta

class K8sOptimizer:
    def __init__(self):
        self.kubectl = "kubectl"
    
    def get_resource_usage(self, namespace="default"):
        """获取资源使用统计"""
        cmd = f"{self.kubectl} top pods -n {namespace} --no-headers"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        pods_usage = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split()
                pods_usage.append({
                    'name': parts[0],
                    'cpu': parts[1],
                    'memory': parts[2]
                })
        
        return pods_usage
    
    def find_unused_resources(self):
        """查找未使用的资源"""
        # 查找未使用的ConfigMaps
        cmd = f"{self.kubectl} get configmaps -o json"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        configmaps = json.loads(result.stdout)
        
        unused_configmaps = []
        for cm in configmaps['items']:
            # 检查是否被Pod使用
            cmd = f"{self.kubectl} get pods -o json"
            pods_result = subprocess.run(cmd.split(), capture_output=True, text=True)
            pods = json.loads(pods_result.stdout)
            
            is_used = False
            for pod in pods['items']:
                # 检查环境变量和挂载卷
                spec = pod.get('spec', {})
                containers = spec.get('containers', [])
                
                for container in containers:
                    # 检查环境变量
                    env = container.get('env', [])
                    for env_var in env:
                        if env_var.get('valueFrom', {}).get('configMapKeyRef', {}).get('name') == cm['metadata']['name']:
                            is_used = True
                            break
                    
                    # 检查挂载卷
                    volume_mounts = container.get('volumeMounts', [])
                    volumes = spec.get('volumes', [])
                    for volume in volumes:
                        if volume.get('configMap', {}).get('name') == cm['metadata']['name']:
                            is_used = True
                            break
            
            if not is_used:
                unused_configmaps.append(cm['metadata']['name'])
        
        return {
            'unused_configmaps': unused_configmaps
        }
    
    def optimize_resources(self):
        """资源优化建议"""
        usage = self.get_resource_usage()
        recommendations = []
        
        for pod in usage:
            cpu_val = float(pod['cpu'].replace('m', '')) if 'm' in pod['cpu'] else float(pod['cpu']) * 1000
            memory_val = float(pod['memory'].replace('Mi', ''))
            
            # CPU使用率过低
            if cpu_val < 50:  # 50m
                recommendations.append({
                    'pod': pod['name'],
                    'type': 'CPU_UNDERUTILIZED',
                    'current': pod['cpu'],
                    'suggestion': 'Consider reducing CPU request/limit'
                })
            
            # 内存使用率检查
            if memory_val < 64:  # 64Mi
                recommendations.append({
                    'pod': pod['name'],
                    'type': 'MEMORY_UNDERUTILIZED',
                    'current': pod['memory'],
                    'suggestion': 'Consider reducing memory request/limit'
                })
        
        return recommendations
    
    def generate_report(self):
        """生成优化报告"""
        print("=" * 50)
        print("Kubernetes 集群优化报告")
        print("=" * 50)
        print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 资源使用情况
        usage = self.get_resource_usage()
        print("资源使用情况:")
        print(f"{'Pod名称':<30} {'CPU':<10} {'内存':<10}")
        print("-" * 50)
        for pod in usage:
            print(f"{pod['name']:<30} {pod['cpu']:<10} {pod['memory']:<10}")
        print()
        
        # 优化建议
        recommendations = self.optimize_resources()
        if recommendations:
            print("优化建议:")
            for rec in recommendations:
                print(f"- {rec['pod']}: {rec['suggestion']} (当前: {rec['current']})")
        else:
            print("暂无优化建议")
        print()
        
        # 未使用资源
        unused = self.find_unused_resources()
        if unused['unused_configmaps']:
            print("未使用的ConfigMaps:")
            for cm in unused['unused_configmaps']:
                print(f"- {cm}")
        else:
            print("未发现未使用的ConfigMaps")

if __name__ == "__main__":
    optimizer = K8sOptimizer()
    optimizer.generate_report()
```

## 故障排查指南

### 1. 常用调试命令

```bash
#!/bin/bash
# k8s-debug.sh

# 集群状态检查
echo "=== 集群状态 ==="
kubectl cluster-info
kubectl get nodes -o wide
kubectl get pods --all-namespaces | grep -v Running

# 资源使用情况
echo "=== 资源使用 ==="
kubectl top nodes
kubectl top pods --all-namespaces --sort-by=cpu
kubectl top pods --all-namespaces --sort-by=memory

# 事件查看
echo "=== 最近事件 ==="
kubectl get events --sort-by=.metadata.creationTimestamp

# 网络连接测试
echo "=== 网络测试 ==="
kubectl run tmp-shell --rm -i --tty --image nicolaka/netshoot -- /bin/bash

# 日志查看
echo "=== Pod日志 ==="
kubectl logs -f deployment/my-app --tail=100

# 进入Pod调试
echo "=== Pod调试 ==="
kubectl exec -it pod-name -- /bin/bash

# 端口转发
echo "=== 端口转发 ==="
kubectl port-forward service/my-service 8080:80
```

### 2. 问题诊断脚本

```python
#!/usr/bin/env python3
# k8s-troubleshoot.py

import subprocess
import json
import re
from datetime import datetime

class K8sTroubleshooter:
    def __init__(self):
        self.kubectl = "kubectl"
        self.issues = []
    
    def check_pod_status(self):
        """检查Pod状态"""
        cmd = f"{self.kubectl} get pods --all-namespaces -o json"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        pods = json.loads(result.stdout)
        
        for pod in pods['items']:
            pod_name = pod['metadata']['name']
            namespace = pod['metadata']['namespace']
            status = pod['status']['phase']
            
            if status != 'Running':
                # 获取详细状态
                conditions = pod['status'].get('conditions', [])
                container_statuses = pod['status'].get('containerStatuses', [])
                
                issue = {
                    'type': 'POD_NOT_RUNNING',
                    'pod': pod_name,
                    'namespace': namespace,
                    'status': status,
                    'conditions': conditions,
                    'containers': container_statuses
                }
                self.issues.append(issue)
    
    def check_resource_limits(self):
        """检查资源限制"""
        cmd = f"{self.kubectl} describe nodes"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        # 解析节点资源使用情况
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if 'Allocated resources:' in line:
                # 查找CPU和内存使用率
                for j in range(i+1, min(i+10, len(lines))):
                    if 'cpu' in lines[j]:
                        cpu_match = re.search(r'(\d+)%', lines[j])
                        if cpu_match and int(cpu_match.group(1)) > 80:
                            self.issues.append({
                                'type': 'HIGH_CPU_USAGE',
                                'usage': cpu_match.group(1) + '%'
                            })
                    elif 'memory' in lines[j]:
                        memory_match = re.search(r'(\d+)%', lines[j])
                        if memory_match and int(memory_match.group(1)) > 80:
                            self.issues.append({
                                'type': 'HIGH_MEMORY_USAGE',
                                'usage': memory_match.group(1) + '%'
                            })
    
    def check_storage_issues(self):
        """检查存储问题"""
        cmd = f"{self.kubectl} get pvc --all-namespaces -o json"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        pvcs = json.loads(result.stdout)
        
        for pvc in pvcs['items']:
            status = pvc['status'].get('phase')
            if status != 'Bound':
                self.issues.append({
                    'type': 'PVC_NOT_BOUND',
                    'pvc': pvc['metadata']['name'],
                    'namespace': pvc['metadata']['namespace'],
                    'status': status
                })
    
    def check_network_policies(self):
        """检查网络策略"""
        cmd = f"{self.kubectl} get networkpolicies --all-namespaces -o json"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        if result.returncode == 0:
            policies = json.loads(result.stdout)
            if not policies['items']:
                self.issues.append({
                    'type': 'NO_NETWORK_POLICIES',
                    'message': '未发现网络策略，可能存在安全风险'
                })
    
    def generate_report(self):
        """生成故障排查报告"""
        print("=" * 60)
        print("Kubernetes 故障排查报告")
        print("=" * 60)
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if not self.issues:
            print("✅ 未发现明显问题")
            return
        
        print(f"🔍 发现 {len(self.issues)} 个问题:")
        print()
        
        for i, issue in enumerate(self.issues, 1):
            print(f"{i}. {self.format_issue(issue)}")
            print()
    
    def format_issue(self, issue):
        """格式化问题描述"""
        issue_type = issue['type']
        
        if issue_type == 'POD_NOT_RUNNING':
            return f"Pod {issue['pod']} (namespace: {issue['namespace']}) 状态异常: {issue['status']}"
        
        elif issue_type == 'HIGH_CPU_USAGE':
            return f"节点CPU使用率过高: {issue['usage']}"
        
        elif issue_type == 'HIGH_MEMORY_USAGE':
            return f"节点内存使用率过高: {issue['usage']}"
        
        elif issue_type == 'PVC_NOT_BOUND':
            return f"PVC {issue['pvc']} (namespace: {issue['namespace']}) 未绑定: {issue['status']}"
        
        elif issue_type == 'NO_NETWORK_POLICIES':
            return issue['message']
        
        return f"未知问题: {issue}"
    
    def run_diagnostics(self):
        """运行所有诊断检查"""
        print("正在进行Kubernetes集群诊断...")
        
        self.check_pod_status()
        self.check_resource_limits()
        self.check_storage_issues()
        self.check_network_policies()
        
        self.generate_report()

if __name__ == "__main__":
    troubleshooter = K8sTroubleshooter()
    troubleshooter.run_diagnostics()
```

## 总结

Kubernetes是现代云原生应用的核心平台，掌握其核心概念和最佳实践至关重要：

### 关键要点：
1. **容器编排**：理解Pod、Service、Deployment等基础概念
2. **配置管理**：合理使用ConfigMap和Secret管理配置
3. **存储方案**：正确配置持久化存储
4. **安全策略**：实施RBAC、网络策略等安全措施
5. **监控运维**：建立完整的监控和日志系统
6. **CI/CD集成**：自动化部署和发布流程

### 学习路径：
1. **基础阶段**：理解容器和Kubernetes基本概念
2. **实践阶段**：搭建测试环境，部署简单应用
3. **进阶阶段**：学习高级功能，如HPA、网络策略等
4. **生产阶段**：掌握安全、监控、故障排查等技能

Kubernetes生态系统庞大且不断发展，持续学习和实践是掌握这项技术的关键！

---

*你在Kubernetes实践中遇到过哪些挑战？欢迎分享你的解决方案和最佳实践！* 