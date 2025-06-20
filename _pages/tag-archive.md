---
title: "标签归档"
layout: tags
permalink: /tags/
author_profile: true
header:
  overlay_color: "#3b82f6"
  overlay_filter: "0.3"
  overlay_image: /assets/images/tags-bg.jpg
excerpt: "按标签浏览文章"
---

<style>
/* Tags Page Styles */
.archive__item {
  background: white;
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.archive__item:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
}

.archive__item-title {
  margin-bottom: 1rem !important;
}

.archive__item-title a {
  color: #2d3748 !important;
  text-decoration: none !important;
  font-weight: 700 !important;
  font-size: 1.4rem !important;
}

.archive__item-title a:hover {
  color: #3b82f6 !important;
}

.archive__item-excerpt {
  color: #4a5568;
  line-height: 1.6;
  margin-bottom: 1rem;
}

.page__meta {
  display: flex;
  align-items: center;
  gap: 1rem;
  font-size: 0.9rem;
  color: #718096;
  margin-bottom: 1rem;
}

.page__meta i {
  color: #3b82f6;
}

/* Tag badges */
.archive__item-tags {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-top: 1rem;
}

.archive__item-tags a {
  background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 15px;
  font-size: 0.8rem;
  font-weight: 500;
  text-decoration: none;
  transition: all 0.3s ease;
}

.archive__item-tags a:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
}
</style> 