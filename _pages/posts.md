---
title: "全部文章"
permalink: /posts/
layout: posts
author_profile: true
classes: wide
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: /assets/images/posts-bg.jpg
  actions:
    - label: "分类浏览"
      url: "/categories/"
      btn_class: "btn--primary"
    - label: "标签浏览"
      url: "/tags/"
      btn_class: "btn--inverse"
excerpt: "探索技术世界，分享实战经验"
---

<div class="posts-hero">
  <div class="hero-content">
    <h1 class="hero-title">📝 技术文章</h1>
    <p class="hero-subtitle">分享AI、实时通信和前沿技术的实战经验</p>
  </div>
</div>

<div class="posts-filter">
  <div class="filter-tabs">
    <button class="filter-tab active" data-filter="all">全部文章</button>
    <button class="filter-tab" data-filter="ai">对话式AI</button>
    <button class="filter-tab" data-filter="webrtc">实时通信</button>
    <button class="filter-tab" data-filter="voice">语音技术</button>
    <button class="filter-tab" data-filter="cloud">云原生</button>
  </div>
  
  <div class="search-box">
    <input type="text" id="search-input" placeholder="搜索文章...">
    <i class="fas fa-search"></i>
  </div>
</div>

<style>
/* Posts Page Styles */
.posts-hero {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 4rem 0;
  margin: -2rem -2rem 3rem -2rem;
  text-align: center;
  color: white;
  border-radius: 0 0 20px 20px;
}

.hero-title {
  font-size: 3rem;
  font-weight: 800;
  margin-bottom: 1rem;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.hero-subtitle {
  font-size: 1.3rem;
  opacity: 0.9;
  font-weight: 400;
}

/* Filter Section */
.posts-filter {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 2rem 0;
  flex-wrap: wrap;
  gap: 1rem;
}

.filter-tabs {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.filter-tab {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 25px;
  background: #f1f5f9;
  color: #4a5568;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.filter-tab:hover {
  background: #e2e8f0;
  transform: translateY(-2px);
}

.filter-tab.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.search-box {
  position: relative;
  width: 300px;
  max-width: 100%;
}

.search-box input {
  width: 100%;
  padding: 0.75rem 1rem 0.75rem 3rem;
  border: 2px solid #e2e8f0;
  border-radius: 25px;
  font-size: 1rem;
  transition: all 0.3s ease;
}

.search-box input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.search-box i {
  position: absolute;
  left: 1rem;
  top: 50%;
  transform: translateY(-50%);
  color: #a0aec0;
}

/* Enhanced Posts Grid */
.list__item {
  margin-bottom: 2rem;
}

.list__item .archive__item {
  background: white;
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.list__item .archive__item:hover {
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
  color: #667eea !important;
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
  color: #667eea;
}

/* Responsive Design */
@media (max-width: 768px) {
  .hero-title {
    font-size: 2.5rem;
  }
  
  .posts-filter {
    flex-direction: column;
    align-items: stretch;
  }
  
  .filter-tabs {
    justify-content: center;
  }
  
  .search-box {
    width: 100%;
  }
}

@media (max-width: 480px) {
  .posts-hero {
    margin: -1rem -1rem 2rem -1rem;
    padding: 2rem 1rem;
  }
  
  .filter-tabs {
    flex-direction: column;
  }
  
  .filter-tab {
    text-align: center;
  }
  
  .list__item .archive__item {
    padding: 1.5rem;
  }
}

/* JavaScript functionality */
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  // Filter tabs functionality
  const filterTabs = document.querySelectorAll('.filter-tab');
  const posts = document.querySelectorAll('.list__item');
  
  filterTabs.forEach(tab => {
    tab.addEventListener('click', function() {
      // Remove active class from all tabs
      filterTabs.forEach(t => t.classList.remove('active'));
      // Add active class to clicked tab
      this.classList.add('active');
      
      const filter = this.getAttribute('data-filter');
      
      posts.forEach(post => {
        if (filter === 'all') {
          post.style.display = 'block';
        } else {
          // Check if post has the filter category/tag
          const categories = post.querySelector('.archive__item-title')?.textContent.toLowerCase() || '';
          const shouldShow = categories.includes(filter) || 
                           post.classList.contains(filter) ||
                           post.getAttribute('data-category') === filter;
          
          post.style.display = shouldShow ? 'block' : 'none';
        }
      });
    });
  });
  
  // Search functionality
  const searchInput = document.getElementById('search-input');
  
  searchInput.addEventListener('input', function() {
    const searchTerm = this.value.toLowerCase();
    
    posts.forEach(post => {
      const title = post.querySelector('.archive__item-title')?.textContent.toLowerCase() || '';
      const excerpt = post.querySelector('.archive__item-excerpt')?.textContent.toLowerCase() || '';
      
      const shouldShow = title.includes(searchTerm) || excerpt.includes(searchTerm);
      post.style.display = shouldShow ? 'block' : 'none';
    });
  });
});
</script> 