---
layout: single
title: "Python网页爬虫进阶教程：从基础到反爬虫对抗"
date: 2024-01-06 10:15:00 +0800
categories: [Python, 数据采集]
tags: [Python, 爬虫, Scrapy, 反爬虫, 数据挖掘]
---

网页爬虫是数据科学和自动化领域的重要技能。本文将从基础概念开始，逐步深入到高级技巧和反爬虫对抗策略，帮助你构建稳定高效的爬虫系统。

## 爬虫基础概念

网页爬虫（Web Crawler）是一种自动获取网页内容的程序，主要用于：

- **数据收集**：获取网站的公开信息
- **价格监控**：跟踪商品价格变化
- **新闻聚合**：收集多个新闻源的文章
- **SEO分析**：分析网站结构和内容

## 基础爬虫实现

### 1. 使用requests和BeautifulSoup

```python
import requests
from bs4 import BeautifulSoup
import time
import csv
from urllib.parse import urljoin, urlparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicCrawler:
    def __init__(self, base_url, headers=None):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_page(self, url, **kwargs):
        """获取网页内容"""
        try:
            response = self.session.get(url, timeout=10, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"请求失败 {url}: {e}")
            return None
    
    def parse_page(self, html_content, url):
        """解析网页内容 - 子类重写此方法"""
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup
    
    def crawl(self, urls, delay=1):
        """批量爬取网页"""
        results = []
        for url in urls:
            logger.info(f"正在爬取: {url}")
            
            response = self.get_page(url)
            if response:
                data = self.parse_page(response.text, url)
                if data:
                    results.append(data)
            
            time.sleep(delay)  # 避免请求过于频繁
        
        return results

# 实际应用示例：爬取新闻网站
class NewsCrawler(BasicCrawler):
    def parse_page(self, html_content, url):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 提取文章信息
        article_data = {
            'url': url,
            'title': self.extract_title(soup),
            'content': self.extract_content(soup),
            'publish_time': self.extract_publish_time(soup),
            'author': self.extract_author(soup)
        }
        
        return article_data
    
    def extract_title(self, soup):
        """提取标题"""
        selectors = ['h1', '.title', '#title', '[class*="title"]']
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return None
    
    def extract_content(self, soup):
        """提取正文内容"""
        # 移除无关元素
        for element in soup.find_all(['script', 'style', 'nav', 'footer']):
            element.decompose()
        
        # 尝试多种选择器
        content_selectors = [
            '.article-content', '.post-content', 
            '#content', '[class*="content"]'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        # 兜底方案：获取最长的段落集合
        paragraphs = soup.find_all('p')
        if paragraphs:
            return '\n'.join([p.get_text().strip() for p in paragraphs])
        
        return None
    
    def extract_publish_time(self, soup):
        """提取发布时间"""
        time_selectors = [
            'time', '.publish-time', '.date', 
            '[datetime]', '[class*="time"]'
        ]
        
        for selector in time_selectors:
            element = soup.select_one(selector)
            if element:
                # 尝试获取datetime属性
                datetime_attr = element.get('datetime')
                if datetime_attr:
                    return datetime_attr
                return element.get_text().strip()
        
        return None
    
    def extract_author(self, soup):
        """提取作者信息"""
        author_selectors = [
            '.author', '.writer', '[class*="author"]',
            '[rel="author"]'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        return None

# 使用示例
crawler = NewsCrawler('https://example-news.com')
news_urls = [
    'https://example-news.com/article1',
    'https://example-news.com/article2'
]
articles = crawler.crawl(news_urls)
```

### 2. 处理JavaScript渲染的页面

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import json

class JavaScriptCrawler:
    def __init__(self, headless=True):
        self.options = Options()
        if headless:
            self.options.add_argument('--headless')
        
        # 优化参数
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--window-size=1920,1080')
        
        self.driver = None
    
    def __enter__(self):
        self.driver = webdriver.Chrome(options=self.options)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()
    
    def get_page(self, url, wait_element=None, timeout=10):
        """获取JavaScript渲染后的页面"""
        self.driver.get(url)
        
        if wait_element:
            # 等待特定元素加载
            wait = WebDriverWait(self.driver, timeout)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_element)))
        
        return self.driver.page_source
    
    def scroll_to_load_more(self, scroll_pause_time=2, max_scrolls=10):
        """滚动加载更多内容"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        scrolls = 0
        
        while scrolls < max_scrolls:
            # 滚动到页面底部
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # 等待新内容加载
            time.sleep(scroll_pause_time)
            
            # 检查是否有新内容
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            
            last_height = new_height
            scrolls += 1
    
    def extract_json_data(self, script_selector):
        """从页面中提取JSON数据"""
        try:
            script_element = self.driver.find_element(By.CSS_SELECTOR, script_selector)
            script_content = script_element.get_attribute('innerHTML')
            return json.loads(script_content)
        except Exception as e:
            logger.error(f"提取JSON数据失败: {e}")
            return None

# 使用示例：爬取动态加载的商品列表
def crawl_dynamic_products():
    with JavaScriptCrawler() as crawler:
        url = 'https://example-shop.com/products'
        
        # 获取初始页面
        html = crawler.get_page(url, wait_element='.product-item')
        
        # 滚动加载更多商品
        crawler.scroll_to_load_more()
        
        # 获取最终页面内容
        soup = BeautifulSoup(crawler.driver.page_source, 'html.parser')
        
        # 提取商品信息
        products = []
        for item in soup.select('.product-item'):
            product = {
                'name': item.select_one('.product-name').get_text().strip(),
                'price': item.select_one('.price').get_text().strip(),
                'image': item.select_one('img')['src'],
                'link': item.select_one('a')['href']
            }
            products.append(product)
        
        return products
```

## 高级爬虫框架：Scrapy

### 1. Scrapy项目结构

```python
# 创建Scrapy项目
# scrapy startproject myspider

# spiders/news_spider.py
import scrapy
from scrapy import Request
from urllib.parse import urljoin
import json

class NewsSpider(scrapy.Spider):
    name = 'news'
    allowed_domains = ['example-news.com']
    start_urls = ['https://example-news.com']
    
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS': 16,
        'COOKIES_ENABLED': True,
        'TELNETCONSOLE_ENABLED': False,
    }
    
    def parse(self, response):
        """解析首页，提取文章链接"""
        # 提取文章链接
        article_links = response.css('.article-list a::attr(href)').getall()
        
        for link in article_links:
            absolute_url = urljoin(response.url, link)
            yield Request(
                url=absolute_url,
                callback=self.parse_article,
                meta={'dont_cache': True}
            )
        
        # 处理分页
        next_page = response.css('.pagination .next::attr(href)').get()
        if next_page:
            yield Request(url=urljoin(response.url, next_page), callback=self.parse)
    
    def parse_article(self, response):
        """解析文章页面"""
        # 使用CSS选择器提取数据
        title = response.css('h1::text').get()
        content = ' '.join(response.css('.article-content p::text').getall())
        author = response.css('.author::text').get()
        publish_time = response.css('time::attr(datetime)').get()
        
        # 使用XPath提取数据
        tags = response.xpath('//div[@class="tags"]/a/text()').getall()
        
        yield {
            'url': response.url,
            'title': title.strip() if title else None,
            'content': content.strip() if content else None,
            'author': author.strip() if author else None,
            'publish_time': publish_time,
            'tags': tags,
        }

# items.py - 定义数据结构
import scrapy

class ArticleItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    author = scrapy.Field()
    publish_time = scrapy.Field()
    tags = scrapy.Field()

# pipelines.py - 数据处理管道
import json
import mysql.connector
from itemadapter import ItemAdapter

class JsonWriterPipeline:
    def open_spider(self, spider):
        self.file = open(f'{spider.name}_items.json', 'w', encoding='utf-8')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(ItemAdapter(item).asdict(), ensure_ascii=False) + "\n"
        self.file.write(line)
        return item

class DatabasePipeline:
    def __init__(self, mysql_host, mysql_db, mysql_user, mysql_password, mysql_port):
        self.mysql_host = mysql_host
        self.mysql_db = mysql_db
        self.mysql_user = mysql_user
        self.mysql_password = mysql_password
        self.mysql_port = mysql_port

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mysql_host=crawler.settings.get("MYSQL_HOST"),
            mysql_db=crawler.settings.get("MYSQL_DATABASE"),
            mysql_user=crawler.settings.get("MYSQL_USER"),
            mysql_password=crawler.settings.get("MYSQL_PASSWORD"),
            mysql_port=crawler.settings.get("MYSQL_PORT"),
        )

    def open_spider(self, spider):
        self.connection = mysql.connector.connect(
            host=self.mysql_host,
            database=self.mysql_db,
            user=self.mysql_user,
            password=self.mysql_password,
            port=self.mysql_port
        )
        self.cursor = self.connection.cursor()

    def close_spider(self, spider):
        self.connection.close()

    def process_item(self, item, spider):
        try:
            adapter = ItemAdapter(item)
            insert_sql = """
                INSERT INTO articles (url, title, content, author, publish_time, tags) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.cursor.execute(insert_sql, (
                adapter.get('url'),
                adapter.get('title'),
                adapter.get('content'),
                adapter.get('author'),
                adapter.get('publish_time'),
                ','.join(adapter.get('tags', []))
            ))
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            spider.logger.error(f"数据库插入失败: {e}")
        
        return item
```

### 2. 中间件和自定义设置

```python
# middlewares.py
import random
import requests
from scrapy.http import HtmlResponse
from scrapy.downloadermiddlewares.retry import RetryMiddleware

class ProxyMiddleware:
    """代理中间件"""
    def __init__(self):
        self.proxies = [
            'http://proxy1:port',
            'http://proxy2:port',
            'http://proxy3:port',
        ]
    
    def process_request(self, request, spider):
        proxy = random.choice(self.proxies)
        request.meta['proxy'] = proxy

class UserAgentMiddleware:
    """用户代理中间件"""
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        ]
    
    def process_request(self, request, spider):
        ua = random.choice(self.user_agents)
        request.headers['User-Agent'] = ua

class CustomRetryMiddleware(RetryMiddleware):
    """自定义重试中间件"""
    def process_response(self, request, response, spider):
        if response.status in [403, 429, 503]:
            spider.logger.warning(f"遇到反爬虫 {response.status}，准备重试")
            return self._retry(request, f"HTTP {response.status}", spider)
        
        return response

# settings.py
BOT_NAME = 'myspider'

SPIDER_MODULES = ['myspider.spiders']
NEWSPIDER_MODULE = 'myspider.spiders'

# 遵守robots.txt
ROBOTSTXT_OBEY = True

# 并发设置
CONCURRENT_REQUESTS = 32
CONCURRENT_REQUESTS_PER_DOMAIN = 16

# 下载延迟
DOWNLOAD_DELAY = 3
RANDOMIZE_DOWNLOAD_DELAY = 0.5

# 中间件配置
DOWNLOADER_MIDDLEWARES = {
    'myspider.middlewares.ProxyMiddleware': 350,
    'myspider.middlewares.UserAgentMiddleware': 400,
    'myspider.middlewares.CustomRetryMiddleware': 550,
}

# 管道配置
ITEM_PIPELINES = {
    'myspider.pipelines.JsonWriterPipeline': 300,
    'myspider.pipelines.DatabasePipeline': 400,
}

# 数据库配置
MYSQL_HOST = 'localhost'
MYSQL_DATABASE = 'crawler_db'
MYSQL_USER = 'user'
MYSQL_PASSWORD = 'password'
MYSQL_PORT = 3306

# 缓存配置
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600
```

## 反爬虫对抗策略

### 1. 请求头伪装

```python
import random
import time
from fake_useragent import UserAgent

class AntiDetectionCrawler:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        
    def get_random_headers(self):
        """生成随机请求头"""
        headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': random.choice(['en-US,en;q=0.5', 'zh-CN,zh;q=0.9']),
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': random.choice(['no-cache', 'max-age=0']),
        }
        
        # 随机添加一些可选头部
        optional_headers = {
            'Referer': 'https://www.google.com/',
            'X-Requested-With': 'XMLHttpRequest',
            'Pragma': 'no-cache',
        }
        
        for key, value in optional_headers.items():
            if random.random() > 0.5:
                headers[key] = value
        
        return headers
    
    def smart_delay(self, base_delay=2):
        """智能延迟"""
        # 使用正态分布生成延迟时间
        delay = random.normalvariate(base_delay, base_delay * 0.3)
        delay = max(1, delay)  # 确保延迟至少1秒
        time.sleep(delay)
    
    def get_page_with_retry(self, url, max_retries=3):
        """带重试的页面获取"""
        for attempt in range(max_retries):
            try:
                headers = self.get_random_headers()
                response = self.session.get(
                    url, 
                    headers=headers,
                    timeout=30,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    # 请求过快，增加延迟
                    time.sleep(60 * (attempt + 1))
                elif response.status_code in [403, 503]:
                    # 可能被封，更换策略
                    self.session.cookies.clear()
                    time.sleep(30 * (attempt + 1))
                
            except Exception as e:
                logger.warning(f"请求失败 {attempt + 1}/{max_retries}: {e}")
                time.sleep(10 * (attempt + 1))
        
        return None
```

### 2. 验证码处理

```python
import cv2
import numpy as np
from PIL import Image
import pytesseract
import requests

class CaptchaHandler:
    def __init__(self):
        # 配置tesseract路径（Windows需要）
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass
    
    def download_captcha(self, captcha_url, session):
        """下载验证码图片"""
        response = session.get(captcha_url)
        with open('captcha.png', 'wb') as f:
            f.write(response.content)
        return 'captcha.png'
    
    def preprocess_image(self, image_path):
        """预处理验证码图片"""
        # 读取图片
        image = cv2.imread(image_path)
        
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 降噪
        denoised = cv2.medianBlur(gray, 3)
        
        # 二值化
        _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去除噪点
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 保存预处理后的图片
        processed_path = 'captcha_processed.png'
        cv2.imwrite(processed_path, cleaned)
        
        return processed_path
    
    def recognize_captcha(self, image_path):
        """识别验证码"""
        # 预处理图片
        processed_path = self.preprocess_image(image_path)
        
        # 使用tesseract识别
        try:
            # 配置OCR参数
            config = '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            text = pytesseract.image_to_string(
                Image.open(processed_path), 
                config=config
            ).strip()
            return text
        except Exception as e:
            logger.error(f"验证码识别失败: {e}")
            return None
    
    def solve_captcha_with_api(self, image_path, api_key):
        """使用第三方API识别验证码"""
        # 这里以2captcha为例
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'key': api_key,
                    'method': 'post'
                }
                
                # 提交验证码
                response = requests.post(
                    'http://2captcha.com/in.php',
                    files=files,
                    data=data
                )
                
                if response.text.startswith('OK|'):
                    captcha_id = response.text.split('|')[1]
                    
                    # 等待识别结果
                    time.sleep(10)
                    
                    result_response = requests.get(
                        f'http://2captcha.com/res.php?key={api_key}&action=get&id={captcha_id}'
                    )
                    
                    if result_response.text.startswith('OK|'):
                        return result_response.text.split('|')[1]
                    
        except Exception as e:
            logger.error(f"API识别验证码失败: {e}")
        
        return None

# 使用示例
def handle_login_with_captcha():
    crawler = AntiDetectionCrawler()
    captcha_handler = CaptchaHandler()
    
    # 获取登录页面
    login_url = 'https://example.com/login'
    response = crawler.get_page_with_retry(login_url)
    
    if response:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找验证码图片
        captcha_img = soup.find('img', {'id': 'captcha'})
        if captcha_img:
            captcha_url = urljoin(login_url, captcha_img['src'])
            
            # 下载验证码
            captcha_path = captcha_handler.download_captcha(captcha_url, crawler.session)
            
            # 识别验证码
            captcha_text = captcha_handler.recognize_captcha(captcha_path)
            
            if captcha_text:
                # 提交登录表单
                login_data = {
                    'username': 'your_username',
                    'password': 'your_password',
                    'captcha': captcha_text
                }
                
                response = crawler.session.post(login_url, data=login_data)
                if '登录成功' in response.text:
                    logger.info("登录成功！")
                    return True
    
    return False
```

### 3. 分布式爬虫

```python
# 使用Scrapy-Redis实现分布式爬虫
# distributed_spider.py

from scrapy_redis.spiders import RedisSpider
import scrapy

class DistributedNewsSpider(RedisSpider):
    name = 'distributed_news'
    redis_key = 'news:start_urls'
    
    def __init__(self, *args, **kwargs):
        super(DistributedNewsSpider, self).__init__(*args, **kwargs)
    
    def parse(self, response):
        # 提取文章链接
        for link in response.css('.article-list a::attr(href)').getall():
            yield scrapy.Request(
                url=response.urljoin(link),
                callback=self.parse_article
            )
        
        # 提取下一页链接并加入队列
        next_page = response.css('.next-page::attr(href)').get()
        if next_page:
            yield scrapy.Request(url=response.urljoin(next_page))
    
    def parse_article(self, response):
        yield {
            'title': response.css('h1::text').get(),
            'content': response.css('.content::text').getall(),
            'url': response.url,
        }

# settings.py for distributed crawling
SCHEDULER = "scrapy_redis.scheduler.Scheduler"
DUPEFILTER_CLASS = "scrapy_redis.dupefilter.RFPDupeFilter"
SCHEDULER_PERSIST = True

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

# 启动多个爬虫实例的脚本
import redis
import subprocess
import time

def start_distributed_crawling():
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    # 添加起始URL到Redis队列
    start_urls = [
        'https://news1.com',
        'https://news2.com',
        'https://news3.com',
    ]
    
    for url in start_urls:
        r.lpush('news:start_urls', url)
    
    # 启动多个爬虫进程
    processes = []
    for i in range(4):  # 启动4个爬虫实例
        cmd = ['scrapy', 'crawl', 'distributed_news']
        process = subprocess.Popen(cmd)
        processes.append(process)
        time.sleep(2)  # 间隔启动
    
    # 等待所有进程完成
    for process in processes:
        process.wait()

if __name__ == '__main__':
    start_distributed_crawling()
```

## 数据存储和处理

### 1. 多种存储方案

```python
import sqlite3
import pymongo
import pandas as pd
from sqlalchemy import create_engine

class DataStorage:
    def __init__(self):
        self.connections = {}
    
    def save_to_sqlite(self, data, db_path, table_name):
        """保存到SQLite数据库"""
        conn = sqlite3.connect(db_path)
        df = pd.DataFrame(data)
        df.to_sql(table_name, conn, if_exists='append', index=False)
        conn.close()
    
    def save_to_mysql(self, data, connection_string, table_name):
        """保存到MySQL数据库"""
        engine = create_engine(connection_string)
        df = pd.DataFrame(data)
        df.to_sql(table_name, engine, if_exists='append', index=False)
    
    def save_to_mongodb(self, data, host, port, db_name, collection_name):
        """保存到MongoDB"""
        client = pymongo.MongoClient(host, port)
        db = client[db_name]
        collection = db[collection_name]
        
        if isinstance(data, list):
            collection.insert_many(data)
        else:
            collection.insert_one(data)
    
    def save_to_csv(self, data, filename):
        """保存到CSV文件"""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8')
    
    def save_to_excel(self, data, filename, sheet_name='Sheet1'):
        """保存到Excel文件"""
        df = pd.DataFrame(data)
        df.to_excel(filename, sheet_name=sheet_name, index=False)

# 数据清洗和处理
class DataProcessor:
    def clean_text(self, text):
        """清洗文本数据"""
        if not text:
            return ''
        
        # 移除多余空白字符
        text = ' '.join(text.split())
        
        # 移除特殊字符
        import re
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        
        return text.strip()
    
    def extract_numbers(self, text):
        """从文本中提取数字"""
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        return [float(num) for num in numbers]
    
    def standardize_date(self, date_str):
        """标准化日期格式"""
        import dateparser
        try:
            return dateparser.parse(date_str)
        except:
            return None
    
    def deduplicate_data(self, data, key_field):
        """去重"""
        seen = set()
        unique_data = []
        
        for item in data:
            key = item.get(key_field)
            if key and key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        return unique_data
```

## 监控和维护

### 1. 爬虫监控系统

```python
import smtplib
from email.mime.text import MimeText
from datetime import datetime, timedelta
import psutil
import logging

class CrawlerMonitor:
    def __init__(self, email_config=None):
        self.email_config = email_config
        self.logger = logging.getLogger(__name__)
        
    def check_crawler_health(self, process_name):
        """检查爬虫进程健康状态"""
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            if process_name in ' '.join(process.info['cmdline'] or []):
                return {
                    'status': 'running',
                    'pid': process.info['pid'],
                    'memory_usage': process.memory_info().rss / 1024 / 1024,  # MB
                    'cpu_percent': process.cpu_percent()
                }
        
        return {'status': 'stopped'}
    
    def check_data_freshness(self, db_connection, table_name, time_field):
        """检查数据新鲜度"""
        cursor = db_connection.cursor()
        cursor.execute(f"""
            SELECT MAX({time_field}) as latest_time 
            FROM {table_name}
        """)
        result = cursor.fetchone()
        
        if result and result[0]:
            latest_time = result[0]
            now = datetime.now()
            
            if isinstance(latest_time, str):
                latest_time = datetime.fromisoformat(latest_time)
            
            time_diff = now - latest_time
            
            return {
                'latest_data_time': latest_time,
                'hours_since_update': time_diff.total_seconds() / 3600,
                'is_fresh': time_diff < timedelta(hours=24)
            }
        
        return {'is_fresh': False, 'latest_data_time': None}
    
    def send_alert(self, subject, message):
        """发送告警邮件"""
        if not self.email_config:
            self.logger.warning("邮件配置未设置，无法发送告警")
            return
        
        try:
            msg = MimeText(message, 'plain', 'utf-8')
            msg['Subject'] = subject
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            
            server = smtplib.SMTP(self.email_config['smtp_host'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"告警邮件已发送: {subject}")
        except Exception as e:
            self.logger.error(f"发送告警邮件失败: {e}")
    
    def generate_report(self, crawler_stats):
        """生成监控报告"""
        report = f"""
        爬虫监控报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        爬虫状态: {crawler_stats.get('status', 'unknown')}
        进程ID: {crawler_stats.get('pid', 'N/A')}
        内存使用: {crawler_stats.get('memory_usage', 0):.2f} MB
        CPU使用率: {crawler_stats.get('cpu_percent', 0):.2f}%
        
        数据状态:
        最新数据时间: {crawler_stats.get('latest_data_time', 'N/A')}
        数据新鲜度: {'正常' if crawler_stats.get('is_fresh', False) else '异常'}
        """
        
        return report

# 使用示例
def monitor_crawler():
    email_config = {
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_password',
        'from': 'your_email@gmail.com',
        'to': 'admin@company.com'
    }
    
    monitor = CrawlerMonitor(email_config)
    
    # 检查爬虫状态
    health_status = monitor.check_crawler_health('scrapy')
    
    if health_status['status'] != 'running':
        monitor.send_alert(
            '爬虫告警：进程已停止',
            '爬虫进程未运行，请检查并重启。'
        )
    
    # 检查数据新鲜度
    import sqlite3
    conn = sqlite3.connect('crawler_data.db')
    data_status = monitor.check_data_freshness(conn, 'articles', 'created_at')
    
    if not data_status['is_fresh']:
        monitor.send_alert(
            '爬虫告警：数据更新异常',
            f'数据已超过24小时未更新，最新数据时间：{data_status["latest_data_time"]}'
        )
    
    # 生成报告
    all_stats = {**health_status, **data_status}
    report = monitor.generate_report(all_stats)
    print(report)

if __name__ == '__main__':
    monitor_crawler()
```

## 最佳实践总结

### 1. 代码规范
- **模块化设计**：将爬虫功能拆分为独立模块
- **异常处理**：完善的错误处理和重试机制
- **日志记录**：详细的日志记录便于调试
- **配置管理**：使用配置文件管理参数

### 2. 性能优化
- **并发控制**：合理设置并发请求数
- **缓存机制**：避免重复请求相同内容
- **数据库优化**：使用索引和批量插入
- **内存管理**：及时释放不需要的对象

### 3. 法律合规
- **遵守robots.txt**：尊重网站的爬虫协议
- **合理频率**：避免对目标网站造成压力
- **数据使用**：确保数据使用符合法律法规
- **隐私保护**：不抓取个人隐私信息

## 总结

网页爬虫技术涉及多个方面：

1. **基础技能**：HTTP协议、HTML解析、正则表达式
2. **高级技术**：JavaScript渲染、反爬虫对抗、分布式爬虫
3. **工程实践**：代码架构、数据存储、监控维护
4. **法律意识**：遵守相关法律法规和网站协议

掌握这些技能将帮助你构建高效、稳定、合规的爬虫系统，为数据驱动的决策提供有力支持！

---

*你在爬虫开发中遇到过哪些有趣的挑战？欢迎分享你的解决方案和心得体会！* 