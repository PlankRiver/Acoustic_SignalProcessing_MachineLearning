import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# ================= 配置区域 =================
# 1. 目标网址
TARGET_URL = 'http://books.toscrape.com/'  # 这里以一个专门供爬虫练习的网站为例

# 2. 伪装头 (非常重要，假装自己是浏览器)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


# ===========================================

def get_html(url):
    """
    第一步：发送请求，获取网页源码
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        # 判断请求是否成功 (状态码 200 表示成功)
        if response.status_code == 200:
            response.encoding = response.apparent_encoding  # 自动识别编码，防止中文乱码
            return response.text
        else:
            print(f"请求失败，状态码: {response.status_code}")
            return None
    except Exception as e:
        print(f"请求出错: {e}")
        return None


def parse_data(html_content):
    """
    第二步 & 第三步：解析 HTML 并提取数据
    这里是唯一需要你根据具体网站修改的地方！
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    data_list = []

    # --- 修改开始：寻找数据 ---
    # 例子：我们抓取 books.toscrape.com 的书名和价格
    # 1. 找到所有包含书本信息的容器 (通常是 li 或 div)
    articles = soup.find_all('article', class_='product_pod')

    for article in articles:
        # 提取书名 (在 h3 标签下的 a 标签的 title 属性中)
        title = article.h3.a['title']

        # 提取价格 (在 class 为 price_color 的 p 标签中)
        price = article.find('p', class_='price_color').text

        # 存入字典
        item = {
            '书名': title,
            '价格': price
        }
        data_list.append(item)
    # --- 修改结束 ---

    return data_list


def save_to_csv(data, filename='result.csv'):
    """
    第四步：保存数据到 CSV 表格
    """
    if not data:
        print("没有数据可以保存")
        return

    df = pd.DataFrame(data)
    # index=False 去掉自动生成的序号，encoding='utf-8-sig' 解决Excel打开中文乱码问题
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"成功保存 {len(data)} 条数据到 {filename}")


def main():
    """
    主程序流程控制
    """
    print("开始爬取...")

    # 1. 获取源码
    html = get_html(TARGET_URL)

    # 2. 解析数据
    data = parse_data(html)

    # 3. 打印预览前 3 条
    print("数据预览:")
    for item in data[:3]:
        print(item)

    # 4. 保存文件
    save_to_csv(data, 'book_data.csv')

    print("爬虫结束。")


if __name__ == '__main__':
    main()