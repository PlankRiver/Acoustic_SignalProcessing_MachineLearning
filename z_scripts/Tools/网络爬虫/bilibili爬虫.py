import requests
import pandas as pd

# ================= 唯一需要修改的地方 =================

# 【请在这里粘贴你刚才复制的 Cookie】
# 格式必须是： '你的Cookie内容' (保留单引号)
MY_COOKIE = """buvid3=B5E22E5E-F211-587B-1868-88CA5497ADD259595infoc; b_nut=1756380859; buvid_fp=723b1ea57d53c3894952e58e7157ee5c; _uuid=6AAE10DF8-F3FD-F378-ED7E-E104FA4D3EBB267994infoc; buvid4=4F7919EF-2B67-3A22-8AD6-357639BD68C926560-025040214-XWlL8HVBSaYYbbqSqZLi1A%3D%3D; enable_web_push=DISABLE; home_feed_column=5; rpdid=|(kJYYJRl~m)0J'u~lYJ)lu)R; DedeUserID=454454696; DedeUserID__ckMd5=8d10feb2f7358f54; theme-tip-show=SHOWED; theme-avatar-tip-show=SHOWED; theme-switch-show=SHOWED; CURRENT_FNVAL=4048; CURRENT_QUALITY=32; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3Njg5OTI5NjYsImlhdCI6MTc2ODczMzcwNiwicGx0IjotMX0.FLAvzL38YzBvbiay50bXICOiS-O9KA2bdRat7FqGNsI; bili_ticket_expires=1768992906; SESSDATA=05215c65%2C1784430018%2C77687%2A12CjD1kH7uYUYZ8DOpuc2bkeP7GU2lofHS5Pfhfmj-YNlRGbVpob5Eax3XbcCUEI6mKCESVkE0S21GUEhiQjdFNFNVRHphd3JHTGRJR3RkNlRZU2IzRGVfa1dYbndnSHFURzJCQ0hJNDR1c281RFlPalNpS211eU82bUFfekh0YTFsTlJtMkcyM1pBIIEC; bili_jct=62902b9e5a2561c1cb8525429eb328a6; bmg_af_switch=1; bmg_src_def_domain=i1.hdslb.com; b_lsid=A94AD8C2_19BDA41C2EF; sid=5x4ns270; browser_resolution=1528-732; bsource=search_google; bp_t_offset_454454696=1159890286613102592"""

# ====================================================

TARGET_URL = 'https://api.bilibili.com/x/web-interface/ranking/v2?rid=0&type=all'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com/v/popular/rank/all',
    # 这一步是关键！把身份证带上
    'Cookie': MY_COOKIE
}


def get_content(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"请求被拦截，状态码: {response.status_code}")
            return None
    except Exception as e:
        print(f"请求网络出错: {e}")
        return None


def parse_data(json_data):
    if not json_data:
        return []

    # --- 调试核心：看看B站到底回了什么 ---
    # 如果 code 不是 0，说明B站拒绝了请求
    code = json_data.get('code')
    if code != 0:
        print(f"【失败】B站返回错误信息: {json_data.get('message')}")
        print(f"错误代码: {code}")
        print("请检查 Cookie 是否过期或复制完整。")
        return []
    # -----------------------------------

    try:
        video_list = json_data['data']['list']
    except KeyError:
        print("【失败】找不到视频列表，数据结构可能变了")
        return []

    data_list = []
    for video in video_list:
        # 为了防止某个字段没有报错，用 .get() 方法更安全
        title = video.get('title', '无标题')
        owner = video.get('owner', {})
        author = owner.get('name', '未知作者')
        stat = video.get('stat', {})
        play_count = stat.get('view', 0)
        score = video.get('score', 0)

        # 链接处理
        bvid = video.get('bvid')
        link = f"https://www.bilibili.com/video/{bvid}" if bvid else ""

        item = {
            '标题': title,
            'UP主': author,
            '播放量': play_count,
            '综合得分': score,
            '链接': link
        }
        data_list.append(item)

    return data_list


def main():
    print("正在尝试带着 Cookie 爬取...")

    # 检查用户有没有填 Cookie
    if '这里粘贴' in MY_COOKIE:
        print("【错误】你还没填 Cookie！请去代码开头把 MY_COOKIE 替换成你浏览器里的 Cookie。")
        return

    content = get_content(TARGET_URL)
    data = parse_data(content)

    if data:
        print(f"成功获取 {len(data)} 条数据！")
        print("前3条预览：")
        for i in data[:3]:
            print(i)

        df = pd.DataFrame(data)
        df.to_csv('bilibili_rank_cookie.csv', index=False, encoding='utf-8-sig')
        print("文件已保存为 bilibili_rank_cookie.csv")
    else:
        print("本次爬取失败，请根据上方报错调整。")


if __name__ == '__main__':
    main()