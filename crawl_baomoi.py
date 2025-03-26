import requests, time, psycopg2, json, traceback
from dateutil import parser
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",  # e.g., "localhost" or an IP address
    "port": "5432",  # Default PostgreSQL port
}
conn = psycopg2.connect(**DB_PARAMS)

def access(url: str):
    while (True):
        try:
            res = requests.get(url, timeout=5)
            return res.text
        except Exception as e:
            time.sleep(1)
            continue

def run(target_url: str, cur, conn):
    print(target_url)
    soup = BeautifulSoup(access(url=target_url), 'html.parser')
    a_tags = soup.select("h3 > a")
    if len(a_tags) == 0:
        return
    
    urls = [f'https://baomoi.com{a.get("href")}' for a in a_tags]

    insert_query = "INSERT INTO news_urls (url) VALUES (%s) ON CONFLICT (url) DO NOTHING;"

    # Execute the query for each item in the list
    cur.executemany(insert_query, [(item,) for item in urls])

    # Commit the transaction
    conn.commit()

def run_json(target_url: str, cur, conn):
    print(target_url)
    res_json = json.loads(access(url=target_url))
    urls = [f'https://baomoi.com{item.get("url")}' for item in res_json['data']['items']]

    insert_query = "INSERT INTO news_urls (url) VALUES (%s) ON CONFLICT (url) DO NOTHING;"

    # Execute the query for each item in the list
    cur.executemany(insert_query, [(item,) for item in urls])

    # Commit the transaction
    conn.commit()

access_by_json = [
    "https://bs-api.baomoi.com/v1/content/get/list-by-type?listType=3&listId=55&page={}"
]

categories = [
    'https://baomoi.com/the-gioi/trang{}.epi',
    'https://baomoi.com/xa-hoi/trang{}.epi',
    'https://baomoi.com/thoi-su/trang{}.epi',
    'https://baomoi.com/giao-thong/trang{}.epi',
    'https://baomoi.com/moi-truong-khi-hau/trang{}.epi',
    'https://baomoi.com/van-hoa/trang{}.epi',
    'https://baomoi.com/nghe-thuat/trang{}.epi',
    'https://baomoi.com/am-thuc/trang{}.epi',
    'https://baomoi.com/du-lich/trang{}.epi',
    'https://baomoi.com/kinh-te/trang{}.epi',
    'https://baomoi.com/lao-dong-viec-lam/trang{}.epi',
    'https://baomoi.com/tai-chinh/trang{}.epi',
    'https://baomoi.com/chung-khoan/trang{}.epi',
    'https://baomoi.com/kinh-doanh/trang{}.epi',
    'https://baomoi.com/the-thao/trang{}.epi',
    'https://baomoi.com/bong-da-quoc-te/trang{}.epi',
    'https://baomoi.com/bong-da-viet-nam/trang{}.epi',
    'https://baomoi.com/quan-vot/trang{}.epi',
    'https://baomoi.com/giai-tri/trang{}.epi',
    'https://baomoi.com/am-nhac/trang{}.epi',
    'https://baomoi.com/thoi-trang/trang{}.epi',
    'https://baomoi.com/dien-anh-truyen-hinh/trang{}.epi',
    'https://baomoi.com/phap-luat/trang{}.epi',
    'https://baomoi.com/an-ninh-trat-tu/trang{}.epi',
    'https://baomoi.com/hinh-su-dan-su/trang{}.epi'
]

def extract_title(url: str, cur, conn):
    print(url)
    try:
        soup = BeautifulSoup(access(url=url), 'html.parser')
        # print(soup)
        title = soup.select_one("title").text.strip()

        insert_query = """INSERT INTO news_urls (url, title) VALUES (%s, %s)
        ON CONFLICT (url) DO UPDATE SET
        title = EXCLUDED.title
        """
        cur.execute(insert_query, (url, title))
        conn.commit()
    except Exception as e:
        print(soup.select_one("title"))
        print(traceback.format_exc())

def extract_content(url: str, cur, conn):
    print(url)
    try:
        soup = BeautifulSoup(access(url=url), 'html.parser')
        script = soup.select_one("#__NEXT_DATA__")
        script_json = json.loads(script.text)

        texts = [t['content'] for t in script_json['props']['pageProps']['resp']['data']['content']['bodys'] if t['type'] == 'text']
        full_text = ""
        for t in texts:
            pt = t
            if len(t) == 0:
                continue
            if t[-1] != ".":
                pt += "."
            # print(t)
            full_text += (BeautifulSoup(pt, "html.parser").text.replace("\n", " ").strip() + " ")

        published_time = script_json['props']['pageProps']['resp']['data']['head']['meta']['articlePublishedTime']
        dt = parser.isoparse(published_time).replace(tzinfo=None)
        # print(dt)

        insert_query = """INSERT INTO news_urls (url, published_time, text) VALUES (%s, %s, %s)
        ON CONFLICT (url) DO UPDATE SET
        published_time = EXCLUDED.published_time,
        text = EXCLUDED.text
        """
        cur.execute(insert_query, (url, dt, full_text))
        conn.commit()
    except Exception as e:
        print(script)
        print(traceback.format_exc())

cur = conn.cursor()
with ThreadPoolExecutor(max_workers=1) as pool:
    # for cate in categories:
    cur.execute("""SELECT url FROM news_urls
WHERE
    title IS NULL AND
    text IS NOT NULL AND
    published_time IS NOT NULL AND
    published_time >= '2025-03-07'""")
    rows = cur.fetchall()

    for r in rows:
        pool.submit(extract_title, r[0], cur, conn)
        # break
    
cur.close()