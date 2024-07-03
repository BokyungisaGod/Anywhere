import asyncio
from playwright.async_api import Playwright, async_playwright
from urllib.parse import quote_plus

async def get_blog_content(context, url):
    try:
        new_page = await context.new_page()
        await new_page.goto(url)
        await new_page.wait_for_selector("body", timeout=20000)

        # 페이지가 완전히 로드될 때까지 대기
        await asyncio.sleep(3)

        # iframe 내에서 내용을 찾는 경우 처리
        iframes = new_page.frames
        content = ""

        for frame in iframes:
            selectors = ["div.se-main-container", "div#postViewArea", "div.tt_article_useless_p_margin", "div.__se_component_area", "div.se-component-content", "div.se-module-text"]
            for selector in selectors:
                content_elements = await frame.query_selector_all(selector)
                if content_elements:
                    for element in content_elements:
                        content += await element.inner_text() + "\n"
                    break
            if content:
                break

        if not content:
            content = "[No content found]"

        await new_page.close()
        return content.strip()
    except Exception as e:
        print(f"Error fetching blog content from {url}: {e}")
        return "[Error fetching content]"

async def run(playwright: Playwright) -> None:
    search_query = "코이코이 맛집"
    encoded_query = quote_plus(search_query)

    # 블로그 탭에서 검색하도록 URL 수정
    search_url = f"https://search.naver.com/search.naver?where=blog&query={encoded_query}&sm=tab_jum&nso=so%3Ar%2Cp%3Aall"
    
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()

    await page.goto(search_url)
    await page.wait_for_selector("body")

    try:
        # 블로그 포스트 리스트 셀렉터
        articles = await page.query_selector_all("li.bx")
        print(f"Found {len(articles)} articles.")
    except Exception as e:
        print("Error selecting articles:", e)
        return

    for article in articles:
        try:
            # 블로그 링크 셀렉터
            link_element = await article.query_selector("a.title_link")
            if link_element:
                blog_url = await link_element.get_attribute("href")
                if "blog.naver.com" in blog_url:
                    print(f"Fetching content from: {blog_url}")

                    blog_content = await get_blog_content(context, blog_url)

                    print("=" * 70)
                    print(blog_content)
                else:
                    print(f"Skipped non-Naver blog URL: {blog_url}")
            else:
                print("No link element found for blog URL.")
        except Exception as e:
            print(f"Error fetching blog URL: {e}")
            continue

    await context.close()
    await browser.close()

async def main():
    async with async_playwright() as playwright:
        await run(playwright)

asyncio.run(main())