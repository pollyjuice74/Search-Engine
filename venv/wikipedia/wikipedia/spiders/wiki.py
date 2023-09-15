import scrapy
from urllib.parse import urlparse, urljoin
from scrapy import signals


class WikiSpider(scrapy.Spider):
    name = "wiki"
    allowed_domains = ["en.wikipedia.org"]
    start_urls = ["https://en.wikipedia.org/wiki/Python_(programming_language)",
                  "https://en.wikipedia.org/wiki/Chemistry",
                  "https://en.wikipedia.org/wiki/Biology",
                  "https://en.wikipedia.org/wiki/Physics",
                  "https://en.wikipedia.org/wiki/Animal",
                  "https://en.wikipedia.org/wiki/Computer",
                  "https://en.wikipedia.org/wiki/Mathematics",
                  ]
    visited = set()
    max_pages = 3
    page_count = 0


    def parse(self, response):
        # Page url
        url = response.url

        # Check if page is visited
        if url in self.visited:
            self.logger.info(f"Already visited {url}")
            return
        else:
            self.visited.add(url)
            self.page_count += 1

        # Title, content
        title = response.xpath('//h1//span/text()').get()
        content = ''.join(txt for txt in response.xpath('//p//text()').getall())

        # Gets wiki paths and joins them to wiki url
        links = list()
        for href in response.css('a::attr(href)').getall():
            if href.startswith('/wiki/'):
                links.append(urljoin(url, href)) 

        yield {
            'title': title,
            'url': url,
            'content': content,
            'links': links,
        }

        # Check page count
        if url in self.start_urls:
            # Crawl links
            for link in links:
                yield scrapy.Request(link, callback=self.parse)