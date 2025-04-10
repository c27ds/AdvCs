import scrapy
import requests
from requests_ip_rotator import ApiGateway
from urllib.parse import urlparse
from scrapy.http import HtmlResponse

class RotatingIPMiddleware:
    """
    Downloader middleware that rotates IP addresses using requests_ip_rotator.
    It creates an ApiGateway session for each domain and uses that session
    to fetch the page, wrapping the result as an HtmlResponse.
    """
    gateways = {}

    def process_request(self, request, spider):
        domain = urlparse(request.url).netloc
        # Initialize a gateway for the domain if not already done.
        if domain not in self.gateways:
            gateway = ApiGateway("https://" + domain)
            gateway.start()
            self.gateways[domain] = gateway
        session = self.gateways[domain].get_session()
        try:
            # Fetch the URL using the rotating IP session.
            r = session.get(request.url, timeout=10)
            encoding = r.encoding if r.encoding else 'utf-8'
            return HtmlResponse(
                url=request.url,
                body=r.content,
                encoding=encoding,
                request=request
            )
        except Exception as e:
            spider.logger.error("Error fetching %s: %s", request.url, e)
            # Let Scrapy handle the request in case of error (or you could retry).
            return None

    def spider_closed(self, spider):
        # Shutdown all gateways when the spider closes.
        for gateway in self.gateways.values():
            gateway.shutdown()

class SupplyChainSpider(scrapy.Spider):
    name = "supply_chain"
    
    allowed_domains = [
        "supplychaindive.com", 
        "freightwaves.com", 
        "logisticsmgmt.com", 
        "theloadstar.com",
        "census.gov",
        "supplychainbrain.com",
        "reuters.com",
        "bloomberg.com",
        "businesswire.com"
    ]
    
    start_urls = [
        "https://www.supplychaindive.com/",
        "https://www.freightwaves.com/news",
        "https://www.logisticsmgmt.com/",
        "https://theloadstar.com/",
        "https://www.census.gov/econ/currentdata/",
        "https://www.supplychainbrain.com/",
        "https://www.reuters.com/business/",
        "https://www.bloomberg.com/markets",
        "https://www.businesswire.com/news/home/"
    ]

    custom_settings = {
        'FEED_FORMAT': 'json',
        'FEED_URI': 'supply_chain_data.json',
        'LOG_LEVEL': 'INFO',
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 2,
        'AUTOTHROTTLE_MAX_DELAY': 10,
        # Register the custom rotating IP middleware.
        'DOWNLOADER_MIDDLEWARES': {
            '__main__.RotatingIPMiddleware': 543,
        }
    }

    def parse(self, response):
        url = response.url
        if "supplychaindive.com" in url:
            yield from self.parse_supplychaindive(response)
        elif "freightwaves.com" in url:
            yield from self.parse_freightwaves(response)
        elif "logisticsmgmt.com" in url:
            yield from self.parse_logisticsmgmt(response)
        elif "theloadstar.com" in url:
            yield from self.parse_loadstar(response)
        elif "census.gov" in url or "bts.gov" in url:
            yield from self.parse_government(response)
        elif "supplychainbrain.com" in url:
            yield from self.parse_supplychainbrain(response)
        elif "reuters.com" in url:
            yield from self.parse_reuters(response)
        elif "bloomberg.com" in url:
            yield from self.parse_bloomberg(response)
        elif "businesswire.com" in url:
            yield from self.parse_businesswire(response)
        else:
            self.logger.info("No specific parser for: %s", url)

    def parse_supplychaindive(self, response):
        for article in response.css("div.news-feed__item"):
            title = article.css("h2 a::text").get(default="").strip()
            link = response.urljoin(article.css("h2 a::attr(href)").get())
            yield {'title': title, 'source': "Supply Chain Dive", 'link': link}
            yield response.follow(link, callback=self.parse_article)

    def parse_freightwaves(self, response):
        for article in response.css("div.td-module-container"):
            title = article.css("h3 a::text").get(default="").strip()
            link = response.urljoin(article.css("h3 a::attr(href)").get())
            yield {'title': title, 'source': "FreightWaves", 'link': link}
            yield response.follow(link, callback=self.parse_article)

    def parse_logisticsmgmt(self, response):
        for article in response.css("article"):
            title = article.css("h2 a::text").get(default="").strip()
            link = response.urljoin(article.css("h2 a::attr(href)").get())
            yield {'title': title, 'source': "Logistics Management", 'link': link}
            yield response.follow(link, callback=self.parse_article)

    def parse_loadstar(self, response):
        for article in response.css("article.post"):
            title = article.css("h2 a::text").get(default="").strip()
            link = response.urljoin(article.css("h2 a::attr(href)").get())
            yield {'title': title, 'source': "The Loadstar", 'link': link}
            yield response.follow(link, callback=self.parse_article)

    def parse_government(self, response):
        for report in response.css("li a"):
            title = report.css("::text").get(default="").strip()
            link = response.urljoin(report.css("::attr(href)").get())
            yield {'title': title, 'source': "Government Data", 'link': link}

    def parse_supplychainbrain(self, response):
        for article in response.css("div.article"):
            title = article.css("h2 a::text").get(default="").strip()
            link = response.urljoin(article.css("h2 a::attr(href)").get())
            yield {'title': title, 'source': "Supply Chain Brain", 'link': link}
            yield response.follow(link, callback=self.parse_article)

    def parse_reuters(self, response):
        for article in response.css("article"):
            title = article.css("h3::text").get(default="").strip()
            link = response.urljoin(article.css("a::attr(href)").get())
            yield {'title': title, 'source': "Reuters", 'link': link}
            yield response.follow(link, callback=self.parse_article)

    def parse_bloomberg(self, response):
        for article in response.css("div.story-package-module__story"):
            title = article.css("h3::text").get(default="").strip()
            link = response.urljoin(article.css("a::attr(href)").get())
            yield {'title': title, 'source': "Bloomberg", 'link': link}
            yield response.follow(link, callback=self.parse_article)

    def parse_businesswire(self, response):
        for article in response.css("div.article"):
            title = article.css("h3::text").get(default="").strip()
            link = response.urljoin(article.css("a::attr(href)").get())
            yield {'title': title, 'source': "BusinessWire", 'link': link}
            yield response.follow(link, callback=self.parse_article)

    def parse_article(self, response):
        """
        Extracts the full article details.
        """
        title = response.css("h1::text").get(default="").strip()
        paragraphs = response.css(
            "article p::text, div.entry-content p::text, section.article-content p::text, div.article-body p::text"
        ).getall()
        article_text = " ".join(p.strip() for p in paragraphs if p.strip())
        date = response.css("time::attr(datetime)").get() or response.css("time::text").get()
        author = (response.css("span.author-name::text").get(default="").strip() or
                  response.css("p.byline::text").get(default="").strip())
        yield {
            'url': response.url,
            'title': title,
            'date': date,
            'author': author,
            'content': article_text,
        }
