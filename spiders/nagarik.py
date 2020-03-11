import scrapy

class nagarikSpider(scrapy.Spider):
	name = "nagarik"
	allowed_domains = ["nagariknews.nagariknetwork.com"]

	start_urls = [
		"https://nagariknews.nagariknetwork.com/category/21"
		]

	def parse(self, response):
		links = response.xpath('.//a/@href').extract()
		next_ = response.xpath('.//a[@rel="next"]')

		if next_:
			for link in links:
				link = response.urljoin(link)
				
				yield scrapy.Request(link, callback=self.parse_article)

			next_page = response.xpath('.//a[@rel="next"]/@href')[0].extract()
			next_page = response.urljoin(next_page)

			yield scrapy.Request(next_page, dont_filter=True)
		else:
			print("EOW")	# End of Website

	def parse_article(self, response):
		authors = response.xpath('.//span[@class="author"]')

		if authors:
			headline = response.xpath('.//div[@class="col-sm-12"]/h1/text()')[0].extract()
			article = ''.join(response.xpath('.//div[@id="newsContent"]//p/text()').extract())

			if article:
				yield {"title": headline, "body": article}