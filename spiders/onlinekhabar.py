import scrapy

class Spider(scrapy.Spider):
	name = "onlinekhabar"
	allowed_domains = ["onlinekhabar.com"]

	start_urls = [
		"https://www.onlinekhabar.com/content/news"
	]

	def parse(self, response):
		links = response.xpath('.//a/@href').extract()
		next_ = response.xpath('.//div[@class="paginate-links"]/a[@class="next page-numbers"]')

		if next_:
			for link in links:
				link = response.urljoin(link)
				
				yield scrapy.Request(link, callback=self.parse_article)

			next_page = response.xpath(".//div[@class='paginate-links']/a[@class='next page-numbers']/@href")[0].extract()
			next_page = response.urljoin(next_page)

			yield scrapy.Request(next_page, dont_filter=True)
		else:
			print("EOW")	# End of Website

	def parse_article(self, response):
		true = response.xpath('.//h2[@class="mb-0"]/text()')

		if true:
			headline = response.xpath('.//h2[@class="mb-0"]/text()')[0].extract()
			article = ''.join(response.xpath('.//div[@class="col colspan3 main__read--content ok18-single-post-content-wrap"]//text()').extract())

			if article:
				yield {"title": headline, "body": article}