import scrapy

class ratopatiSpider(scrapy.Spider):
	name = "ratopati"
	allowed_domains = ["ratopati.com"]

	start_urls = [
		"http://www.ratopati.com/category/news"
		]

	def parse(self, response):
		links = response.xpath('.//a/@href').extract()
		# Making sure there is the "next" button so that we don't loop endlessly
		next_ = response.xpath('.//a[@class="next page-numbers"]')

		if next_:
			for link in links:
				link = response.urljoin(link)
				
				yield scrapy.Request(link, callback=self.parse_article)

			next_page = response.xpath('.//a[@class="next page-numbers"]/@href')[0].extract()
			next_page = response.urljoin(next_page)

			yield scrapy.Request(next_page, dont_filter=True)
		else:
			print("EOW")	# End of Website

	def parse_article(self, response):
		# To make sure we are on the right type of page, i.e. an article
		authors = response.xpath('.//span[@class="meta-item"]')

		if authors:
			headline = response.xpath('.//h1[@itemprop="headline"]/text()')[0].extract()
			article = ''.join(response.xpath('.//div[@class="ratopati-table-border-layout"]//p/text()').extract())

			if article:
				yield {"title": headline, "body": article}