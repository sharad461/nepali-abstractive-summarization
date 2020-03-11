import scrapy

class setopatiSpider(scrapy.Spider):
	name = "setopati"
	allowed_domains = ["setopati.com"]

	start_urls = ["https://www.setopati.com/art"]

	def parse(self, response):
		links = response.xpath('.//a/@href').extract()
		# Making sure there is the "next" button so that we don't loop endlessly
		next_ = response.xpath('.//div[@class="pagination"]/a[@rel="next"]')

		if next_:
			for link in links:
				link = response.urljoin(link)
				
				yield scrapy.Request(link, callback=self.parse_article)

			next_page = response.xpath(".//div[@class='pagination']/a[@rel='next']/@href")[0].extract()
			next_page = response.urljoin(next_page)

			yield scrapy.Request(next_page, dont_filter=True)
		else:
			print("EOW")	# End of Website

	def parse_article(self, response):
		# To make sure we are on the right type of page, i.e. an article
		authors = response.xpath('.//div[@class="row authors-box"]')

		if authors:
			headline = response.xpath('.//span[@class="news-big-title"]/text()')[0].extract()
			article = ''.join(response.xpath('.//div[@class="editor-box"]//text()').extract())

			if article:
				yield {"title": headline, "body": article}