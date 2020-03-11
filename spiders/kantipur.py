import scrapy

class kantipurSpider(scrapy.Spider):
	name = "kantipur"
	allowed_domains = ["ekantipur.com"]

	category_list = ["समाचार", "अर्थ / वाणिज्य", "खेलकुद", "उपत्यका", "विश्व", "मनोरञ्जन", "प्रवास", "विज्ञान र प्रविधि", "स्वास्थ्य", "कला"]

	start_urls = [
		"https://ekantipur.com/"
		]

	def parse(self, response):
		links = response.xpath('.//a/@href').extract()

		authors = response.xpath('.//span[@class="author"]')

		if authors:
			headline = response.xpath('.//div[@class="article-header"]/h1/text()')[0].extract()
			cat = response.xpath('.//div[@class="cat_name"]//text()')[0].extract()
			a = response.xpath(".//div[@class='description']//text()[not(ancestor::script)]").extract()

			b = a.index("Share on Facebook")

			article = "".join(a[:b])
			if article and cat in self.category_list:
				yield {"title": headline, "body": article}

		for link in links:
			link = response.urljoin(link)
			
			yield scrapy.Request(link, callback=self.parse)