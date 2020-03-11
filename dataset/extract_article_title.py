import re, os, sys
import pandas as pd
from tqdm import tqdm

def mkdir(path):
	if not os.path.exists(path): os.makedirs(path)

def write_lines(lines, filename):
	with open(filename, "w", encoding="utf-8") as f:
		for line in lines:
			f.write(line + "\n")

newspaper = sys.argv[1]	# Newspaper name, folder name

root = "extract"

place_delim = None

try:
	if sys.argv[2]:
		if sys.argv[2] == "purnabiram":
			place_delim = "।"
		else:
			place_delim = sys.argv[2] 
except:
	pass

print(f"special delimiter is {place_delim}")

mkdir(root)
mkdir(os.path.join(root, newspaper))

sections = os.listdir(newspaper)

for filename in tqdm(sections):
	data = pd.read_json(os.path.join(newspaper, filename))

	titles = data["title"]
	articles = data["body"]

	def strip(text):
		return re.sub("[\n\r]", "", text)

	def shorten(text):
		if place_delim: 
			tx = text.split(place_delim, 1)

			if len(tx[0]) < 27: # Make sure the bit we cut out is actually just the place name
				text = tx[-1]

		text = re.sub("[\n\r]", "", text)

		lines = text.split("।")
		lines = [line.strip() + "।" for line in lines if line.strip()]

		stopat = 3

		sent = " ".join(lines[:stopat])
		return sent

	articles = articles.apply(shorten)
	titles = titles.apply(strip)

	section = filename.split(".")[0]

	mkdir(os.path.join(root, newspaper, section))

	write_lines(titles.tolist(), os.path.join(root, newspaper, section, "title.txt"))
	write_lines(articles.tolist(), os.path.join(root, newspaper, section, "article.txt"))