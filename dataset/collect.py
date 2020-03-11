import os
import sys

def write_lines(lines, filename):
	with open(filename, "w", encoding="utf-8") as f:
		for line in lines:
			f.write(line + "\n")

# root = "data"
root = sys.argv[1]

newspapers = os.listdir(root)

pairs = []

for newspaper in newspapers:
	sections = os.listdir(os.path.join(root, newspaper))

	for section in sections:
		a_ = open(os.path.join(root, newspaper, section, "article.txt"), "r", encoding="utf-8").read().split("\n")
		t_ = open(os.path.join(root, newspaper, section, "title.txt"), "r", encoding="utf-8").read().split("\n")

		assert len(a_) == len(t_), f"article and title files do not have equal lines: {newspaper}/{section}"

		pairs += list(zip(a_, t_))

pairs_ = set(pairs)

print(f"{len(pairs)} vs {len(pairs_)}")

arts, titles = zip(*pairs_)

write_lines(arts, "article.txt")
write_lines(titles, "title.txt")
