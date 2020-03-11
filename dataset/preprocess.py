import sys
import re
from indicnlp.tokenize import indic_tokenize
from random import shuffle
from tqdm import tqdm

def write_lines(lines, filename):
	with open(filename, "w", encoding="utf-8") as f:
		for line in lines:
			f.write(line + "\n")

def process(filename, maxlen=250):
	lines = open(filename, "r", encoding="utf-8")#.read().split("\n")

	new = []
	word_counts = []

	# Pre-tokenize replacements
	regs = {"[‘’]":"'", '[“”]':'"', "[\s]+":" "}
	for line in tqdm(lines):
		line = line.strip()
		for reg in regs:
			line = re.sub(reg, regs[reg], line)
		new.append(line)

	lines.close()

	print(f"finished pre-processing {filename}")

	tokenized = []

	# Add a post-tokenize replacements section if needed
	reg = "[०-९]"
	for line in new:
		tokens = indic_tokenize.trivial_tokenize(line)
		if len(tokens) < maxlen:
			tok = " ".join(tokens)
			tokenized.append(re.sub(reg, "#", tok))
			word_counts.append(len(tokens))
		else:
			tokenized.append("")	
			# A quicker way would be load in (title, article) pairs into this function 
			# and skip a pair entirely whenever length > maxlen for articles

	print(f"finished tokenizing and post-processing {filename}")

	sort = sorted(word_counts, reverse=True)
	print(f"the 10 longest sequence lengths in {filename} are {sort[:10]}")

	print(f"average length: {sum(word_counts)/len(word_counts)}")

	return tokenized

articles, titles = sys.argv[1], sys.argv[2]
maxlen1, maxlen2 = int(sys.argv[3]), int(sys.argv[4])

a_, t_ = process(articles, maxlen=maxlen1), process(titles, maxlen=maxlen2)

pairs = list(zip(a_, t_))

shuffle(pairs)

final = []

for (a, t) in pairs:
	if a and t:
		final.append((a, t))

train = int(0.96 * len(final))

print(f"{train} train samples, {len(final) - train} valid samples")

files = [articles, titles]

for i, x in enumerate(zip(*final)):
	write_lines(x[:train], "train." + files[i])
	write_lines(x[train:], "valid." + files[i])