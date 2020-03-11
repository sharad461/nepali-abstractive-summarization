# Adapated from https://github.com/anukulu/Nepali-Text-Summarizer-Using-Scraping

from collections import Counter
import re, nltk
from tqdm import tqdm
import time
import sys

file = sys.argv[1]

articles = open(file, "r", encoding="utf-8")
summary_file = open("freq_summary.txt","w+", encoding="utf-8")

stopwords = set(nltk.corpus.stopwords.words("nepali"))

start = time.time()

for article in tqdm(articles):
	sentenceList = article.split('।')

	article = re.sub('(मा |को |ले |बाट |का |हरु |हरुसँग |सँग |लाई |हरू |हरूसँग |हरू )', ' ', article)
	article = article.replace('।', '')
	
	wordlist = article.split(" ") 
	article = [word for word in wordlist if word not in stopwords]

	word_frequencies = Counter(article)
	maxFrequency = max(word_frequencies.values())

	word_scores = {}
	for word in word_frequencies:  
		word_scores[word] = word_frequencies[word]/maxFrequency

	sentScores = {}
	for sent in sentenceList:
		split = sent.split(" ")
		for word in split:
			if word in word_scores.keys():
				if sent not in sentScores.keys():
					sentScores[sent] = word_scores[word]
				else:
					sentScores[sent] += word_scores[word]

	for sent in sentScores.keys():
		sentScores[sent] = sentScores[sent]/len(sent.split(" "))

	summary_sentences = sorted(sentScores, key=sentScores.get, reverse=True)[:5]

	summary = ""
	for sentence in sentenceList:
		if sentence in summary_sentences:
			summary += sentence.strip() + "। "

	summary_file.write(summary + "\n")

summary_file.close()

print(f"time taken: {time.time() - start}")