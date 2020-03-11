import pandas as pd
import sys

filename = sys.argv[1]

a = pd.read_json(f"{filename}.json")

total = len(a)

half = total//2

b = a.iloc[:half, :]

with open(f"{filename}-1.json", 'w', encoding='utf-8') as file:
	b.to_json(file, orient="records", force_ascii=False)

b = a.iloc[half:, :]

with open(f"{filename}-2.json", 'w', encoding='utf-8') as file:
	b.to_json(file, orient="records", force_ascii=False)