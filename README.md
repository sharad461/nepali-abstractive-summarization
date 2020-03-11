# Nepali Abstractive Summarization

This project is focused on the tasks of headline generation and summary generation in the Nepali language. We prepare our own datasets for both tasks. The scripts that we prepared while working on the project are provided here.

The dataset is collected by scraping online Nepali news portals. The headline generation dataset has 280k article-title pairs and the summary generation dataset is still very much in progress.

Currently the system is based on the Pointer-Generator Network discussed in _Get To The Point: Summarization with Pointer-Generator Networks_ (See et. al. (2017)) together with the intra attention and reinforcement learning techniques discussed in _A Deep Reinforced Model For Abstractive Summarization_ (Paulus et. al. (2017)). We currently use the implementation of this ensemble network as found [here](https://github.com/rohithreddy024/Text-Summarizer-Pytorch), with plans to implement a Transformer network for the same task. This new system is incomplete.

## Results

The ROUGE-1, ROUGE-2 and ROUGE-L scores for the headline generation task are 35.71, 18.53 and 32.89 respectively, far from 40+ ROUGE-1 and ROUGE-L for the same in the English language. The go-to headline generation dataset for English is the English Gigaword which has about 4 million article-title pairs.

## Usage
Web scraping was done with the Scrapy Python package. The spiders are inside the `spiders` folder.

A comprehensive guide to creating the dataset using these scripts will be provided soon. A link to the dataset will also be added soon.

## Sample Summaries

### Headline generation

| Article  | काठमाडौं उपत्यकासहित आसपासका क्षेत्रमा पनि शुक्रबार बिहानै भीषण हावाहुरी चलेको छ । हावाहुरीसँगै वर्षा पनि भएको छ । बिहान साढे # बजेबाट सुरु भीषण हावाहुरी र वर्षा एक घण्टाभन्दा बढी सयमयसम्म आएको थियो । |
| Reference  | काठमाडौंमा पनि भीषण हावाहुरी |
| System  | काठमाडौंमा भीषण हावाहुरी , हावाहुरीसँगै वर्षा |

## Citation
If you use any part of this project in your work, please cite:

```bibtex
@techreport{nepali-abstractive-summ-2020,
  title={Abstractive Summarization},
  author={Bhandari, Pragya and Duwal, Sharad and Sah, Anjelika and Subedi, Subarna},
  institution={Kathmandu University},
  year={2020}
}
```

For the fulfillment of requirements for Machine Learning (COMP 484) course at Kathmandu University. Mar 2020.