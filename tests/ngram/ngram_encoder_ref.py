"""Reference impl of ngram extraction algorithm."""

from collections import defaultdict
import re
import math


class FindNgrams:
    def __init__(self, min_count=0, min_pmi=0, language="en"):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.words = defaultdict(int)
        self.ngrams, self.pairs = defaultdict(int), defaultdict(int)
        self.total = 0.0
        self.language = language

    def text_filter(self, sentence):
        cleaned_text = []
        index = 0
        for i, w in enumerate(sentence):
            if re.match("[^\u0600-\u06ff\u0750-\u077f\u4e00-\u9fa50-9a-zA-Z]+", w):
                if i > index:
                    cleaned_text.append([w.lower() for w in sentence[index:i]])
                index = 1 + i
        if index < len(sentence):
            cleaned_text.append([w.lower() for w in sentence[index:]])
        return cleaned_text

    def count_ngram(self, texts, n):
        self.ngrams = defaultdict(int)
        for sentence in texts:
            sub_sentence = sentence.split()
            for i in range(n):
                n_len = i + 1
                for j in range(len(sub_sentence) - i):
                    ngram = tuple([w for w in sub_sentence[j : j + n_len]])
                    self.ngrams[ngram] += 1
        self.ngrams = {i: j for i, j in self.ngrams.items() if j > self.min_count}

    def find_ngrams_pmi(self, token_seq_vec: list[list[int]], n, freq_threshold):
        # for sentence in texts:
        for token_seq in token_seq_vec:
            self.words[token_seq[0]] += 1
            for i in range(len(token_seq) - 1):
                self.words[token_seq[i + 1]] += 1
                self.pairs[(token_seq[i], token_seq[i + 1])] += 1
                self.total += 1
        self.words = {i: j for i, j in self.words.items() if j > self.min_count}
        self.pairs = {i: j for i, j in self.pairs.items() if j > self.min_count}

        min_mi = math.inf
        max_mi = -math.inf

        self.strong_segments = set()
        for i, j in self.pairs.items():
            if i[0] in self.words and i[1] in self.words:
                mi = math.log2(self.total * j / (self.words[i[0]] * self.words[i[1]]))

                if mi > max_mi:
                    max_mi = mi
                if mi < min_mi:
                    min_mi = mi
                if mi >= self.min_pmi:
                    self.strong_segments.add(i)

        self.ngrams = defaultdict(int)
        # for sentence in texts:
        for token_seq in token_seq_vec:
            # sub_sentence = sentence.split()
            s = [token_seq[0]]
            for i in range(len(token_seq) - 1):
                if (token_seq[i], token_seq[i + 1]) in self.strong_segments:
                    s.append(token_seq[i + 1])
                else:
                    self.ngrams[tuple(s)] += 1
                    s = [token_seq[i + 1]]
        # print("[debug]: ngram length")
        # for k,v in self.ngrams.items():
        #     if len(k) >=10:
        #         print(k)

        self.ngrams = {i: j for i, j in self.ngrams.items() if j > self.min_count and len(i) <= n}

        self.renew_ngram_by_freq(token_seq_vec, freq_threshold, n)

    def renew_ngram_by_freq(self, token_seq_vec: list[list[int]], min_feq, ngram_len=10):
        new_ngram2count = {}

        # for sentence in all_sentences:
        for token_seq in token_seq_vec:
            # sentence = sentence.split()

            # sen = sentence
            for i in range(len(token_seq)):
                for n in range(1, ngram_len + 1):
                    if i + n > len(token_seq):
                        break
                    n_gram = tuple(token_seq[i : i + n])
                    if n_gram not in self.ngrams:
                        continue
                    if n_gram not in new_ngram2count:
                        new_ngram2count[n_gram] = 1
                    else:
                        new_ngram2count[n_gram] += 1
        self.ngrams = {gram: c for gram, c in new_ngram2count.items() if c > min_feq}
