import numpy as np
import os
import re
import string
import matplotlib.pyplot as plt
import time

base_address_en = 'data/train/en/'
base_address_es = 'data/train/es/'

base_address_en_test = 'data/test/en/'
base_address_es_test = 'data/test/es/'

en_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z', '$']
es_letters = ['z', 'h', 'l', 'b', 'a', 'ä', 'à', 'o', 'i', 'e', 'k', 'é', 'w', 'á', 'ô', 'u', 'æ', 'ç', 'ë', 'f', 'ñ',
              'c', 'º', '$', 'p', 'v', 's', 'ü', 'ó', 'q', 'n', 'y', 'ù', 'm', 'd', 'ú', 'ï', 'x', 'í', 'g', 'è', 't',
              'r', 'j']

en_corpus = ''
es_corpus = ''


def normalize_text(text, size=10000):
    corpus = text.lower()
    words = re.split('\W+', ''.join(i for i in corpus if not i.isdigit()).translate(
        str.maketrans('', '', string.punctuation)))
    words = words[:size]
    nens = '$'.join(words)
    nens = '$' + nens

    return nens


def calc_prob(text, model, count, context_size=10000):
    ntext = normalize_text(text, size=context_size)
    p = 0

    for i in range(1, len(ntext) - 1):
        if ntext[i] != '$':
            if ntext[i] in model.keys():
                if ntext[i - 1] in model[ntext[i]].keys():
                    if model[ntext[i]][ntext[i - 1]] == 0:
                        p += np.log((model[ntext[i]][ntext[i - 1]] + 1) / (count[ntext[i - 1]] + 26))
                    else:
                        p += np.log(model[ntext[i]][ntext[i - 1]] / count[ntext[i - 1]])
    return p


def eval_model(test_dir):
    en_acc = []
    es_acc = []
    t = len(os.listdir(test_dir))
    for i in range(100, 11000, 200):
        enc = 0
        for file_name in os.listdir(base_address_en_test):
            content = ''
            for line in open(base_address_en_test + file_name):
                content += line + ' '

            p = calc_prob(content, en_table, count_table_en, context_size=i)
            p1 = calc_prob(content, es_table, count_table_es, context_size=i)

            if p > p1:
                enc += 1
        en_acc.append((enc / t) * 100)

        esc = 0
        for file_name in os.listdir(base_address_es_test):
            content = ''
            for line in open(base_address_es_test + file_name):
                content += line + ' '

            p = calc_prob(content, en_table, count_table_en, context_size=i)
            p1 = calc_prob(content, es_table, count_table_es, context_size=i)

            if p < p1:
                esc += 1

        es_acc.append((esc / t) * 100)

    plt.plot(range(100, 11000, 200), en_acc)
    plt.plot(range(100, 11000, 200), es_acc)
    plt.legend(['EN', 'ES'])
    plt.show()



for line in open(base_address_en + 'all_en.txt',encoding="utf8"):
    en_corpus += line + ' '

for line in open(base_address_es + 'all_es.txt',encoding="utf8"):
    es_corpus += line + ' '

ngram_en_corpus = normalize_text(en_corpus)
ngram_es_corpus = normalize_text(es_corpus)

en_table = {letter: {letter: 0 for letter in en_letters} for letter in en_letters}
es_table = {letter: {letter: 0 for letter in es_letters} for letter in es_letters}

for i in range(1, len(ngram_en_corpus) - 1):
    en_table[ngram_en_corpus[i]][ngram_en_corpus[i - 1]] += 1

for i in range(1, len(ngram_es_corpus) - 1):
    es_table[ngram_es_corpus[i]][ngram_es_corpus[i - 1]] += 1

count_table_en = {letter: sum(en_table[letter].values()) for letter in en_letters}
count_table_es = {letter: sum(es_table[letter].values()) for letter in es_letters}

# calculate probability
en_sample_num = len(os.listdir(base_address_en_test))
es_sample_num = len(os.listdir(base_address_es_test))

for file_name in os.listdir(base_address_en_test):
    content = ''
    for line in open(base_address_en_test + file_name,encoding="utf8"):
        content += line + ' '

    p = calc_prob(content, en_table, count_table_en)
    p1 = calc_prob(content, es_table, count_table_es)

    if p > p1:
        print(base_address_en_test + file_name, 'EN')
    else:
        print(base_address_en_test + file_name, 'ES')

for file_name in os.listdir(base_address_es_test):
    content = ''
    for line in open(base_address_es_test + file_name,encoding="utf8"):
        content += line + ' '

    p = calc_prob(content, en_table, count_table_en)
    p1 = calc_prob(content, es_table, count_table_es)

    if p < p1:
        print(base_address_es_test + file_name, 'ES')
    else:
        print(base_address_es_test + file_name, 'EN')
