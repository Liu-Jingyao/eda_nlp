# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *
import pandas as pd
from gensim import corpora
from gensim import models

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# arguments to be parsed from command line
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float,
                help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
ap.add_argument("--tfidf", required=False, type=int, help="use tfidf to maintain keywords")
args = ap.parse_args()

# the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join

    output = join(dirname(args.input), 'eda_' + basename(args.input))

# number of augmented sentences to generate per original sentence
num_aug = 9  # default
if args.num_aug:
    num_aug = args.num_aug

# how much to replace each word by synonyms
alpha_sr = 0.1  # default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

# how much to insert new words that are synonyms
alpha_ri = 0.1  # default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

# how much to swap words
alpha_rs = 0.1  # default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

# how much to delete words
alpha_rd = 0.1  # default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
    ap.error('At least one alpha should be greater than zero')

tfidf = 0
if args.tfidf is not None:
    tfidf = args.tfidf


def get_only_chars(line):
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


def get_word_tfidf_list(sentences):
    # vectorizer = TfidfVectorizer(stop_words=None, token_pattern=r"(?u)\b[A-Za-z]+\b")
    # vectors = vectorizer.fit_transform([get_only_chars(line) for line in lines])
    # feature_names = vectorizer.get_feature_names()
    # dense = vectors.todense()
    # denselist = dense.tolist()
    # df = pd.DataFrame(denselist, columns=feature_names)
    #
    # word_tfidf_list = df.to_dict('records')
    # Create a set of frequent words
    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in get_only_chars(document).lower().split()]for document in sentences]
    # Count word frequencies
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    # Only keep words that appear more than once
    processed_corpus = texts
    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    # train the model
    tfidf = models.TfidfModel(bow_corpus)
    # transform the "system minors" string
    word_tfidf_list = []
    id2token = {item[1]: item[0] for item in dictionary.token2id.items()}
    for text in processed_corpus:
        word_tfidf = {id2token[_[0]]: _[1] for _ in tfidf[dictionary.doc2bow(text)]}
        word_tfidf_list.append(word_tfidf)
    word_tfidf_list = [[word_tfidf_new for word_tfidf_new in sorted(word_tfidf.items(), key=lambda kv: kv[1]) if
                        word_tfidf_new[1]] for word_tfidf in word_tfidf_list]
    return word_tfidf_list


# generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    writer = open(output_file, 'w')
    lines = open(train_orig, 'r', encoding="utf-8").readlines()
    lines_data = [line[:-1].split('\t') for line in lines]
    sentences = [data[1] for data in lines_data]
    word_tfidf_list = get_word_tfidf_list(sentences)
    for i, data in enumerate(lines_data):
        label = data[0]
        sentence = data[1]
        if tfidf:
            aug_sentences = tfidf_eda(sentence, word_tfidf_list[i], alpha_sr=alpha_sr, alpha_ri=alpha_ri,
                                      alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        else:
            aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                                num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(label + "\t" + aug_sentence + '\n')

    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(
        num_aug))


# main function
if __name__ == "__main__":
    # generate augmented sentences and output into a new file
    gen_eda(args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd,
            num_aug=num_aug)
