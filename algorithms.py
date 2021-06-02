import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk import word_tokenize, sent_tokenize
import numpy as np
import networkx as nx
import math
import spacy
import re


# feature möglichekeiten: tf, tf-isf, textrank, senposition, occurence of entities, amount of numerslas


def preprocessing_input_text(input_text):
    # create array [orignal sentence, clean sentence, score]
    sentences_array = []
    tokenized_sentences = sent_tokenize(input_text)

    for sentence_original in tokenized_sentences:
        # remove all special characters, puncutation, extra white spaces, make it lower, spolit in single tokens
        sentence_clean = re.sub(r'[^\w\s]', '', sentence_original.lower())
        sentence_clean = re.sub(r'_', '', sentence_clean)
        sentence_clean = re.sub('\s+', ' ', sentence_clean)
        sentence_tokens = word_tokenize(sentence_clean)
        # wtf, tf-isf, textrank, senlength, senpos, occur entities, amount numbers
        sentence_feature_vector = [0, 0, 0, 0, 0, 0, 0]
        sentences_array.append([sentence_original, sentence_tokens, sentence_feature_vector])
    return sentences_array


def weighted_term_frequency(sentences_array):
    # also called word probability
    #  Words with highest probability are assumed to represent the topic of the document and are included in the summary
    # weighted term frequency here defined
    # calc how often a term appears the whole text
    # weight this frequqncy by the total amount of words in the text
    # calc the weitghed tf for each sentence
    lt = nltk.WordNetLemmatizer()  # bring word to base form with lemmatizer
    frequency_dic = dict()
    for sentence in sentences_array:
        for word in sentence[1]:
            if word not in stopwords.words("english"):
                word = lt.lemmatize(word)
                if word not in frequency_dic:
                    frequency_dic[word] = 1
                else:
                    frequency_dic[word] += 1

    # calc score for each sentence
    total_words_in_text = 0
    for sentence in sentences_array:
        total_words_in_text += len(sentence[1])

    for sentence in sentences_array:
        for word in sentence[1]:
            word = lt.lemmatize(word)
            if word in frequency_dic.keys():
                term_frequency_word = frequency_dic[word] / total_words_in_text
                sentence[2][0] += term_frequency_word
    return sentences_array


def tf_score(word, sentence):
    lt = nltk.WordNetLemmatizer()  # bring word to base form with lemmatizer
    word_freq_in_sentence = 0
    lemmatized_words = [lt.lemmatize(word2) for word2 in sentence]
    for entry in lemmatized_words:
        if entry == word:
            word_freq_in_sentence += 1
    return word_freq_in_sentence / len(sentence)


def idf_score(word, sentences_array):
    lt = nltk.WordNetLemmatizer()
    # count in how many sentences a word occurs
    sentences_that_contain_word = 0
    for sentence in sentences_array:
        lemmatized_words = [lt.lemmatize(word) for word in sentence[1]]
        if word in lemmatized_words:
            sentences_that_contain_word += 1
    return math.log(len(sentences_array) / sentences_that_contain_word)


def tf_idf(sentences_array):
    # tf-isd term frequwncey inverse sentence frequqnce -> statt documents nun sentences
    # sentences seen as documents
    # calc the score of each sentence by summing up tfidf-values of the words in the sentence
    # calc tfidf of each word
    # calc the TF for each word of a sentence -> #word / #words in a sentence
    # calc the IDF for each word -> #sentences / #sentences that contain the word

    lt = nltk.WordNetLemmatizer()
    #calc socre for each sentence by summing the tf-idf values of the words in the sentence
    for sentence in sentences_array:
        score_of_sentence = 0
        for word in sentence[1]:
            if word not in stopwords.words("english"):
                word = lt.lemmatize(word)
                tf_score_word = tf_score(word, sentence[1])
                idf_score_word = idf_score(word, sentences_array)
                tf_idf_score_word = tf_score_word * idf_score_word
                score_of_sentence += tf_idf_score_word
        sentence[2][1] = score_of_sentence

    return sentences_array


def sentence_similarity_cosine(sentence1, sentence2):
    # count vector, für jeden satz gezählt wie häufig wort vorkommt
    # word2vec, glove, bert, elmo
    all_words = list(set(sentence1 + sentence2))
    vector_sent1 = [0] * len(all_words)
    vector_sent2 = [0] * len(all_words)

    # vector for first sentence
    for word in sentence1:
        if word not in stopwords.words("english"):
            vector_sent1[all_words.index(word)] += 1

    # vector for second sentence
    for word in sentence2:
        if word not in stopwords.words("english"):
            vector_sent2[all_words.index(word)] += 1

    cosine_score = 1 - cosine_distance(vector_sent1, vector_sent2)
    return cosine_score


def textrank(sentences_array):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences_array), len(sentences_array)))

    for i in range(len(sentences_array)):
        for j in range(len(sentences_array)):
            if i != j:  # ignore if both are same sentences
                similarity_matrix[i][j] = sentence_similarity_cosine(sentences_array[i][1], sentences_array[j][1])

    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    for key, value in scores.items():
        sentences_array[key][2][2] += value

    return sentences_array


def sentence_length(sentences_array):
    # sentence with length samller 4 are scored 0 as they wpnt cointain so much information
    # sents with length greater 20 are scored 0 as they might be long for a summary
    # other snetence are normalized with the longest sentence length / toal amount of words!
    # !!!!!!!!!! nocgmal anschauen
    longest_sentence_length = 0
    for sentence in sentences_array:
        if len(sentence[1]) > longest_sentence_length:
            longest_sentence_length = len(sentence[1])
    total_words = 0
    for sentence in sentences_array:
        total_words += len(sentence[1])
    for sentence in sentences_array:
        if len(sentence[1]) < 4 or len(sentence[1]) > 20:
            sentence[2][3] = 0
        else:
            sentence[2][3] = len(sentence[1]) / total_words
    return sentences_array


def sentence_position(sentences_array):
    # snetcen at beginning and end are scored with 1
    # sentences in between are are progressively decremented
    for index, sentence in enumerate(sentences_array):
        if index == 0 or index == len(sentences_array) - 1:
            sentence[2][4] = 1
        else:
            sentence[2][4] = (len(sentences_array)-index)/len(sentences_array)
    return sentences_array


def ooccurences_named_entities(sentences_array):
    # the amount of named entities is recvied and normalized by the lenght of the sentence
    sp = spacy.load('en_core_web_lg')
    for sentence in sentences_array:
        doc = sp(sentence[0])
        score = len(doc.ents) / len(sentence[1])
        sentence[2][5] = score
    return sentences_array


def amount_numerals(sentences_array):
    # the amount of numbers is recvied and normalized by the lenght of the sentence
    # Since gures are always crucial to presenting facts,
    # this feature gives importance to sentences having certain gures
    for sentence in sentences_array:
        num_array = re.findall(r'\d+', sentence[0])
        score = len(num_array) / len(sentence[1])
        sentence[2][6] = score
    return sentences_array


def total_feature_mix_score(sentences_array):
    # add up all featres score to get the total score for each sentence
    for sentence in sentences_array:
        total_score = sum(sentence[2])
        sentence[2] = total_score
    return sentences_array


def generate_summary(sorted_sentences_array, sentence_count_abstract):
    summary = []
    for index, sentence in enumerate(sorted_sentences_array[:int(sentence_count_abstract)]):
        ranked_sentence = " " + "Ranked " + str(index+1) + ": " + sentence[0]
        summary.append(ranked_sentence)
    return summary


def array_transfomer(summary_result):
    summary = []
    for index, sentence in enumerate(summary_result):
        ranked_sentence = " " + "Ranked " + str(index + 1) + ": " + str(sentence)
        summary.append(ranked_sentence)
    return summary
