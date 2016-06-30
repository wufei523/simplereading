import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import sent_tokenize
from nltk.corpus import brown
import os
from os.path import join
import numpy as np
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from translate import functions as function
from translate import constructs
from gensim.models.word2vec import Word2Vec
import os
import json
import string
from os.path import join
import scipy.stats as ss
import math
from nltk import bigrams
from nltk import trigrams
from translate.alchemyapi import AlchemyAPI
alchemyapi = AlchemyAPI()



#model = Word2Vec.load_word2vec_format("/Users/luo/desktop/GoogleNews-vectors-negative300.bin.gz", binary = True)
#model = Word2Vec.load_word2vec_format("/home/twang/dataset/GoogleNews-vectors-negative300.bin.gz", binary = True)
path = os.getcwd()

#w2vFilePath = join(path, 'GoogleNews-vectors-negative300.bin.gz')
#model = Word2Vec.load_word2vec_format(w2vFilePath, binary = True)
w2vFilePath = join(path, 'glove_model.txt')
w2v_model = Word2Vec.load_word2vec_format(w2vFilePath, binary=False)

news_text = brown.words()
emma = nltk.corpus.gutenberg.words()
r = nltk.corpus.reuters.words()
#form corpus
corpus = emma + news_text
corpus += r
#lower case frequency dictionary based on brown + emma + reuters
fdist = nltk.FreqDist(w.lower() for w in corpus)

#language model
cfreq_corpus_2gram = nltk.ConditionalFreqDist(nltk.bigrams(w.lower() for w in corpus))
cprob_corpus_2gram = nltk.ConditionalProbDist(cfreq_corpus_2gram, nltk.MLEProbDist)
len_corpus = len(corpus)




def simplify(passage, min_frequent=100, min_frequent_diff = 1.2, min_similarity = 0.5, top_n_elements=20):


    sentences_tokenized_list = sent_tokenize(passage)
    simplified_passage = ''
    w2v_result_passage = []
    bluemix_result_passage = []
    wordnet_result_passage = []



    # process each sentence in passage
    for s in sentences_tokenized_list:

        print("")
        print("THIS sentence: " + str(s))



        # use bluemix alchemy language api to find key concepts
        bluemix_concept_str_list = []
        bluemix_concept_obj_list = []
        bluemix_keyword_str_list = []
        bluemix_keyword_obj_list = []

        bluemix_concept_str_list, bluemix_concept_obj_list = function.getBluemixConcept(s)
        bluemix_keyword_str_list, bluemix_keyword_obj_list = function.getBluemixKeyword(s)

        bluemix_concept_keyword_str_list = bluemix_concept_str_list + bluemix_keyword_str_list
        bluemix_concept_keyword_obj_list = bluemix_concept_obj_list + bluemix_keyword_obj_list

        #hold results
        simplified_sentence = ''
        wordnet_result_words = []
        w2v_result_list = []
        candidate_object_list = []


        stemmer = LancasterStemmer()# stem: reducing inflected (or sometimes derived) words to their stem

        tokenized_sentence = nltk.word_tokenize(s)
        tags = nltk.pos_tag(tokenized_sentence)#part of speech tag




        #bigrams and trigrams
        tokens_for_ngram = [token.lower() for token in tokenized_sentence if token not in string.punctuation]
        bi_grams = list(bigrams(tokens_for_ngram))
        tri_grams = list(trigrams(tokens_for_ngram))

        #go through each tag/go through token
        # tag[0] is word, tag[1] is tag
        for tag in tags:

            word_processing = tag[0]
            word_processing_pos = tag[1]
            word_processing_frequency = fdist[word_processing]
            word_processing_stem = stemmer.stem(word_processing)
            replace = False
            top_words = []


            if needToProcess(word_processing, word_processing_pos, bluemix_concept_keyword_str_list, min_frequent):

                # find syn using w2v model
                # print("NOW PROCESSING THIS WORD")
                # print(word_processing)

                top_words = w2v_model.most_similar(positive=[word_processing], topn=top_n_elements)
                # top 20 similar words and their frequency
                candidate_list_w2v = [w[0] for w in top_words]
                freq_list_w2v = [(fdist[w]) for w in candidate_list_w2v]
                similarity_list_w2v = [w2v_model.similarity(word_processing, w) for w in candidate_list_w2v]
                syllables_list_w2v = [function.countSyllables(w) for w in candidate_list_w2v]

                c_f_list_w2v = zip(candidate_list_w2v, freq_list_w2v, similarity_list_w2v, syllables_list_w2v)




                # filter candidate: >min_similarity and freq > original word
                # c_list, s_list, f_list = zip(*((c, s, f) for c, s, f in c_f_list_w2v if s > min_similarity and f > word_processing_frequency))
                # c_f_list_w2v_filtered = [(c, s, f) for c, s, f in c_f_list_w2v if s > min_similarity]

                c_list = []
                f_list = []
                s_list = []
                slb_list = []
                try:
                    candidate_list_w2v, freq_list_w2v, similarity_list_w2v, syllables_list_w2v = zip(*((candidate, freq, similarity, slb) for candidate, freq, similarity, slb in c_f_list_w2v if similarity > min_similarity and freq > word_processing_frequency and function.samePos(word_processing,candidate)))
                except:
                    pass



                # compute context similarity
                context_similarity_list_w2v = [getContextSimilarity(w, tokenized_sentence) for w in candidate_list_w2v]

                #print("context similarity of each candidate for " + str(word_processing))
                #print(context_similarity_list_w2v)

                # get ranking of each feature
                freq_rank_w2v = function.rankListReverse(freq_list_w2v)
                similarity_rank_w2v = function.rankListReverse(similarity_list_w2v)
                syllables_rank_w2v = function.rankList(syllables_list_w2v)
                context_similarit_rank_w2v = function.rankListReverse(context_similarity_list_w2v)

                complexity_list_w2v = [-math.log(f+0.1) for f in freq_list_w2v]
                complexity_rank_w2v = function.rankList(complexity_list_w2v)

                #print("This word: " + str(word_processing))
                #for x, y, z in zip(candidate_list_w2v, freq_list_w2v, freq_rank_w2v):
                    #print(x, y, z)



                '''ATTENTION HERE: compute Ngram probability for each candidate'''
                bigram_prob_list_w2v =[]
                trigram_prob_list_w2v = []
                ngram_prob_list_w2v = []
                for w in candidate_list_w2v:
                    # print("NOW DOING THIS CANDIDATE: " + str(w))
                    bigram_prob, tri_grams_prob, ngram_prob = getNgramProbability(bi_grams, tri_grams, word_processing, w)
                    bigram_prob_list_w2v.append(bigram_prob)
                    trigram_prob_list_w2v.append(tri_grams_prob)
                    ngram_prob_list_w2v.append(ngram_prob)




                bigram_prob_rank_w2v = function.rankListReverse(bigram_prob_list_w2v)
                trigram_prob_rank_w2v = function.rankListReverse(trigram_prob_list_w2v)
                ngram_prob_rank_w2v = function.rankListReverse(ngram_prob_list_w2v)


                data = np.array([complexity_rank_w2v, syllables_rank_w2v, similarity_rank_w2v, context_similarit_rank_w2v, ngram_prob_rank_w2v])
                print("check avg. before: ")
                print(data)
                avg_rank_w2v = np.average(data, axis=0)
                print(avg_rank_w2v)



                c_f_list_w2v_filtered = zip(candidate_list_w2v, freq_list_w2v, complexity_list_w2v, similarity_list_w2v, syllables_list_w2v, context_similarity_list_w2v, freq_rank_w2v, complexity_rank_w2v, similarity_rank_w2v, syllables_rank_w2v, context_similarit_rank_w2v, bigram_prob_list_w2v, bigram_prob_rank_w2v, trigram_prob_list_w2v, trigram_prob_rank_w2v, ngram_prob_list_w2v, ngram_prob_rank_w2v, avg_rank_w2v)

                # print("AFTER")
                # for (a, b, c) in c_f_list_w2v_filtered:
                # print(a, b, c)

                # find syn using wordnet
                candidate_list_wordnet = function.get_Candidate_Frequency_from_wordnet(word_processing, tag[1], tokenized_sentence)
                similarity_list_wordnet = []
                for w in candidate_list_wordnet:
                    try:
                        similarity_list_wordnet.append(w2v_model.similarity(word_processing, w))
                    except:
                        similarity_list_wordnet.append(0)
                freq_list_wordnet = [fdist[w] for w in candidate_list_wordnet]
                c_f_list_wordnet = zip(candidate_list_wordnet, freq_list_wordnet, similarity_list_wordnet)

                # candidate_frequency list sorted
                ordered_list = sorted(c_f_list_w2v_filtered, key=lambda c_f_list_w2v_filtered: c_f_list_w2v_filtered[17])
                ordered_list_wordnet = sorted(c_f_list_wordnet, key=lambda c_f_list_wordnet: c_f_list_wordnet[1], reverse=True)

                '''ATTENTION HERE'''
                candidate_object_list = []
                print("check length: " + str(len(ordered_list)))
                # 17 attributes
                for a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r in ordered_list:
                    candidate_object_list.append(constructs.CandidateObject(str(a), str(b), str(c), str(d), str(e), str(f), str(g), str(h), str(i), str(j), str(k), str(l), str(m), str(n), str(o), str(p), str(q), str(r)))

                #for c in candidate_object_list:
                    #print(str(c))


                    # print("ORDERED LIST")
                    # for (a, b, c) in ordered_list:
                    # print(a, b, c)


                    # print("ordered_list_wordnet for " + word_processing)
                    # for (a, b, c) in ordered_list_wordnet:
                    # print(a, b, c)
                # frequency of the original word_processing


                # w[0] is word_processing, w[1] is frequency
                # ordered_list = [w for w in ordered_list if (w[1] > word_processing_frequency * min_frequent_diff) and w[2] > min_similarity]
                # ordered_list_wordnet = [w for w in ordered_list_wordnet]

                '''ATTENTION HERE'''
                wordnet_result_words.append("Original_Word: " + tag[0] + "(" + str(fdist[word_processing]) + "), Candidates = " + str(ordered_list_wordnet))

                # synonmys = f.getSynonmys(word_processing)  ## get synonmys from wordnet
                # print synonmys


                ordered_list_same_POS = [w for w in ordered_list if (stemmer.stem(w[0]) != word_processing_stem)]
                if (len(ordered_list_same_POS) > 0):
                    word_processing = ordered_list_same_POS[0][0]
                    replace = True

                # for each candidate word_processing
                '''
                for w in ordered_list:

                    #only replace when stem(candidate) != stem(original word)
                    # and they have the same tag

                    if (st.stem(w[0]) != word_stem) and (f.samePos(word, w[0])): ##exclude morphological derivations and same pos
                        word = w[0]  ### do not use wordnet
                        #  if w[0] in synonmys:
                        #   word = w[0]
                        # else:
                        #   for syn in synonmys:
                        #       if st.stem(w[0]) == st.stem(syn):
                        #           word = w[0]
                        replace = True
                        break
                '''




            # update result
            if word_processing not in string.punctuation:
                simplified_sentence = simplified_sentence + ' ' + word_processing
            else:
                simplified_sentence = simplified_sentence + word_processing

            if replace is True:
                #process.append("Original_Word: " + tag[0] + "(" + str(word_processing_frequency) + "), Replaced with: " + word_processing + ", Candidates = " + str(ordered_list_same_POS))
                w2v_result_list.append(constructs.ResultObject(tag[0], word_processing_frequency, word_processing, candidate_object_list))

        simplified_passage += simplified_sentence
        w2v_result_passage += w2v_result_list
        bluemix_result_passage += bluemix_concept_keyword_obj_list
        wordnet_result_passage += wordnet_result_words

    return simplified_passage, wordnet_result_passage, bluemix_result_passage, w2v_result_passage




def needToProcess(word, pos, bluemix_list, min_freq):

    debugging = False
    word = word.lower()
    # only change tag = ['NN', 'NNS', 'JJ', 'RB', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    # correct pos
    # from translate import functions as f
    condition1 = function.checkPos(pos)
    # not in bluemix list
    condition2 = str(word) not in bluemix_list
    # in w2v model
    condition3 = str(word) in w2v_model
    # low frequency
    condition4 = fdist[str(word)] < min_freq

    if debugging:
        print(" ")
        print("This word: " + str(word))
        print("Pos?        "+str(condition1))
        print("in Bluemix? "+str(condition2))
        print("in w2v?     "+str(condition3))
        print("hard word?  "+str(condition4))

    if condition1 and condition2 and condition3 and condition4:
        return True
    else:
        return False



def isValidCandidate():
    return 0


def getContextSimilarity(c, tokenized_sentence):

    #print(" ")
    #print("For this candidate: " + str(c))
    sum = 0
    count = 0
    for w in tokenized_sentence:
        w = w.lower()
        if w not in string.punctuation and w in w2v_model:
            #print("between: " + str(c) + " and " + str(w))
            sum += w2v_model.similarity(c, w)
            count += 1

    return sum/count



def unigram_prob(word):
    return fdist[word]/len_corpus





def getNgramProbability(bi_grams, tri_grams, t, candidate):

    #print("NOW doing this word: " + str(t))
    bi_grams_for_this = [bi_token for bi_token in bi_grams if t in bi_token]
    #print("bi: ")
    #print(bi_grams_for_this)
    num_bi_grams = len(bi_grams_for_this)
    bi_sum = 0

    for bi in bi_grams_for_this:
        bi = list(bi)
        if bi[0] == t:
            bi[0] = candidate
        if bi[1] == t:
            bi[1] = candidate

        # P(how do) = P(how) * P(do|how)
        # print(bi)
        bi_sum += unigram_prob(bi[0]) * cprob_corpus_2gram[bi[0]].prob(bi[1])
    avg_bi_gram_prob = bi_sum / num_bi_grams
    # print("bi gram prob: " + str(avg_bi_gram_prob))


    tri_grams_for_this = [tri_token for tri_token in tri_grams if t in tri_token]
    #print("tri: ")
    #print(tri_grams_for_this)
    num_tri_grams = len(tri_grams_for_this)
    tri_sum = 0


    for tri in tri_grams_for_this:
        tri = list(tri)
        if tri[0] == t:
            tri[0] = candidate
        if tri[1] == t:
            tri[1] = candidate
        if tri[2] == t:
            tri[2] = candidate
        # P(how do you) = P(how) * P(do|how) * P(you|do)
        # print(tri)
        tri_sum += unigram_prob(tri[0]) * cprob_corpus_2gram[tri[0]].prob(tri[1]) * cprob_corpus_2gram[tri[1]].prob(tri[2])
    avg_tri_gram_prob = tri_sum / num_tri_grams
    #print("tri gram prob: " + str(avg_tri_gram_prob))

    avg_ngram_prob = (bi_sum + tri_sum) / (num_bi_grams + num_tri_grams)
    #print("Ngram prob: " + str(avg_ngram_prob))

    return avg_bi_gram_prob, avg_tri_gram_prob, avg_ngram_prob