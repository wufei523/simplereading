import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import sent_tokenize
from nltk.corpus import brown
import os
from os.path import join
import numpy as np
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from translate import functions as f
from gensim.models.word2vec import Word2Vec
import os
import json
from os.path import join
from translate.alchemyapi import AlchemyAPI
alchemyapi = AlchemyAPI()

#model = Word2Vec.load_word2vec_format("/Users/luo/desktop/GoogleNews-vectors-negative300.bin.gz", binary = True)
#model = Word2Vec.load_word2vec_format("/home/twang/dataset/GoogleNews-vectors-negative300.bin.gz", binary = True)
path = os.getcwd()

#w2vFilePath = join(path, 'GoogleNews-vectors-negative300.bin.gz')
#model = Word2Vec.load_word2vec_format(w2vFilePath, binary = True)
w2vFilePath = join(path, 'glove_model.txt')
model = Word2Vec.load_word2vec_format(w2vFilePath, binary=False)

news_text = brown.words()
emma = nltk.corpus.gutenberg.words()
r = nltk.corpus.reuters.words()
#form corpus
corpus = emma + news_text
corpus += r
#lower case frequency dictionary based on brown + emma + reuters
fdist = nltk.FreqDist(w.lower() for w in corpus)


def simplify_old(s):
    res = ''
    st = LancasterStemmer()
    text = nltk.word_tokenize(s)
    tags = nltk.pos_tag(text)

    for tag in tags:
        word = tag[0]
        if f.checkPos(tag[1]):
            if word in model:
                word_stem = st.stem(word)
                top_words = model.most_similar(positive=[word], topn = 20)
                candidate_list = [w[0] for w in top_words]
                freq_list = [fdist[w] for w in candidate_list]
                c_f_list = zip(candidate_list, freq_list)
                ordered_list = sorted(c_f_list, key=lambda c_f_list:c_f_list[1], reverse=True)
                word_freq = fdist[word]
                #			synonmys = f.getSynonmys(word)  ## get synonmys from wordnet
                # print synonmys
                for w in ordered_list:
                    if not f.freq_diff(word_freq, w[1]):  ## break for loop if candidate word frequency does not exceed the word frequency by a threshold
                            break
                    if st.stem(w[0]) != word_stem and f.samePos(word, w[0]): ##exclude morphological derivations and same pos
                            word = w[0]  ### do not use wordnet
        # if w[0] in synonmys:
        # 	word = w[0]
        # else:
        # 	for syn in synonmys:
        # 		if st.stem(w[0]) == st.stem(syn):
        # 			word = w[0]

        res = res + word + ' '
    return res

def simplify(s, min_frequent=100, min_frequent_diff = 1.2):
    res = ''
    process = []
    wordnet_result = []
    # stem: reducing inflected (or sometimes derived) words to their stem
    st = LancasterStemmer()
    #tokenize input text
    text = nltk.word_tokenize(s)
    #part of speech tag
    #eg: tags = [('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'), ('completely', 'RB'), ('different', 'JJ')]
    tags = nltk.pos_tag(text)
    #go through each tag = go through token
    for tag in tags:
        #get the word, tag[0] is word, tag[1] is tag
        word = tag[0]
        replace = False
        top_words = []

        #from translate import functions as f
        #only change tag = ['NN', 'NNS', 'JJ', 'RB', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        if f.checkPos(tag[1]):

            # if this word in word2vector
            if word in model:
                #only change frequency<100
                if fdist[word] < min_frequent:  ## the min frequent less than
                    #stem the word
                    word_stem = st.stem(word)
                    #find similar words top 20
                    top_words = model.most_similar(positive=[word], topn = 20)
                    #top 20 similar words and their frequency
                    candidate_list = [w[0] for w in top_words]
                    similarity_list = [model.similarity(word,w) for w in candidate_list]
                    freq_list = [fdist[w] for w in candidate_list]
                    #candidate_frequency dictionary
                    c_f_list = zip(candidate_list, freq_list, similarity_list)

                    candidate_list_wordnet = f.get_Candidate_Frequency_from_wordnet(word,tag[1],text)

                    similarity_list_wordnet = []
                    for w in candidate_list_wordnet:
                        try:
                            similarity_list_wordnet.append(model.similarity(word, w))
                        except:
                            similarity_list_wordnet.append(0)

                     #similarity_list_wordnet = [model.similarity(word, w) or 0 for w in candidate_list_wordnet]


                    freq_list_wordnet = [fdist[w] for w in candidate_list_wordnet]
                    c_f_list_wordnet = zip(candidate_list_wordnet, freq_list_wordnet, similarity_list_wordnet)









                    # candidate_frequency list sorted
                    ordered_list = sorted(c_f_list, key=lambda c_f_list:c_f_list[1], reverse=True)
                    ordered_list_wordnet = sorted(c_f_list_wordnet, key=lambda c_f_list_wordnet:c_f_list_wordnet[1], reverse=True)
                    print("ordered_list_wordnet for " + word)
                    for (a, b, c) in ordered_list_wordnet:
                        print(a, b, c)
                    #frequency of the original word
                    word_freq = fdist[word]

                    #w[0] is word, w[1] is frequency
                    #only want the candidates with frequency higher original word
                    ordered_list = [w for w in ordered_list if (word_freq * min_frequent_diff) < w[1]]
                    ordered_list_wordnet = [w for w in ordered_list_wordnet]

                    '''ATTENTION HERE'''
                    wordnet_result.append("Original_Word: " + tag[0] + "(" + str(fdist[word]) + "), Candidates = " + str(ordered_list_wordnet))

                    # synonmys = f.getSynonmys(word)  ## get synonmys from wordnet
                    # print synonmys


                    ordered_list_same_POS = [w for w in ordered_list if ((st.stem(w[0]) != word_stem) and (f.samePos(word, w[0])))]
                    if (len(ordered_list_same_POS)>0):
                        word = ordered_list_same_POS[0][0]
                        replace = True


                    #for each candidate word
                    '''
                    for w in ordered_list:

                        #only replace when stem(candidate) != stem(original word)
                        # and they have the same tag

                        if (st.stem(w[0]) != word_stem) and (f.samePos(word, w[0])): ##exclude morphological derivations and same pos
                            word = w[0]  ### do not use wordnet
                            #  if w[0] in synonmys:
							# 	word = w[0]
							# else:
							# 	for syn in synonmys:
							# 		if st.stem(w[0]) == st.stem(syn):
							# 			word = w[0]
                            replace = True
                            break
                    '''

        res = res + word + ' '
        if replace is True:
            #word has been replaced
            #tag[0] is original
            #process = process + tag[0] + ' : ' + str(top_words) + '\n'
           process.append("Original_Word: " + tag[0] + "(" + str(word_freq) + "), Replaced with: " + word + ", Candidates = " + str(ordered_list_same_POS))

    #use bluemix alchemy language api

    response = alchemyapi.concepts('text', s)
    bluemix_concept = []
    print("BLUE MIX")
    if response['status'] == 'OK':
        for concept in response['concepts']:
            bluemix_concept.append(str(concept['text']) + ", " + str(concept['relevance']))
        print(bluemix_concept)
    else:
        print('Error in concept tagging call: ', response['statusInfo'])


    return res, process, wordnet_result, bluemix_concept
