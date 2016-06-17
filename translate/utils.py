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
from translate import constructs
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
w2v_model = Word2Vec.load_word2vec_format(w2vFilePath, binary=False)

news_text = brown.words()
emma = nltk.corpus.gutenberg.words()
r = nltk.corpus.reuters.words()
#form corpus
corpus = emma + news_text
corpus += r
#lower case frequency dictionary based on brown + emma + reuters
fdist = nltk.FreqDist(w.lower() for w in corpus)

min_similarity = 0.5

def simplify(s, min_frequent=100, min_frequent_diff = 1.2):

    # use bluemix alchemy language api to find key concepts
    alchemyapi_concept_response = alchemyapi.concepts('text', s)
    bluemix_concept = []
    bluemix_concept_list = []
    #print("BLUE MIX Concept")
    if alchemyapi_concept_response['status'] == 'OK':
        for concept in alchemyapi_concept_response['concepts']:
            bluemix_concept.append(str(concept['text']))
            bluemix_concept_list.append(constructs.BluemixConceptKeywordObject(str(concept['text']), str(concept['relevance'])))
        print(bluemix_concept)
    else:
        print('Error in concept tagging call: ', alchemyapi_concept_response['statusInfo'])



    alchemyapi_keyword_response = alchemyapi.keywords('text', s, {'sentiment': 1})
    bluemix_keyword = []
    bluemix_keyword_list = []
    #print("BLUE MIX keyword")
    if alchemyapi_keyword_response['status'] == 'OK':
        for keyword in alchemyapi_keyword_response['keywords']:
            bluemix_keyword.append(str(keyword['text']))
            bluemix_keyword_list.append(constructs.BluemixConceptKeywordObject(str(keyword['text']), str(keyword['relevance'])))
            print(bluemix_keyword)
    else:
        print('Error in keyword extaction call: ', alchemyapi_keyword_response['statusInfo'])


    #hold results
    simplified_sentence = ''
    process = []
    wordnet_result_words = []
    w2v_result_list = []
    candidate_object_list = []
    bluemix_concept_keyword_list = bluemix_concept_list + bluemix_keyword_list
    bluemix_concept_keyword = bluemix_concept + bluemix_keyword

    stemmer = LancasterStemmer()# stem: reducing inflected (or sometimes derived) words to their stem

    tokenized_sentence = nltk.word_tokenize(s)
    tags = nltk.pos_tag(tokenized_sentence)#part of speech tag

    #go through each tag/go through token
    # tag[0] is word, tag[1] is tag
    for tag in tags:

        word_processing = tag[0]
        word_processing_pos = tag[1]
        word_processing_frequency = fdist[word_processing]
        word_processing_stem = stemmer.stem(word_processing)
        replace = False
        top_words = []

        #from translate import functions as f
        #only change tag = ['NN', 'NNS', 'JJ', 'RB', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        if f.checkPos(word_processing_pos) and (word_processing not in bluemix_concept_keyword):

            # if this word_processing in word2vector
            if word_processing in w2v_model:

                #only change frequency<100
                if word_processing_frequency < min_frequent:

                    #find syn using w2v model
                    #print("NOW PROCESSING THIS WORD")
                    #print(word_processing)

                    top_words = w2v_model.most_similar(positive=[word_processing], topn = 20)
                    #top 20 similar words and their frequency
                    candidate_list_w2v = [w[0] for w in top_words]
                    freq_list_w2v = [fdist[w] for w in candidate_list_w2v]
                    similarity_list_w2v = [w2v_model.similarity(word_processing,w) for w in candidate_list_w2v]
                    syllables_list_w2v = [f.countSyllables(w) for w in candidate_list_w2v]
                    c_f_list_w2v = zip(candidate_list_w2v, freq_list_w2v, similarity_list_w2v, syllables_list_w2v)

                    # filter candidate: >min_similarity and freq > original word
                    #c_list, s_list, f_list = zip(*((c, s, f) for c, s, f in c_f_list_w2v if s > min_similarity and f > word_processing_frequency))
                    #c_f_list_w2v_filtered = [(c, s, f) for c, s, f in c_f_list_w2v if s > min_similarity]

                    c_list=[]
                    f_list=[]
                    s_list=[]
                    slb_list=[]
                    try:
                        c_list, f_list, s_list, slb_list = zip(*((candidate, freq, similarity, slb) for candidate, freq, similarity, slb in c_f_list_w2v if similarity > 0.7 and freq > word_processing_frequency and f.samePos(word_processing,candidate)))
                    except:
                        pass
                    c_f_list_w2v_filtered = zip(c_list, f_list, s_list, slb_list)


                    #print("AFTER")
                    #for (a, b, c) in c_f_list_w2v_filtered:
                        #print(a, b, c)

                    #find syn using wordnet
                    candidate_list_wordnet = f.get_Candidate_Frequency_from_wordnet(word_processing,tag[1],tokenized_sentence)
                    similarity_list_wordnet = []
                    for w in candidate_list_wordnet:
                        try:
                            similarity_list_wordnet.append(w2v_model.similarity(word_processing, w))
                        except:
                            similarity_list_wordnet.append(0)
                    freq_list_wordnet = [fdist[w] for w in candidate_list_wordnet]
                    c_f_list_wordnet = zip(candidate_list_wordnet, freq_list_wordnet, similarity_list_wordnet)


                    # candidate_frequency list sorted
                    ordered_list = sorted(c_f_list_w2v_filtered, key=lambda c_f_list_w2v_filtered:c_f_list_w2v_filtered[1], reverse=True)
                    ordered_list_wordnet = sorted(c_f_list_wordnet, key=lambda c_f_list_wordnet:c_f_list_wordnet[1], reverse=True)

                    '''ATTENTION HERE'''
                    candidate_object_list = []
                    for a,b,c,d in ordered_list:
                        candidate_object_list.append(constructs.CandidateObject(str(a), str(b), str(c), str(d)))

                    for c in candidate_object_list:
                        print(str(c))


                    #print("ORDERED LIST")
                    #for (a, b, c) in ordered_list:
                        #print(a, b, c)


                    #print("ordered_list_wordnet for " + word_processing)
                    #for (a, b, c) in ordered_list_wordnet:
                        #print(a, b, c)
                    #frequency of the original word_processing


                    #w[0] is word_processing, w[1] is frequency
                    #ordered_list = [w for w in ordered_list if (w[1] > word_processing_frequency * min_frequent_diff) and w[2] > min_similarity]
                    #ordered_list_wordnet = [w for w in ordered_list_wordnet]

                    '''ATTENTION HERE'''
                    wordnet_result_words.append("Original_Word: " + tag[0] + "(" + str(fdist[word_processing]) + "), Candidates = " + str(ordered_list_wordnet))

                    # synonmys = f.getSynonmys(word_processing)  ## get synonmys from wordnet
                    # print synonmys


                    ordered_list_same_POS = [w for w in ordered_list if (stemmer.stem(w[0]) != word_processing_stem)]
                    if (len(ordered_list_same_POS)>0):
                        word_processing = ordered_list_same_POS[0][0]
                        replace = True


                    #for each candidate word_processing
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

        simplified_sentence = simplified_sentence + word_processing + ' '
        if replace is True:
            #word_processing has been replaced
            #tag[0] is original
            #process = process + tag[0] + ' : ' + str(top_words) + '\n'
            process.append("Original_Word: " + tag[0] + "(" + str(word_processing_frequency) + "), Replaced with: " + word_processing + ", Candidates = " + str(ordered_list_same_POS))
            w2v_result_list.append(constructs.ResultObject(tag[0], word_processing_frequency, word_processing, candidate_object_list))

    return simplified_sentence, process, wordnet_result_words, bluemix_concept_keyword_list, w2v_result_list
