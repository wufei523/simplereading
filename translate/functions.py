import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import sent_tokenize
from nltk.corpus import brown
from nltk.corpus import wordnet
import os
from os.path import join
import numpy as np
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def checkPos(word):
	pos = ['NN', 'NNS', 'JJ', 'RB', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	if word in pos:
		return True
	else:
		return False

def readSetting():
	path = os.getcwd()
	with open(join(path, 'setting'), 'r') as f:
		setting = f.readlines()
	return setting

def readDoc():
	path = os.getcwd()
	p = open(join(path, 'doc'), 'r')
	text = [line.decode('utf-8').strip() for line in p.readlines()]
	p.close()
	return '\n'.join(text)

def getTopwords():
	s = readDoc()
	setting = readSetting()
	model_path = setting[0]
	model = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=True)
	text = nltk.word_tokenize(s)
	tags = nltk.pos_tag(text)
	res = ''
	for tag in tags:
		word = tag[0]
		if checkPos(tag[1]):
			if word in model:
				res = res + word + ': ' 
				top_words = model.most_similar(positive=[word], topn = 10)
				for w in top_words:
					res = res + '(' + w[0] + ',' + str(w[1]) + ')'
				res = res + '\n'
			
	path = os.getcwd()
	p = open(join(path, 'topwords'), 'w')
	p.write(res.encode('utf-8'))
	p.close()

def samePos(word1, word2):
	return nltk.pos_tag([word1])[0][1] == nltk.pos_tag([word2])[0][1]

def freq_diff(freq1, freq2):
	return freq1*1.2 < freq2

def getSynonmys(word):
	res = []
	syns = wordnet.synsets(word)
	for s in syns:
		for l in s.lemmas:
			res.append(l.name)
	return list(set(res))



