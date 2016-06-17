

class CandidateObject:
    def __init__(self,n,f,s,slb):
     self.name = n
     self.frequency = f
     self.similarity = s
     self.syllables = slb

    def __str__(self):
        return self.name + ", " + self.frequency + ", " + self.similarity + ", " + self.syllables + "/n"



class ResultObject:

    def __init__(self, w,f, r,c):
        self.original_word = w
        self.frequency = f
        self.replace_with = r
        self.candidate_list = c # a list of candidate objects



class BluemixConceptKeywordObject:

    def __init__(self, n, r):
        self.name = n
        self.relevance = r