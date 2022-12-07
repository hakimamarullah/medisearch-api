
import numpy as np
import pickle
from scipy.spatial.distance import cosine

import os
from .search_response_mapper import to_search_response

current_dir =  os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(current_dir + "/../")
CORPUS_PATH = os.path.join(parent_dir, 'data', 'nfcorpus')



def pickle_load_utils_trained(filename):
   with open(os.path.join(os.getcwd(), 'utils','trained_model',filename), 'rb') \
            as docs:
            return pickle.load(docs)


class CorpusDocuments(object):

    def __init__(self) -> None: 
        [self.documents] = pickle_load_utils_trained('documents.docs')
    
    def __str__(self) -> str:
        """
         RETURN FIRST DOC IN DOCUMENTS
        """
        return str(list(self.documents.items())[0])

class CorpusQuery(object):

    def __init__(self) -> None:
        [self.queries] = pickle_load_utils_trained('queries.queries')
    
    def __str__(self) -> str:
        """
         RETURN FIRST QUERY IN QUERIES
        """
        return str(list(self.queries.items())[0])

class PretrainedModel(CorpusDocuments, CorpusQuery):
    NUM_NEGATIVES = 1
    NUM_LATENT_TOPICS = 200

    def __init__(self) -> None:
        print("Building The Model....")
        CorpusDocuments.__init__(self)
        CorpusQuery.__init__(self)
        self.q_docs_rel = dict()
        self.group_qid_count = list()
        self.dataset = list()
        self.docs_vector = dict()
        
        [self.docs_vector] = pickle_load_utils_trained('docs-vetor.vector')
        self.init_q_docs_rel()
        self.load_dataset()
    
    
    def load_lsa(self):
        [self.lsi_model] = pickle_load_utils_trained('lsa.pkl')
       
  
    def load_model(self):
        [self.ranker] = pickle_load_utils_trained('model.pkl')
    
    def init_q_docs_rel(self) -> None:
        [self.q_docs_rel] = pickle_load_utils_trained('model.qrel')
        
    
    def load_dataset(self) -> None:
        [self.group_qid_count] = pickle_load_utils_trained('group-qid-count.data')
        
        [self.dataset] = pickle_load_utils_trained('dataset.data')

        
        [self.dictionary] = pickle_load_utils_trained('dictionary.dict')


        [self.bow_corpus] = pickle_load_utils_trained('bow-corpus.corpus')

        assert sum(self.group_qid_count) == len(self.dataset), "Ooops...Something's wrong!!"
        print("Dataset loaded successfully!")
    
    
    # test melihat representasi vector dari sebuah dokumen & query
    def vector_rep(self, text):
        assert self.lsi_model != None, "Run generate or lsa model first"
        rep = [topic_value for (_, topic_value) in self.lsi_model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS
    
    def get_features(self, query, doc, did):
        v_q = self.vector_rep(query)
        v_d = self.docs_vector[did]
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]
    
    def get_dataset(self):
        X = []
        Y = []
        for (query, doc, rel) in self.dataset:
            X.append(self.get_features(query, doc))
            Y.append(rel)

        # ubah X dan Y ke format numpy array
        X = np.array(X)
        Y = np.array(Y)

        print(X.shape)
        print(Y.shape)

        return X, Y
         

    def get_documents(self, query:str) -> list:
        self.load_lsa()
        self.load_model()
        x_unseen = []
        did_doc = []
        result = []
        with open(os.path.join(CORPUS_PATH, 'test.docs'), 'r') as docs:
            for doc in docs:
                content = doc.split('\t')
                did_doc.append((content[0], content[1]))
                x_unseen.append(self.get_features(query.split(), content[1].split(), content[0]))
            x_unseen = np.array(x_unseen)
            scores = self.ranker.predict(x_unseen)

            did_scores = [to_search_response(x) for x in zip(did_doc, scores)]
            result = sorted(did_scores, key = lambda res: res.score, reverse = True)

        docs.close()
        return result

    
    def __str__(self) -> str:
        return f'''\
        {CorpusDocuments.__str__(self)} 
        {CorpusQuery.__str__(self)}'''

class Ranker():

    @staticmethod
    async def get_documents(model: PretrainedModel,query: str) -> list:
        return model.get_documents(query)
    


