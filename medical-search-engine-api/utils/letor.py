import random
import os
import numpy as np

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary

from scipy.spatial.distance import cosine

import lightgbm
import  requests 
import  tarfile
import pickle


def prepare_data():
    with requests.get('https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz' , 
        stream=True, auth=('user', 'pass')) as  rx,\
        tarfile.open(fileobj=rx.raw  , mode="r:gz") as tarobj  :
        tarobj.extractall() 


class CorpusDocuments(object):

    def __init__(self, path: str) -> None:
        self.documents = dict()
        with open(path) as file:
            for line in file:
                doc_id, content = line.split("\t")
                self.documents[doc_id] = content.split()
    
    def __str__(self) -> str:
        """
         RETURN FIRST DOC IN DOCUMENTS
        """
        return str(list(self.documents.items())[0])

class CorpusQuery(object):

    def __init__(self, path:str) -> None:
        self.queries = dict()
        try:
            with open(path, encoding='utf-8') as file:
                for line in file:
                    q_id, content = line.split("\t")
                    self.queries[q_id] = content.split()
        except Exception as e:
            print(e)
    
    def __str__(self) -> str:
        """
         RETURN FIRST QUERY IN QUERIES
        """
        return str(list(self.queries.items())[0])

class Model(CorpusDocuments, CorpusQuery):
    NUM_NEGATIVES = 1
    NUM_LATENT_TOPICS = 200

    def __init__(self, docs_path="nfcorpus/train.docs", query_path="nfcorpus/train.vid-desc.queries", qrel_path="nfcorpus/train.3-2-1.qrel") -> None:
        CorpusDocuments.__init__(self,docs_path)
        CorpusQuery.__init__(self,query_path)
        self.q_docs_rel = dict()
        self.group_qid_count = list()
        self.dataset = list()
        self.init_q_docs_rel(qrel_path)
        self.load_dataset()
    
    def init_q_docs_rel(self, qrel_path:str) -> None:
        with open(qrel_path) as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))
    
    def save_model(self):
        with open("trained_model/model-1.pkl", 'wb') as f:
            pickle.dump([self.ranker], f)

    def load_model(self):
        with open("trained_model/model-1.pkl", 'rb') as f:
            [self.ranker] = pickle.load(f)

    def load_dataset(self) -> None:
        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + self.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))
            # tambahkan satu negative (random sampling saja dari documents)
            self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))
        
        # bentuk dictionary, bag-of-words corpus, dan kemudian Latent Semantic Indexing
        # dari kumpulan 3612 dokumen.
        self.dictionary = Dictionary()
        self.bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]

        assert sum(self.group_qid_count) == len(self.dataset), "Ooops...Something's wrong!!"
        
    
    def generate_lsa(self):
        assert self.bow_corpus != None, "Run load_dataset() before generating LSA Model"
        self.lsi_model = LsiModel(self.bow_corpus, num_topics = self.NUM_LATENT_TOPICS)
        self.save_lsa()
       
    
    def save_lsa(self):
        with open(f'trained_model/lsa-1.pkl', 'wb') as f:
            pickle.dump([self.lsi_model], f)
      

    def load_lsa(self):
        with open(f'trained_model/lsa-1.pkl', 'rb') as f:
            [self.lsi_model] = pickle.load(f)
    
    # test melihat representasi vector dari sebuah dokumen & query
    def vector_rep(self, text):
        assert self.lsi_model != None, "Run generate or lsa model first"
        rep = [topic_value for (_, topic_value) in self.lsi_model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS
    
    def get_features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
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

        return X, Y
    
    def train_model(self, objective="lambdarank", boosting_type="gbdt",
     importance_type="gain", metric="ndcg", verbose=10):

        X, Y = self.get_dataset()
        self.ranker = lightgbm.LGBMRanker(
                    objective=objective,
                    boosting_type = boosting_type,
                    n_estimators = 100,
                    importance_type = importance_type,
                    metric = metric,
                    num_leaves = 40,
                    learning_rate = 0.02,
                    max_depth = -1)
        # di contoh kali ini, kita tidak menggunakan validation set
        # jika ada yang ingin menggunakan validation set, silakan saja
        self.ranker.fit(X, Y,
                group = self.group_qid_count,
                verbose = verbose)
        self.save_model()

    def predict(self, docs, query):
        assert self.lsi_model != None, "Run generate_lsa or load_lsa first"
        assert self.ranker != None, "Run train_model or load_model first"
        # bentuk ke format numpy array
        X_unseen = []
        for _, doc in docs:
            X_unseen.append(self.get_features(query.split(), doc.split()))

        X_unseen = np.array(X_unseen)
        scores = self.ranker.predict(X_unseen)
        return scores
    
    def interpret_serp_ranking(self, docs, query, scores):
        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

       
        for (did, score) in sorted_did_scores:
            print(did, score)
    
 
    def get_documents(self, query):
        assert self.lsi_model != None, "Run generate_lsa or load_lsa first"
        assert self.ranker != None, "Run train_model or load_model first"

        X_unseen = []
        result = []
        with open('nfcorpus/test.docs', 'r') as docs:
            all_docs = []
            for line in docs:
                content = line.split("\t")
                all_docs.append(content[1])
                X_unseen.append(self.get_features(query.split(), content[1].split()))

            X_unseen = np.array(X_unseen)
            scores = self.ranker.predict(X_unseen)

            did_scores = [x for x in zip(all_docs, scores)][:50]
            sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)
            result = [line for (line, _) in sorted_did_scores]
        return result
        

    def __str__(self) -> str:
        return f'''\
        {CorpusDocuments.__str__(self)} 
        {CorpusQuery.__str__(self)}'''




class Ranker():

    @staticmethod
    def get_documents(query:str) -> list:
        model = Model()
        model.load_lsa()
        model.load_model()
        return model.get_documents(query)

