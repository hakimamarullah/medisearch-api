import random

import numpy as np
from gensim.models import LsiModel
from gensim.corpora import Dictionary
import pickle
from scipy.spatial.distance import cosine

import lightgbm
import os
from model.search_response import SearchResponse
from .search_response_mapper import to_search_response

current_dir =  os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(current_dir + "/../")
CORPUS_PATH = os.path.join(parent_dir, 'data', 'nfcorpus')





class CorpusDocuments(object):

    def __init__(self, path: str) -> None: 
        with open(os.path.join(os.getcwd(), 'utils','trained_model','documents.docs'), 'rb') \
            as docs:
            [self.documents] = pickle.load(docs)
    
    def __str__(self) -> str:
        """
         RETURN FIRST DOC IN DOCUMENTS
        """
        return str(list(self.documents.items())[0])

class CorpusQuery(object):

    def __init__(self, path:str) -> None:
        
        with open(os.path.join(os.getcwd(), 'utils','trained_model','queries.queries'), 'rb') \
            as queries:
            [self.queries] = pickle.load(queries)
    
    def __str__(self) -> str:
        """
         RETURN FIRST QUERY IN QUERIES
        """
        return str(list(self.queries.items())[0])

class Model(CorpusDocuments, CorpusQuery):
    NUM_NEGATIVES = 1
    NUM_LATENT_TOPICS = 200

    def __init__(self, docs_path: str = 'train.docs', query_path:str = 'train.vid-desc.queries', qrel_path:str = 'train.3-2-1.qrel') -> None:
        print("Building The Model....")
        CorpusDocuments.__init__(self,docs_path)
        CorpusQuery.__init__(self,query_path)
        self.q_docs_rel = dict()
        self.group_qid_count = list()
        self.dataset = list()
        self.docs_vector = dict()
        with open(os.path.join(os.getcwd(), 'utils','trained_model','docs-vetor.vector'), 'rb') as vectors:
           [self.docs_vector] = pickle.load(vectors)
        self.init_q_docs_rel(qrel_path)
        self.load_dataset()
    
    def save_lsa(self):
        with open(os.path.join(os.getcwd(), 'trained_model','lsa.pkl'), 'wb+') as lsa:
            pickle.dump([self.lsi_model], lsa)
        lsa.close()
    
    def load_lsa(self):
        with open(os.path.join(os.getcwd(), 'utils','trained_model','lsa.pkl'), 'rb') as lsa:
            [self.lsi_model] = pickle.load(lsa)
        lsa.close()
    
    def save_model(self):
        with open(os.path.join(os.getcwd(), 'trained_model', 'model.pkl'), 'wb+') as model:
            pickle.dump([self.ranker], model)
        model.close()
    
    def load_model(self):
        with open(os.path.join(os.getcwd(), 'utils', 'trained_model', 'model.pkl'), 'rb') as model:
            [self.ranker] = pickle.load(model)
        model.close()
    
    def init_q_docs_rel(self, qrel_path:str) -> None:
        with open(os.path.join(os.getcwd(), 'utils','trained_model','model.qrel'), 'rb') as qrel:
            [self.q_docs_rel] = pickle.load(qrel)
        
    
    def load_dataset(self) -> None:
        
        with open(os.path.join(os.getcwd(), 'utils','trained_model','group-qid-count.data'), 'rb') as group:
            [self.group_qid_count] = pickle.load(group)
        
        with open(os.path.join(os.getcwd(), 'utils','trained_model','dataset.data'), 'rb') as data:
            [self.dataset] = pickle.load(data)

        with open(os.path.join(os.getcwd(), 'utils','trained_model','dictionary.dict'), 'rb') as dicts:
            [self.dictionary] = pickle.load(dicts)
        
        with open(os.path.join(os.getcwd(), 'utils','trained_model','bow-corpus.corpus'), 'rb') as bow:
            [self.bow_corpus] = pickle.load(bow)

        assert sum(self.group_qid_count) == len(self.dataset), "Ooops...Something's wrong!!"
        print("Dataset loaded successfully!")
    
    def generate_lsa(self):
        assert self.bow_corpus != None, "Run load_dataset() before generating LSA Model"
        print("Generate LSA model..")
        self.lsi_model = LsiModel(self.bow_corpus, num_topics = self.NUM_LATENT_TOPICS)
        self.save_lsa()
        print("LSA Model generated..")
    
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

class Ranker(Model):

    @staticmethod
    def get_documents(query: str) -> list:
        model = Model()
        return model.get_documents(query)[:2]
    

def main():
  
    query = "how much cancer risk can be avoided through lifestyle change ?"

    docs =[("D1", "dietary restriction reduces insulin-like growth factor levels modulates apoptosis cell proliferation tumor progression num defici pubmed ncbi abstract diet contributes one-third cancer deaths western world factors diet influence cancer elucidated reduction caloric intake dramatically slows cancer progression rodents major contribution dietary effects cancer insulin-like growth factor igf-i lowered dietary restriction dr humans rats igf-i modulates cell proliferation apoptosis tumorigenesis mechanisms protective effects dr depend reduction multifaceted growth factor test hypothesis igf-i restored dr ascertain lowering igf-i central slowing bladder cancer progression dr heterozygous num deficient mice received bladder carcinogen p-cresidine induce preneoplasia confirmation bladder urothelial preneoplasia mice divided groups ad libitum num dr num dr igf-i igf-i/dr serum igf-i lowered num dr completely restored igf-i/dr-treated mice recombinant igf-i administered osmotic minipumps tumor progression decreased dr restoration igf-i serum levels dr-treated mice increased stage cancers igf-i modulated tumor progression independent body weight rates apoptosis preneoplastic lesions num times higher dr-treated mice compared igf/dr ad libitum-treated mice administration igf-i dr-treated mice stimulated cell proliferation num fold hyperplastic foci conclusion dr lowered igf-i levels favoring apoptosis cell proliferation ultimately slowing tumor progression mechanistic study demonstrating igf-i supplementation abrogates protective effect dr neoplastic progression"), 
       ("D2", "study hard as your blood boils"), 
       ("D3", "processed meats risk childhood leukemia california usa pubmed ncbi abstract relation intake food items thought precursors inhibitors n-nitroso compounds noc risk leukemia investigated case-control study children birth age num years los angeles county california united states cases ascertained population-based tumor registry num num controls drawn friends random-digit dialing interviews obtained num cases num controls food items principal interest breakfast meats bacon sausage ham luncheon meats salami pastrami lunch meat corned beef bologna hot dogs oranges orange juice grapefruit grapefruit juice asked intake apples apple juice regular charcoal broiled meats milk coffee coke cola drinks usual consumption frequencies determined parents child risks adjusted risk factors persistent significant associations children's intake hot dogs odds ratio num num percent confidence interval ci num num num hot dogs month trend num fathers intake hot dogs num ci num num highest intake category trend num evidence fruit intake provided protection results compatible experimental animal literature hypothesis human noc intake leukemia risk potential biases data study hypothesis focused comprehensive epidemiologic studies warranted"), 
       ("D4", "long-term effects calorie protein restriction serum igf num igfbp num concentration humans summary reduced function mutations insulin/igf-i signaling pathway increase maximal lifespan health span species calorie restriction cr decreases serum igf num concentration num protects cancer slows aging rodents long-term effects cr adequate nutrition circulating igf num levels humans unknown report data long-term cr studies num num years showing severe cr malnutrition change igf num igf num igfbp num ratio levels humans contrast total free igf num concentrations significantly lower moderately protein-restricted individuals reducing protein intake average num kg num body weight day num kg num body weight day num weeks volunteers practicing cr resulted reduction serum igf num num ng ml num num ng ml num findings demonstrate unlike rodents long-term severe cr reduce serum igf num concentration igf num igfbp num ratio humans addition data provide evidence protein intake key determinant circulating igf num levels humans suggest reduced protein intake important component anticancer anti-aging dietary interventions"), 
       ("D5", "cancer preventable disease requires major lifestyle abstract year num million americans num million people worldwide expected diagnosed cancer disease commonly believed preventable num num cancer cases attributed genetic defects remaining num num roots environment lifestyle lifestyle factors include cigarette smoking diet fried foods red meat alcohol sun exposure environmental pollutants infections stress obesity physical inactivity evidence cancer-related deaths num num due tobacco num num linked diet num num due infections remaining percentage due factors radiation stress physical activity environmental pollutants cancer prevention requires smoking cessation increased ingestion fruits vegetables moderate alcohol caloric restriction exercise avoidance direct exposure sunlight minimal meat consumption grains vaccinations regular check-ups review present evidence inflammation link agents/factors cancer agents prevent addition provide evidence cancer preventable disease requires major lifestyle")]
    
    model = Model("train.docs", "train.vid-desc.queries", "train.3-2-1.qrel")
    model.generate_lsa()
    model.train_model()
    scores = model.predict(docs, query)

    model.interpret_serp_ranking(docs, query, scores)

if __name__ == '__main__':
    main()

