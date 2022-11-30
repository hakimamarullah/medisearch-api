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


def prepare_data():
    print("Preparing data")
    print("Downloading corpus...")
    with requests.get('https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz' , 
        stream=True, auth=('user', 'pass')) as  rx,\
        tarfile.open(fileobj=rx.raw  , mode="r:gz") as tarobj  :
        print("Extracting files...")
        tarobj.extractall() 
   
    print("All files downloaded...")


class CorpusDocuments(object):

    def __init__(self, path: str) -> None:
        print(f"Generating corpus documents for {path}")
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
        print(f"Generating corpus queries for {path}")
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

    def __init__(self, docs_path: str, query_path:str, qrel_path:str) -> None:
        print("Building The Model....")
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
    
    def load_dataset(self) -> None:
        print("Loading dataset....")
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
        print("Dataset loaded successfully!")
    
    def generate_lsa(self):
        assert self.bow_corpus != None, "Run load_dataset() before generating LSA Model"
        print("Generate LSA model..")
        self.lsi_model = LsiModel(self.bow_corpus, num_topics = self.NUM_LATENT_TOPICS)
        print("LSA Model generated..")
    
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

        print("query        :", query)
        print("SERP/Ranking :")
        for (did, score) in sorted_did_scores:
            print(did, score)
    
    def __str__(self) -> str:
        return f'''\
        {CorpusDocuments.__str__(self)} 
        {CorpusQuery.__str__(self)}'''

def main():
    if(not os.path.exists('./nfcorpus')):
        prepare_data()

    query = "how much cancer risk can be avoided through lifestyle change ?"

    docs =[("D1", "dietary restriction reduces insulin-like growth factor levels modulates apoptosis cell proliferation tumor progression num defici pubmed ncbi abstract diet contributes one-third cancer deaths western world factors diet influence cancer elucidated reduction caloric intake dramatically slows cancer progression rodents major contribution dietary effects cancer insulin-like growth factor igf-i lowered dietary restriction dr humans rats igf-i modulates cell proliferation apoptosis tumorigenesis mechanisms protective effects dr depend reduction multifaceted growth factor test hypothesis igf-i restored dr ascertain lowering igf-i central slowing bladder cancer progression dr heterozygous num deficient mice received bladder carcinogen p-cresidine induce preneoplasia confirmation bladder urothelial preneoplasia mice divided groups ad libitum num dr num dr igf-i igf-i/dr serum igf-i lowered num dr completely restored igf-i/dr-treated mice recombinant igf-i administered osmotic minipumps tumor progression decreased dr restoration igf-i serum levels dr-treated mice increased stage cancers igf-i modulated tumor progression independent body weight rates apoptosis preneoplastic lesions num times higher dr-treated mice compared igf/dr ad libitum-treated mice administration igf-i dr-treated mice stimulated cell proliferation num fold hyperplastic foci conclusion dr lowered igf-i levels favoring apoptosis cell proliferation ultimately slowing tumor progression mechanistic study demonstrating igf-i supplementation abrogates protective effect dr neoplastic progression"), 
       ("D2", "study hard as your blood boils"), 
       ("D3", "processed meats risk childhood leukemia california usa pubmed ncbi abstract relation intake food items thought precursors inhibitors n-nitroso compounds noc risk leukemia investigated case-control study children birth age num years los angeles county california united states cases ascertained population-based tumor registry num num controls drawn friends random-digit dialing interviews obtained num cases num controls food items principal interest breakfast meats bacon sausage ham luncheon meats salami pastrami lunch meat corned beef bologna hot dogs oranges orange juice grapefruit grapefruit juice asked intake apples apple juice regular charcoal broiled meats milk coffee coke cola drinks usual consumption frequencies determined parents child risks adjusted risk factors persistent significant associations children's intake hot dogs odds ratio num num percent confidence interval ci num num num hot dogs month trend num fathers intake hot dogs num ci num num highest intake category trend num evidence fruit intake provided protection results compatible experimental animal literature hypothesis human noc intake leukemia risk potential biases data study hypothesis focused comprehensive epidemiologic studies warranted"), 
       ("D4", "long-term effects calorie protein restriction serum igf num igfbp num concentration humans summary reduced function mutations insulin/igf-i signaling pathway increase maximal lifespan health span species calorie restriction cr decreases serum igf num concentration num protects cancer slows aging rodents long-term effects cr adequate nutrition circulating igf num levels humans unknown report data long-term cr studies num num years showing severe cr malnutrition change igf num igf num igfbp num ratio levels humans contrast total free igf num concentrations significantly lower moderately protein-restricted individuals reducing protein intake average num kg num body weight day num kg num body weight day num weeks volunteers practicing cr resulted reduction serum igf num num ng ml num num ng ml num findings demonstrate unlike rodents long-term severe cr reduce serum igf num concentration igf num igfbp num ratio humans addition data provide evidence protein intake key determinant circulating igf num levels humans suggest reduced protein intake important component anticancer anti-aging dietary interventions"), 
       ("D5", "cancer preventable disease requires major lifestyle abstract year num million americans num million people worldwide expected diagnosed cancer disease commonly believed preventable num num cancer cases attributed genetic defects remaining num num roots environment lifestyle lifestyle factors include cigarette smoking diet fried foods red meat alcohol sun exposure environmental pollutants infections stress obesity physical inactivity evidence cancer-related deaths num num due tobacco num num linked diet num num due infections remaining percentage due factors radiation stress physical activity environmental pollutants cancer prevention requires smoking cessation increased ingestion fruits vegetables moderate alcohol caloric restriction exercise avoidance direct exposure sunlight minimal meat consumption grains vaccinations regular check-ups review present evidence inflammation link agents/factors cancer agents prevent addition provide evidence cancer preventable disease requires major lifestyle")]
    
    model = Model("nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")
    model.generate_lsa()
    model.train_model()
    scores = model.predict(docs, query)

    model.interpret_serp_ranking(docs, query, scores)

if __name__ == '__main__':
    main()

