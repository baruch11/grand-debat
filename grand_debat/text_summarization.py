""" text summarization """
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# create_title
TITLE_LEN_MIN = 100
TITLE_LEN_MAX = 200
# sentence selection for each topic
PURE_TOPIC_THRESH = 0.7
# pagerank
MAX_SENTENCES_PAGERANK = 3000

class GDebatSummerization:
    """Summarization algorithms.

    These algorithm work on sentences.

    parameters:
        answers (list of str): answers to the question
        topic_detector (GDataTopicDetection): fitted topic detector
    """

    def __init__(self, answers, topic_detector):
        self.topic_detector = topic_detector
        nlp = topic_detector.data_preparation.nlp

        # sentence segmentation
        print("Sentence segmentation")
        answ_sents = []
        for idoc, doc in tqdm(enumerate(nlp.pipe(answers, n_process=1)),
                              total=len(answers)):
            for sent in doc.sents:
                answ_sents.append((idoc, sent.text))

        self.answ_sents = [sent for idoc, sent in answ_sents]

        # apply topic detection on the sentences
        data_prep = self.topic_detector.data_preparation
        print("Topic detection")
        self.sents_lems = data_prep.tokenize(self.answ_sents)
        self.topic_proba = topic_detector.tf_lda.transform(
            data_prep.tf_bow.transform(self.sents_lems))


    def create_titles(self):
        """For each topic, choose a chracteristic short sentence.
        """

        topics = self.topic_detector.get_topics_by_relevance(lambd=1)

        def get_title(itop):
            tdf = pd.DataFrame(
                columns=["topic proba", "nterms", "len"],
                data=list(zip(
                    self.topic_proba[:, itop],
                    [len(set(lem).intersection(set(topics[itop])))
                     for lem in self.sents_lems],
                    [len(ans) for ans in self.answ_sents]
                )))
            df_sort = tdf[(tdf["topic proba"] > 0.9) &
                          (tdf["len"] < TITLE_LEN_MAX) &
                          (tdf["len"] > TITLE_LEN_MIN)
                         ].sort_values(by="nterms", ascending=False)
            return self.answ_sents[df_sort.index[0]]

        return [get_title(itop) for itop, _ in enumerate(topics)]


    def group_sentences_by_topics(self):
        """For each topic, get the most representative sentences.

        Returns:
            pd.Series(list of int): indices of the sentences sorted by topic probability
        """
        maxtopics = pd.DataFrame({
            "proba_max": self.topic_proba.max(axis=1),
            "topic": self.topic_proba.argmax(axis=1)}).sort_values(
                by="proba_max", ascending=False)
        return maxtopics[maxtopics.proba_max > PURE_TOPIC_THRESH].reset_index().groupby(
            "topic")["index"].apply(list)


    def get_topic_summary(self, lidx, nb_sentences):
        """Return topic summary."""

        # doc2vec
        print("Doc2vec embedding")
        tagged_data = [TaggedDocument(d, [i])
                       for i, d in enumerate(self.sents_lems)]
        d2v_model = Doc2Vec(tagged_data, vector_size=50, epochs=40,
                            workers=3)

        topic_lems = np.array(self.sents_lems)[lidx]

        sents_emb = [d2v_model.infer_vector(lems) for lems in tqdm(topic_lems)]
        sim_mat = cosine_similarity(sents_emb[:MAX_SENTENCES_PAGERANK])
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        sub_idx = pd.Series(scores).sort_values(ascending=False).index[:nb_sentences]

        return np.array(self.answ_sents)[np.array(lidx)[sub_idx]].tolist()
