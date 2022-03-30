"""Topic detection"""

import warnings
import pyLDAvis
import pyLDAvis.sklearn
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import spacy


pyLDAvis.enable_notebook()
warnings.simplefilter('ignore')

class GDebatDataPreparation:
    """data preparation for theme detection

    The role of this class is to provide data as inputs of the theme detection
    algorithm

    parameters :
        answers (list of str): corpus of answers used
        for countvectorizer fitting
        stopsentence (str): if not None, stopsentence are added to the default list
           after tokenization. (Try to put the question for e.g.)
    """
    def __init__(self, answers, n_process=1, stopsentence=""):
        print("load spacy pipeline")
        self.nlp = spacy.load("fr_core_news_md", exclude=["ner"])

        # tokenization
        print("tokenizing datas")
        self.answ_lems = self.tokenize(answers, n_process)
        add_stopwords = set(self.tokenize([stopsentence])[0])
        if len(add_stopwords)>0:
            print("Additional stopwords ", add_stopwords)

        # fit countVecorizer
        def dummy(doc):
            return doc
        self.tf_bow = CountVectorizer(
            min_df=5, max_df=0.9, tokenizer=dummy, preprocessor=dummy,
            stop_words=self.nlp.Defaults.stop_words.union(add_stopwords))
        print("fitting countvectorizer")
        self.answ_bow = self.tf_bow.fit_transform(self.answ_lems)
        print("data preparation done")

    def tokenize(self, answs, n_process=1):
        """Lemmatize answers.
        and keep only nouns and adj

        Args:
            answs (list of str): strings representing answers
        Returns:
            list of list of words
        """
        enable_pos = ["NOUN", "ADJ"]
        with self.nlp.select_pipes(disable=["parser"]):
            answ_lems = [
                [w.lemma_ for w in doc if w.pos_ in enable_pos]
                for doc in tqdm(self.nlp.pipe(answs, n_process=n_process),
                                total=len(answs))]

        return answ_lems

    def get_internal_lemmatized_datas(self):
        return self.answ_lems

    def get_internal_bow_datas(self):
        return self.answ_bow

    def prepare_data(self, answers):
        """Prepare data for theme detection.
        Args:
            answers (list of str): strings representings answers
        Returns (spase matrix):
            bag of words after lemmatization
        """
        return self.tf_bow.transform(self.tokenize(answers))


class GDebatTopicDetection:
    def __init__(self, data_preparation, n_topics=20, max_iter=100, verbose=0):
        self.data_preparation = data_preparation
        self.tf_lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=0, n_jobs=-1, verbose=verbose,
            max_iter=max_iter, perp_tol=0.5, evaluate_every=4)

    def compute_topic_detection(self, data_bow=None, data_lems=None,
                                LDAVis=False):
        lda_input = data_bow
        if data_lems:
            lda_input = self.data_preparation.tf_bow.transform(data_lems)
        self.tf_lda.fit(lda_input)

        if LDAVis:
            pyLDAVIS_tf = pyLDAvis.sklearn.prepare(
                self.tf_lda, lda_input, self.data_preparation.tf_bow)
            pyLDAvis.save_html(pyLDAVIS_tf, 'pyLDAVIS_tf.html')
            print("LDA visualization in pyLDAVIS_tf.html")

    def get_topics_by_relevance(self, lambd, n_top=5):
        """Get the topics from a sklearn countvectorizer and lda sorted by relevance
        see def of relevance in pyLDAvis article
        https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf

        Args:
            lambd: float, lambda in pyLDAvis paper
            n_top: nomber of top words for topic representation
        Returns:
            list of strings, topic words set representation
        """
        lda_components = self.tf_lda.components_

        pw = lda_components.sum(axis=0) / lda_components.sum()
        pw_t = lda_components / lda_components.sum(axis=1)[:, np.newaxis]
        relevance = lambd * pw_t + (1 - lambd) * pw_t / pw

        feature_names = self.data_preparation.tf_bow.get_feature_names_out()
        ret = []
        for n_topic in range(lda_components.shape[0]):
            ret.append(feature_names[np.argsort(
                relevance[n_topic, :])[:-n_top-1:-1]].tolist())
        return ret


def compute_coherence_vs_ntopics(responses, data_prep, num_topics,
                                 lambda_coh=0):
    """Try many numbers of topics and return lda coherence scores
    Args:
        responses (list of list of str): matrix of lemmatized tokens
        data_prep (GDebatDataPreparation): data preparation pipeline
        num_topics: list of int, vector of topics numbers
    Returns:
        v_coh: list of int, coherence vector
        l_topics: list of topics, each topics is a list of token
    """

    # compute c_v coherence for each lda
    dictionary = Dictionary(responses)

    # compute topics for different n_topics
    l_topics, v_coh = [], []
    for n_topics in num_topics:
        print(f"compute lda for {n_topics} topics")
        topd = GDebatTopicDetection(data_prep, n_topics=n_topics)
        topd.compute_topic_detection(data_lems=responses)
        # retrieve topics
        topics = topd.get_topics_by_relevance(lambd=lambda_coh)

        l_topics.append(topics)

        cmodel = CoherenceModel(topics=topics, texts=responses,
                                dictionary=dictionary, coherence='c_v')
        coherence = cmodel.get_coherence()  # get coherence value
        v_coh.append(coherence)
        print(f"Lda done, coherence {coherence:.3f}")

    # retrieve last idx before decrease
    vdec = np.nonzero(np.diff(v_coh) < 0)[0]
    if len(vdec) < 1:
        best_n = num_topics[len(v_coh)-1]
    else:
        best_n = num_topics[vdec[0]]

    print(f"Best coherence achieved for n_topics = {best_n}")

    return v_coh, l_topics, best_n
