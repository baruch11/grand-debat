import pyLDAvis
import pyLDAvis.sklearn
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import warnings
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

warnings.simplefilter('ignore')
pyLDAvis.enable_notebook()


def get_themes(responses, n_topics=20, max_iter=100, stop_words=None):
    """
    Get the main themes of selected question and save them in the following
    file: pyLDAVIS_tf.html

    Args:
        responses: list of documents (strings)
    """
    tf_vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                                    stop_words=stop_words)
    dtm_tf = tf_vectorizer.fit_transform(responses)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0,
                                    n_jobs=-1, verbose=1, max_iter=max_iter,
                                    perp_tol=0.5, evaluate_every=4)
    lda.fit(dtm_tf)

    pyLDAVIS_tf = pyLDAvis.sklearn.prepare(lda, dtm_tf, tf_vectorizer)
    pyLDAvis.save_html(pyLDAVIS_tf, 'pyLDAVIS_tf.html')

    return tf_vectorizer, lda


def get_topic_by_relevance(vectorizer, lda, lambd, n_topic, n_top=10):
    """Get the topics from a sklearn countvectorizer and lda sorted by relevance
    see def of relevance in pyLDAvis article
    https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf

    Args:
        vectorizer: sklearn CountVectorizer
        lda: sklearn LatentDirichletAllocation
        lambd: float, lambda in pyLDAvis paper
        n_topic: int, indice of the topic
        n_top: nomber of top words for topic representation
    Returns:
        list of strings, topic words set representation
    """
    feature_names = vectorizer.get_feature_names_out()
    pw = lda.components_.sum(axis=0) / lda.components_.sum()
    pw_t = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    relevance = lambd * pw_t + (1 - lambd) * pw_t / pw
    return feature_names[np.argsort(relevance[n_topic, :])[:-n_top:-1]]


def lemmatize_answers(answs, nlp):
    """Lemmatize answers
    and keep only nouns and adj

    Args:
        answs: list of strings representing answers
        nlp: spacy pipeline
    Returns:
        new clean list of strings
    """
    answ_lems = []
    disable_pos = ['PUNCT', 'AUX', 'DET', 'ADP', 'NUM', 'CCONJ', 'PRON', 'ADV',
                   'VERB']
    with nlp.select_pipes(disable=["ner", "parser"]):
        for doc in tqdm(nlp.pipe(answs, n_process=1), total=len(answs)):
            answ_lems.append(" ".join([w.lemma_ for w in doc if w.pos_ not in
                                       disable_pos]))
    return answ_lems


def compute_coherence_vs_ntopics(responses, num_topics=[], stop_words=None,
                                 lambda_coh=0):
    """Try many numbers of topics and return lda coherence scores
    Args:
        responses (list of str): vector of tokens separated by space
          representing one reponse
        num_topics: list of int, vector of topics numbers
    Returns:
        v_coh: list of int, coherence vector
        l_topics: list of topics, each topics is a list of token
    """
    tf_vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                                    stop_words=stop_words)
    dtm_tf = tf_vectorizer.fit_transform(responses)

    # compute topics for different n_topics
    l_topics = []
    for n_topics in num_topics:
        print("compute lda for {} topics".format(n_topics))
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0,
                                        n_jobs=2, verbose=0, max_iter=100,
                                        perp_tol=0.5, evaluate_every=4)
        lda.fit(dtm_tf)
        # retrieve topics
        topics = []
        for ii in range(lda.components_.shape[0]):
            topic = get_topic_by_relevance(
                tf_vectorizer, lda, lambd=lambda_coh, n_topic=ii).tolist()
            topics.append(topic)

        l_topics.append(topics)

    # compute c_v coherence for each lda
    corpus = [doc.split(' ') for doc in responses]
    dictionary = Dictionary(corpus)

    v_coh = []
    for topics in l_topics:
        cm = CoherenceModel(topics=topics, texts=corpus, dictionary=dictionary,
                            coherence='c_v')
        coherence = cm.get_coherence()  # get coherence value
        v_coh.append(coherence)

    # retrieve last idx before decrease
    vdec = np.nonzero(np.diff(v_coh) < 0)[0]
    if len(vdec) < 1:
        best_n = num_topics[len(v_coh)-1]
    else:
        best_n = num_topics[vdec[0]]

    print("Best coherence achieved for n_topics = {}".format(best_n))

    return v_coh, l_topics, best_n
