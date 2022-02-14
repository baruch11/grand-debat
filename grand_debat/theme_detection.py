import pyLDAvis
import pyLDAvis.sklearn
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import warnings
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
