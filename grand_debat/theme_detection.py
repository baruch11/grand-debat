import pyLDAvis
import pyLDAvis.sklearn
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.simplefilter('ignore')
pyLDAvis.enable_notebook()


def get_themes(data_theme_response_dict, selected_question, n_topics=20, max_iter=100):
    """
    Get the main themes of selected question and save them in the following
    file: pyLDAVIS_tf.html
    """
    tf_vectorizer = CountVectorizer(min_df=5, max_df=0.9)
    dtm_tf = tf_vectorizer.fit_transform(data_theme_response_dict[selected_question])

    lda_tf = LatentDirichletAllocation(n_components=n_topics, random_state=0, n_jobs=-1, verbose=1, max_iter=max_iter, perp_tol=0.5, evaluate_every=4)
    lda_tf.fit(dtm_tf)

    pyLDAVIS_tf = pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)
    pyLDAvis.save_html(pyLDAVIS_tf, 'pyLDAVIS_tf.html')

    return tf_vectorizer, lda_tf


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
