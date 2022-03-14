from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
from grand_debat.theme_detection import GDebatTopicDetection


def create_titles(answers, topic_detector):
    """
    Applies clustering algorithm in order to get most characteristic themes

    Parameters
    ----------
    answers (list of str): answers to the question
    topic_detector (GDataTopicDetection): fitted topic detector
    """
    # sentence segmentation
    nlp = topic_detector.data_preparation.nlp
    answ_sents = []
    print("Sentence segmentation")
    for doc in tqdm(nlp.pipe(answers, n_process=1), total=len(answers)):
        for sent in doc.sents:
            answ_sents.append(sent.text)
    answ_sents = [sent for sent in answ_sents
                  if len(sent) < 200 and len(sent) > 100]

    # apply topic detection on the sentences
    data_prep = topic_detector.data_preparation
    print("Data preparation")
    sents_lems = data_prep.tokenize(answ_sents)
    print("LDA transformation")
    topic_proba = topic_detector.tf_lda.transform(
        data_prep.tf_bow.transform(sents_lems))

    topics = topic_detector.get_topics_by_relevance(lambd=1)

    def get_title(itop):
        """  """
        df = pd.DataFrame(
            columns=["topic proba", "nterms"],
            data=list(zip(topic_proba[:, itop],
                          [len(set(lem).intersection(set(topics[itop]))) for
                           lem in sents_lems])),
                    )
        df_sort = df[df["topic proba"] > 0.9].sort_values(by="nterms",
                                                          ascending=False)
        return answ_sents[df_sort.index[0]]

    return [get_title(itop) for itop, _ in enumerate(topics)]


def apply_page_rank_algorithm(clean_sentences, sentences_paragraph, word_embeddings, sn):
    """
    Apply the page rank algorithm over the sentence graph to get the
    text summarization
    """
    sentences_summary = [x for i, x in enumerate(clean_sentences) if sentences_paragraph.get(i, -1)==1]
    sentences_summary_emb = []
    for i in sentences_summary:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentences_summary_emb.append(v)
    sim_mat = cosine_similarity(sentences_summary_emb)
    nx_graph = nx.from_numpy_array(sim_mat)
    try:
        scores = nx.pagerank(nx_graph)
    except:
        scores = nx.pagerank_numpy(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences_summary)), reverse=True)
    for i in range(sn):
        print('•', ranked_sentences[i][1], '\n')
