import logging
from time import time
from typing import Dict, List

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings
import os

from .AbsTask import AbsTask
from ..evaluation.evaluators import RetrievalEvaluator

logger = logging.getLogger(__name__)

DRES_METHODS = ["encode_queries", "encode_corpus"]

class AbsTaskRetrieval(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = Dict[id, Dict[str, str]] #id => dict with document datas like title and text
    self.queries = Dict[id, str] #id => query
    self.relevant_docs = List[id, id, score]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def is_dres_compatible(model):
        for method in DRES_METHODS:
            op = getattr(model, method, None)
            if not (callable(op)):
                return False
        return True

    def evaluate(
        self,
        indexer,
        split="test",
        batch_size=128,
        corpus_chunk_size=None,
        score_function="cos_sim",
        **kwargs
    ):
        if not self.data_loaded:
            self.load_data()

        corpus, queries, relevant_docs = self.corpus[split], self.queries[split], self.relevant_docs[split]

        corpus_name = self.description["name"]
        if "build_index" in kwargs and kwargs["build_index"] is True:
            indexer.make_index(corpus_name, corpus)
            scores = {}
        else:
            indexer.load_index(corpus_name, corpus)
            evaluator = RetrievalEvaluator(queries, corpus, relevant_docs)
            scores = evaluator.compute_metrics(indexer)
        return scores
