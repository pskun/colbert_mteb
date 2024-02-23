import os
import math
import json
from loguru import logger
from typing import Any, Dict, List, Optional
from .colbert.infra import Run, RunConfig, ColBERTConfig
from .colbert import Indexer, Searcher
    
class ColbertIndexer():

    def __init__(self, index_path, checkpoint_path):
        self.index_path = index_path
        self.checkpoint_path = checkpoint_path
        self.cids = []  

    def make_index(self, corpus_name: str, corpus: Dict[str, str]) -> None:
        # corpus: cid => doc
        index_path = os.path.join(self.index_path, corpus_name) # 索引路径
        # if index exist, no need to create a new index, just use the old one
        logger.info("pid:", os.getpid())
        if os.path.exists(index_path):
            logger.info(f'the index of {index_path} already exists, path: {index_path}')
            return
        # create new directory for index
        os.makedirs(index_path)
        # load and parse pdf data, then save the text trunks into json line file
        
        collection_path = os.path.join(index_path, "collection.jsonl")
        docs = []
        for cid, doc in corpus.items():
            self.cids.append(cid)
            docs.append(doc)
        fout = open(collection_path, mode="w")
        for doc in docs:
            print(json.dumps(doc, ensure_ascii=False), file=fout)
        fout.close()

        with Run().context(RunConfig(nranks=1, index_root=self.index_path)):  # nranks specifies the number of GPUs to use.  
            logger.info(f'configure indexer and start to create index of {corpus_name}....')
            config = ColBERTConfig(
                doc_maxlen=512,
                query_maxlen=120,
                query_token_id="[unused2]",
                nbits=4,
                similarity = "l2",
                kmeans_niters=4,
                )
            self.indexer = Indexer(checkpoint=self.checkpoint_path, config=config)
            self.indexer.configure(index_path=index_path)
            self.indexer.index(name=corpus_name, collection=collection_path, overwrite=True)
            logger.info(f"index of {corpus_name} created successfully.")
        return corpus_name

    def load_index(self, corpus_name: str, corpus: Dict[str, str]):
        """Load PLAID index from the paths given to the class and initialize a Searcher object."""
        for cid, doc in corpus.items():
            self.cids.append(cid)
        with Run().context(RunConfig(index_root=self.index_path, nranks=1)):
            self.searcher = Searcher(
                index=corpus_name,
                checkpoint=self.checkpoint_path
            )

        logger.info(f"Loaded {corpus_name} index")

    def query(self, query: str, topk:int=1) -> Dict:
        pids, ranks, scores = self.searcher.search(query, k=topk)
        # logger.info(f'pids:{pids}, ranks:{ranks}, scores:{scores}')
        
        probs = [math.exp(score) for score in scores]
        probs = [prob / sum(probs) for prob in probs]
        topk_list = []
        for pid, rank, score, prob in zip(pids, ranks, scores, probs):
            text = self.searcher.collection[pid]
            cid = self.cids[pid]
            d = {'text': text, 'cid': cid, 'rank': rank, 'score': score, 'prob': prob}
            topk_list.append(d)
        topk_list = list(sorted(topk_list, key=lambda p: p['score'], reverse=True))

        output_dict = {"query" : query, "topk": topk_list}
        return output_dict
