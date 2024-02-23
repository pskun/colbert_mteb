#encoding=utf8

import os
import sys
import argparse

from mteb import MTEB
from mteb.indexer import ColbertIndexer

def launch(args):
    indexer = ColbertIndexer(
        index_path=args.index_path,
        checkpoint_path=args.model_path,
    )

    evaluation = MTEB(task_langs=['zh'], tasks=[args.task_name])
    results = evaluation.run(
        indexer, output_folder=f"zh_results_{args.model_name}")
    return results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--index_path", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    args = parser.parse_args()
    
    print("start!")
    launch(args)
    print("end!")