#coding:utf-8
import sys
sys.path.extend(["./textcnn","./BM25","./Bool","./kmeans","./SIF"])
from textcnn_predict import TextcnnPredict
from km_recall import KmRecall
from bm25_recall import Bm25Recall
from bool_recall import BoolRecall
from SIF_rank import SIFRank
import time


class Rank(object):
    
    def __init__(self):
        self.textcnn = TextcnnPredict()
        self.km_recall = KmRecall()
        self.bm_recall = Bm25Recall()
        self.bool_recall = BoolRecall()
        self.sif_rank = SIFRank()
        
    
    def _recall_topn(self,query,topn=10):
        
        start = time.time()
        
        ques_type = self.textcnn.predict(query)
        
        median = time.time()
        
        topn_km = self.km_recall.recall(query,ques_type)
        topn_bm = self.bm_recall.recall(query,ques_type)
        topn_bool = self.bool_recall.recall(query,ques_type)
        
        end = time.time()
        
        print("\nTime usage for binary classify is {}\n".format(median - start))
        print("\nTime usage for recall is {}\n".format(end - median))
        
        topn_recall = list(set(topn_km + topn_bm + topn_bool))
        
        return topn_recall
    
    def get_top_one(self,query,topn=10):
        
        topn_recall = self._recall_topn(query)
        
        start = time.time()
        top_one = self.sif_rank.rank(query,topn_recall,threshold=0.3)
        end = time.time()
        
        print("\nTime usage for rank is {}\n".format(end - start))
        
        return top_one
    
if __name__ == "__main__":
    
    questions = ["办信用卡需要准备哪些证件","社保业务","存单业务","查询外汇汇率","你家乡在哪里","今天的风儿甚是喧嚣","你有男朋友吗","你觉得自己长得怎么样"]

    ranker = Rank()
    
    for query in questions:
        
        match_ques = ranker.get_top_one(query)
        
        print("\nThe question matched is %s \n" % str(match_ques))
        
    
    




    
