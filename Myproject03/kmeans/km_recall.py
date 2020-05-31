#coding:utf-8
import pandas as pd
import numpy as np
import pickle,os,jieba,re
from km_config import KmConfig
from scipy.spatial.distance import cosine


config = KmConfig()

""" 加载pickle对象 """ 
def load_pickle(file_path):
    with open(file_path,'rb') as f:
        s = pickle.load(f)
    return s

def transfer_char(line):
    line = re.sub(
            "[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+", '',str(line))
    return [char.strip() for char in line if char.strip()]

class KmRecall(object):
    def __init__(self):
        self.km_busi = load_pickle(os.path.join(config.km_dir,"km_busi.model"))
        self.km_chat = load_pickle(os.path.join(config.km_dir,"km_chat.model"))
        self.ques_emb_busi = load_pickle(os.path.join(config.km_dir,"km_cls_busi.pickle"))
        self.ques_emb_chat = load_pickle(os.path.join(config.km_dir,"km_cls_chat.pickle"))
        self.word_emb_busi = load_pickle(os.path.join(config.w2v_dir,"word2emb_busi.pickle"))
        self.word_emb_chat = load_pickle(os.path.join(config.w2v_dir,"word2emb_chat.pickle"))
        
        
    def _sentence_embs(self,words,class_):
        
        if class_ == "busi":
            word_emb_dic = self.word_emb_busi
        elif class_ == "chat":
            word_emb_dic = self.word_emb_chat
        
        words_emb = [word_emb_dic[word] for word in words if word in word_emb_dic]
        ques_emb = np.mean(words_emb,axis=0)
        
        return ques_emb
    
    def _km_classify(self,emb,class_):
        
        if class_ == "busi":
            km = self.km_busi
        elif class_ == "chat":
            km = self.km_chat
            
        class_new = km.predict([emb])[0]
        
        return class_new
            
        
    """ 得到语义最相似的5个问题 """
    def _get_topk(self,query_emb,class_,class_new,topn=10):
        
        if class_ == "busi":
            ques_emb_dic = self.ques_emb_busi
        elif class_ == "chat":
            ques_emb_dic = self.ques_emb_chat
            
        all_ques = ques_emb_dic[class_new]
        
        cos_similars = [1 - cosine(query_emb,emb) for emb,_ in all_ques]
        questions = [ques for _,ques in all_ques]
        top_n = np.argsort(cos_similars)[::-1][:topn]
        
        return [questions[i] for i in top_n]
        
        
    """ 大于阈值，则到最相似的5个问题，否则调用爬虫 """
    def recall(self, query, ques_type):
        
        chars = transfer_char(query)
        ques_emb = self._sentence_embs(chars, ques_type)
        class_new = self._km_classify(ques_emb, ques_type)
        topn_ques = self._get_topk(ques_emb, ques_type, class_new)
        return topn_ques
        
        
if __name__ == "__main__":

    model = KmRecall() 
    questions = ["办信用卡需要准备哪些证件","社保业务","存单业务","查询外汇汇率","你家乡在哪里","今天的风儿甚是喧嚣","你有男朋友吗","你觉得自己长得怎么样"]
    ques_types = ["busi"] * 4 + ["chat"] * 4
    
    for ques,type_ in zip(questions,ques_types):
        chars = transfer_char(ques)
        print(model.recall(chars,type_))