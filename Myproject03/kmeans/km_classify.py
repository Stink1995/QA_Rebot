#coding:utf-8
import pandas as pd
import numpy as np
from gensim.models import word2vec
import jieba,os,re
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pickle
from km_config import KmConfig
from collections import defaultdict
from sklearn.cluster import KMeans
from numpy import math
from itertools import chain

np.random.seed(10)
config = KmConfig()

""" 保存为pickle对象 """ 
def save_pickle(s,file_path):
    with open(file_path,'wb') as f:
        pickle.dump(s,f, protocol=2)

""" 一:清洗文本并划分数据集 """
def load_corpus(config):
    
    """ 读取数据 """
    print("\nLoading the dataset ... \n")
    corpus_xls = pd.ExcelFile(config.corpus_path)
    business_df = corpus_xls.parse("business_question")
    chatting_df = corpus_xls.parse("chatting_question")
    
    """ 进行分词 """
    business_df["question_seg"] = business_df["question"].apply(transfer_char)
    chatting_df["question_seg"] = chatting_df["question"].apply(transfer_char)
    business_df["answer_seg"] = business_df["answer"].apply(transfer_char)
    chatting_df["answer_seg"] = chatting_df["answer"].apply(transfer_char)    
    
    business_df.dropna(inplace=True)
    chatting_df.dropna(inplace=True)    
    
    """ train wor2vec """
    busi_ques_seg = business_df["question_seg"].tolist()
    busi_ans_seg = business_df["answer_seg"].tolist()
    busi_corpus = [ques + ans for ques,ans in zip(busi_ques_seg, busi_ans_seg)]
    busi_ques = business_df["question"].tolist()
    
    w2v_busi = train_w2v(busi_corpus)
    busi_ques_emb = calcu_weighted_emb(w2v_busi, busi_ques_seg, busi_corpus, config, "busi")

    chat_ques_seg = chatting_df["question_seg"].tolist()
    chat_ans_seg = chatting_df["answer_seg"].tolist()
    chat_corpus = [ques + ans for ques,ans in zip(chat_ques_seg, chat_ans_seg)]
    chat_ques = chatting_df["question"].tolist()
    
    w2v_chat = train_w2v(chat_corpus)
    chat_ques_emb = calcu_weighted_emb(w2v_chat, chat_ques_seg, chat_corpus, config, "chat")
    
    return busi_ques_emb, busi_ques, chat_ques_emb, chat_ques


""" 进行文本清洗，并用jieba分词 """
def clean_seg(line):
    line = re.sub(
            "[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+", '',str(line))
    words = jieba.lcut(line, cut_all=False)
    if not words:
        return np.nan
    return words

def transfer_char(line):
    line = re.sub(
            "[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+", '',str(line))
    if not line:
        return np.nan
    return [char.strip() for char in line if char.strip()]
    

def calcu_weighted_emb(w2v, queses,corpus,config, type_):
    
    idf_dic = calcu_idf(corpus)
    
    word_emb_wei = {}
    for w,idf in idf_dic.items():
        word_emb_wei[w] = idf * w2v.wv[w]
        
    sentence_embs = [[word_emb_wei[w] for w in ques] for ques in queses]
    sentence_embs = [np.average(emb, axis=0) for emb in sentence_embs]
        
    save_path = os.path.join(config.w2v_dir,"word2emb_%s.pickle" % type_)
    save_pickle(word_emb_wei, save_path)
    
    return sentence_embs

def calcu_idf(docs):
    
    words = set(chain.from_iterable(docs))
    
    df = {}
    for w in words:
        for doc in docs:
            if w in doc:
                df[w] = df.get(w,0) + 1
         
    idf_dic = {}
    for word, freq in df.items():
        idf_dic[word] = math.log( (len(docs)-freq + 0.5) / (freq + 0.5) )
        
    return idf_dic

def train_w2v(corpus):
    
    model = word2vec.Word2Vec(corpus,
                              size      = 50, 
                              min_count = 1, 
                              window    = 5,
                              workers   = 4,
                              sg        = 1,
                              iter      = 5)

    
    return model


def train_kmeans(ques_embs,config, type_):
    
    """ train kmeans"""
    silhouette_all = []
    for n in range(10,200,5):
        km = KMeans(n_clusters=n,max_iter=500)
        km.fit(ques_embs)
        ques_class = km.predict(ques_embs)
    
        """ 计算聚类的轮廓系数 """
        sil = silhouette_score(ques_embs, ques_class, metric='euclidean')
        print("{}类数量为{}时的轮廓系数为: {}\n".format(type_,n,sil)) 

        silhouette_all.append(sil)
        
    class_ = list(range(10,200,5))
    sil_max = max(silhouette_all)
    index = silhouette_all.index(sil_max)
    n_cluster = class_[index]
    print("%s 类轮廓系数最高的个数为 %d,对应的轮廓系数为：%.4f\n" % (type_, n_cluster,sil_max))  
    
    visualize_sil(silhouette_all, class_,config,type_)
    
    return n_cluster
    
    
def visualize_sil(sil_all,class_,config,type_):
    
    print("对个数和轮廓系数的关系进行可视化 ...\n")
    plt.plot(class_,sil_all,'r',label=type_)
    plt.title("Silhouette values and cluster numbers")
    plt.xlabel('cluster numbers')
    plt.ylabel('silhouette value') 
    plt.legend()
    save_path = os.path.join(config.km_dir,'silhouette_{}.png'.format(type_))
    plt.savefig(save_path)
    plt.clf()
            
    
def retrain_kmeans(n,ques_embs,queses,config, type_):
    
    assert len(ques_embs) == len(queses)
    
    """ train kmeans"""
    km = KMeans(n_clusters=n,max_iter=500)
    km.fit(ques_embs)
    
    ques_class = km.predict(ques_embs)
    
    ques_dic = defaultdict(list)
    for class_, emb, ques in zip(ques_class,ques_embs,queses):
        ques_dic[class_].append((emb,ques))
    
    km_path = os.path.join(config.km_dir,'km_{}.model'.format(type_))
    result_path = os.path.join(config.km_dir,'km_cls_{}.pickle'.format(type_))
    
    save_pickle(km, km_path)
    save_pickle(ques_dic,result_path)
    

def main(config):
    
    busi_ques_emb, busi_ques, chat_ques_emb, chat_ques = load_corpus(config)
    
    #busi_n_cluster = train_kmeans(busi_ques_emb, config, "busi")
    #chat_n_cluster = train_kmeans(chat_ques_emb, config, "chat")
    
    busi_n_cluster = 20
    chat_n_cluster = 50
     
    retrain_kmeans(busi_n_cluster, busi_ques_emb, busi_ques, config, "busi")
    retrain_kmeans(chat_n_cluster, chat_ques_emb, chat_ques, config, "chat")
    
    
if __name__ == "__main__":
    
    main(config)
    
    