# Author:  Fangdong Wu
# Time: 04.20.2021
# A funtion for VLAD  is an image retrieval method
# VLAD is used to extract appearance feature and compute similarity

import cv2
import numpy as np
from sklearn.cluster import KMeans

def Coslength(p,q, alpha):
    # 幂归一 + L2归一
    p = np.power(p, alpha)
    q = np.power(q, alpha)
    p = [i / np.sqrt(np.sum(np.power(p,2))) for i in p]
    q = [i / np.sqrt(np.sum(np.power(q,2))) for i in q]
    r = np.dot(p,q)/(np.linalg.norm(p)*(np.linalg.norm(q)))
    return r

def sift_discriptor(img):
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    sift_val_back = sift.compute(gray, kp)[1]
    return sift_val_back

def similarity(sift_val_back1, sift_val_back2, SeedNum):
    [r1, c1] = sift_val_back1.shape
    [r2, c2] = sift_val_back2.shape
    '''
    按垂直方向（行顺序）堆叠数组构成一个新的数组
    '''
    combine = np.vstack((sift_val_back1, sift_val_back2))
    '''
    distance[i]是每个特征向量到最近邻BoW的距离，
    BoW是聚类中心
    Cord
    '''
    distance = KMeans(n_clusters=SeedNum, random_state=0). \
        fit_transform(combine)
    bow = KMeans(n_clusters=SeedNum, random_state=0). \
        fit(combine).predict(combine)
    cord = KMeans(n_clusters=SeedNum, random_state=0). \
        fit(combine).cluster_centers_
    point_num = len(bow)
    center_num = len(cord)
    L1 = [0 for i in range(center_num)]
    L2 = [0 for i in range(center_num)]
    '''
    Normlization
    '''
    for i in range(0, point_num):
        for j in range(center_num):
            if i < r1:
                L1[bow[i]] += distance[i, j]
            else:
                L2[bow[i]] += distance[i, j]
    '''
    Compute Cosine Distance
    '''
    return Coslength(L1, L2, 0.8)

def VLAD(img1,img2,SeedNum):
    """
    :param img1:
    :param img2:
    :param SeedNum:cluster number
    :return: similarity between two pictures
    """
    sift = cv2.xfeatures2d.SIFT_create()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    '''
    Search Key Points
    '''
    kp1 = sift.detect(gray1, None)
    kp2 = sift.detect(gray2, None)
    '''
    Compute SIFT descriptors
    '''
    sift_val_back1 = sift.compute(gray1, kp1)[1]
    sift_val_back2 = sift.compute(gray2, kp2)[1]

    [r1, c1] = sift_val_back1.shape
    [r2, c2] = sift_val_back2.shape
    '''
    按垂直方向（行顺序）堆叠数组构成一个新的数组
    '''
    combine = np.vstack((sift_val_back1, sift_val_back2))
    # 如果特征点总数少于给定的k值
    if SeedNum > (r1+r2):
        SeedNum = int((r1+r2)/2)
    '''
    distance[i]是每个特征向量到最近邻BoW的距离，
    BoW是每个点label
    Cord是聚类中心
    '''
    km = KMeans(n_clusters=SeedNum, random_state=0).fit(combine)
    distance = km.transform(combine)
    bow = km.labels_
    cord = km.cluster_centers_
    point_num = len(bow)
    center_num = len(cord)
    L1 = [0 for i in range(center_num)]
    L2 = [0 for i in range(center_num)]
    '''
    Normlization
    '''
    for i in range(0, point_num):
        for j in range(center_num):
            if i < r1:
                L1[bow[i]] += distance[i,j]
            else:
                L2[bow[i]] += distance[i,j]
    '''
    Compute Cosine Distance
    '''
    return Coslength(L1, L2, 0.8)

if __name__=='__main__':
    imgname1 = '1.png'
    imgname2 = '2.png'
    imgname3 = '3.png'
    img1 = cv2.imread( imgname1 )
    img2 = cv2.imread( imgname2 )
    img3 = cv2.imread(imgname3)
    SeedNum = 15
    delta = []
    s1 = []
    s2 = []
    '''
    for i in range(50):
        similarity_1_2 = VLAD( img1, img2, SeedNum)
        similarity_1_3 = VLAD(img1,img3,SeedNum)
        s1.append(similarity_1_2)
        s2.append(similarity_1_3)
        delta.append( similarity_1_2 - similarity_1_3 )
        SeedNum = SeedNum + 1
    print(delta)
    print(s1)
    print(s2)
    plt.plot(range(len(delta)),delta)
    plt.show()

    for SeedNum in range(2,30):
        similarity_1_2,l1,l2 = VLAD(img1,img2,SeedNum)
        print(l1)
    '''