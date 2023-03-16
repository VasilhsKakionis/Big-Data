import sys
import pandas 
import random
import numpy
import csv
import matplotlib.pyplot as plt

userList = {}
movieList = {}
movieMap = {}
K=0
N=0
n=40

def main():
    #file_name = "ratings.csv"
    file_name = sys.argv[1]
    data = pandas.read_csv(file_name)
    calculateLists(data,file_name)
    SIG = calculateMinHash(file_name)
    if file_name == "ratings_100users.csv":
        test100users(SIG)
    else:
        test605users(SIG)
        
def test100users(signatures):
    movieIdList=list(range(1, 22))
    movieIdList.remove(9)
    
    s=0.25
    SIG = signatures
    
    pairs=[]
    truePositivePairs=[]
    
    #*********MINHASHING TEST*************
    
    print("~~~~~~MINHASHING TEST~~~~~~") 
    falsePositives_test1=[0,0,0,0,0,0,0,0]
    falseNegatives_test1=[0,0,0,0,0,0,0,0]
    PRECISION_test1=[0,0,0,0,0,0,0,0]
    RECALL_test1=[0,0,0,0,0,0,0,0]
    F1_test1=[0,0,0,0,0,0,0,0]
    
    nList =[5, 10, 15, 20, 25, 30, 35, 40]    
    
    for i in movieIdList:
        for j in range(i+1,22):
            if j!=9:
                t=calculateJaccardSimilarity(movieList[i],movieList[j])
                pairs.append((i,j))
                if t>=s :
                    truePositivePairs.append((i,j))
                for n in range(len(nList)):
                    if signatureSimilarity(i,j,nList[n],SIG)<s and t>=s:
                        falsePositives_test1[n]+=1
                    if signatureSimilarity(i,j,nList[n],SIG)>=s and t<s:
                        falseNegatives_test1[n]+=1
                        
    for i in range(len(nList)):
        PRECISION_test1[i]=len(truePositivePairs)/(len(truePositivePairs)+falsePositives_test1[i])
        RECALL_test1[i]=len(truePositivePairs)/(len(truePositivePairs)+falseNegatives_test1[i])
        F1_test1[i]=2*RECALL_test1[i]*PRECISION_test1[i]/(RECALL_test1[i]+PRECISION_test1[i])
        
     
    d = {'n': nList, 'PRECISION': PRECISION_test1,'RECALL': RECALL_test1,'F1':F1_test1,'false-positives':falsePositives_test1,'false-negatives':falseNegatives_test1}
    df_test1 = pandas.DataFrame(data=d)
    print(df_test1)
    #plt.figure(); df_test1.plot(x = 'n', y = 'PRECISION');
    #plt.savefig('Precision_MINHASH_ratings_100users.png')
    #plt.figure(); df_test1.plot(x = 'n', y = 'RECALL');
    #plt.savefig('Recall_MINHASH_ratings_100users.png')
    #plt.figure(); df_test1.plot(x = 'n', y = 'F1');
    #plt.savefig('F1_MINHASH_ratings_100users.png')
    #plt.figure(); df_test1.plot(x = 'n', y = 'false-negatives');
    #plt.savefig('false-negatives_MINHASH_ratings_100users.png')
    #plt.figure(); df_test1.plot(x = 'n', y = 'false-positives');
    #plt.savefig('false-positives_MINHASH_ratings_100users.png')
    
    print()
    
    #*********LSH TEST*********
    print("~~~~~~LSH TEST~~~~~~") 
    truePositives_test2=[0,0,0,0,0,0]
    falsePositives_test2=[0,0,0,0,0,0]
    falseNegatives_test2=[0,0,0,0,0,0]
    PRECISION_test2=[0,0,0,0,0,0]
    RECALL_test2=[0,0,0,0,0,0]
    F1_test2=[0,0,0,0,0,0]
    rlist=[2, 4, 5, 8, 10, 20]
    blist=[20, 10, 8, 5, 4, 2]
    slist=[0,0,0,0,0,0]

    for n in range(len(rlist)):
        slist[n]= (1/blist[n])**(1/rlist[n])
        r=rlist[n]
        b=blist[n]
        LSH_pairs=LSH(len(SIG),b,r,SIG)
        new_pairs2=[]
        
        #LSH pairs containing only the 20 first movies
        #LSH_pairs=[i for i in LSH_pairs if (i[0] in movieIdList and i[1] in movieIdList)]
        LSH_pairs=[i for i in LSH_pairs]
        
        lst= set(LSH_pairs).intersection(set(truePositivePairs))
        truePositives_test2[n]=len(lst)
        falsePositives_test2[n]=len(set(LSH_pairs)-set(lst))
        falseNegatives_test2[n]=len(set(truePositivePairs)-set(lst))
        #print(LSH_pairs,truePositivePairs)
        #print(len(LSH_pairs),truePositives_test2[n],falsePositives_test2[n])
        
    for i in range(len(rlist)):
        if truePositives_test2[i]!= 0 or falsePositives_test2[i]!=0:
            PRECISION_test2[i]=truePositives_test2[i]/(truePositives_test2[i]+falsePositives_test2[i])
        if truePositives_test2[i]!= 0 or falseNegatives_test2[i]!=0:
            RECALL_test2[i]=truePositives_test2[i]/(truePositives_test2[i]+falseNegatives_test2[i])
        if RECALL_test2[i]!= 0 or PRECISION_test2[i]!=0:
            F1_test2[i]=2*RECALL_test2[i]*PRECISION_test2[i]/(RECALL_test2[i]+PRECISION_test2[i])
            
    d = {'r': rlist,'b':blist,'s':slist, 'PRECISION': PRECISION_test2,'RECALL': RECALL_test2,'F1':F1_test2,'false-positives':falsePositives_test2,'false-negatives':falseNegatives_test2}
    df_test2 = pandas.DataFrame(data=d)
    print(df_test2)
    #plt.figure(); df_test2.plot(x = 's', y = 'PRECISION');
    #plt.savefig('Precision_LSH_ratings_100users.png')
    #plt.figure(); df_test2.plot(x = 's', y = 'RECALL');
    #plt.savefig('Recall_LSH_ratings_100users.png')
    #plt.figure(); df_test2.plot(x = 's', y = 'F1');
    #plt.savefig('F1_LSH_ratings_100users.png')
    #plt.figure(); df_test2.plot(x = 's', y = 'false-negatives');
    #plt.savefig('false-negatives_LSH_ratings_100users.png')
    #plt.figure(); df_test2.plot(x = 's', y = 'false-positives');
    #plt.savefig('false-positives_LSH_ratings_100users.png')
    
def test605users(signatures):
    movieIdList=list(range(1, 111))
    movieIdList.remove(33)
    movieIdList.remove(35)
    movieIdList.remove(37)
    movieIdList.remove(51)
    movieIdList.remove(56)
    movieIdList.remove(59)
    movieIdList.remove(67)
    movieIdList.remove(84)
    movieIdList.remove(90)
    movieIdList.remove(91)
    movieIdList.remove(98)
    movieIdList.remove(109)
    notInlist=[33,35,37,51,56,59,67,84,90,91,98,109]
    
    s=0.25
    SIG = signatures
    
    pairs=[]
    truePositivePairs=[]
    
    #*********MINHASHING TEST*************
    falsePositives_test1=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    falseNegatives_test1=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PRECISION_test1=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    RECALL_test1=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F1_test1=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    nList =[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80] 
    
    for i in movieIdList:
        for j in range(i+1,111):
            if j not in notInlist:
                t=calculateJaccardSimilarity(movieList[i],movieList[j])
                pairs.append((i,j))
                if t>=s:
                    truePositivePairs.append((i,j))
                for n in range(len(nList)):
                    if signatureSimilarity(i,j,nList[n],SIG)<s and t>=s:
                        falsePositives_test1[n]+=1
                    if signatureSimilarity(i,j,nList[n],SIG)>=s and t<s:
                        falseNegatives_test1[n]+=1
    
    
    
    for i in range(len(nList)):
        PRECISION_test1[i]=len(truePositivePairs)/(len(truePositivePairs)+falsePositives_test1[i])
        RECALL_test1[i]=len(truePositivePairs)/(len(truePositivePairs)+falseNegatives_test1[i])
        F1_test1[i]=2*RECALL_test1[i]*PRECISION_test1[i]/(RECALL_test1[i]+PRECISION_test1[i])
        
     
    d = {'n': nList, 'PRECISION': PRECISION_test1,'RECALL': RECALL_test1,'F1':F1_test1,'false-positives':falsePositives_test1,'false-negatives':falseNegatives_test1}
    df_test1 = pandas.DataFrame(data=d)
    print(df_test1)
    #plt.figure(); df_test1.plot(x = 'n', y = 'PRECISION');
    #plt.savefig('Precision_MINHASH_ratings_605users.png')
    #plt.figure(); df_test1.plot(x = 'n', y = 'RECALL');
    #plt.savefig('Recall_MINHASH_ratings_605users.png')
    #plt.figure(); df_test1.plot(x = 'n', y = 'F1');
    #plt.savefig('F1_MINHASH_ratings_605users.png')
    #plt.figure(); df_test1.plot(x = 'n', y = 'false-negatives');
    #plt.savefig('false-negatives_MINHASH_ratings_605users.png')
    #plt.figure(); df_test1.plot(x = 'n', y = 'false-positives');
    #plt.savefig('false-positives_MINHASH_ratings_605users.png')
    
    print()
                             
    #*********LSH TEST*********
    print("~~~~~~LSH TEST~~~~~~") 
    truePositives_test2=[0,0,0,0,0,0]
    falsePositives_test2=[0,0,0,0,0,0]
    falseNegatives_test2=[0,0,0,0,0,0]
    PRECISION_test2=[0,0,0,0,0,0]
    RECALL_test2=[0,0,0,0,0,0]
    F1_test2=[0,0,0,0,0,0]
    rlist=[2, 4, 5, 8, 10, 20]
    blist=[20, 10, 8, 5, 4, 2]
    slist=[0,0,0,0,0,0]

    for n in range(len(rlist)):
        slist[n]= (1/blist[n])**(1/rlist[n])
        r=rlist[n]
        b=blist[n]
        LSH_pairs=LSH(len(SIG),b,r,SIG)
        new_pairs2=[]
        
        #LSH pairs containing only the 20 first movies
        #LSH_pairs=[i for i in LSH_pairs if (i[0] in movieIdList and i[1] in movieIdList)]
        LSH_pairs=[i for i in LSH_pairs]
        
        lst= set(LSH_pairs).intersection(set(truePositivePairs))
        truePositives_test2[n]=len(lst)
        falsePositives_test2[n]=len(set(LSH_pairs)-set(lst))
        falseNegatives_test2[n]=len(set(truePositivePairs)-set(lst))
        #print(LSH_pairs,truePositivePairs)
        #print(len(LSH_pairs),truePositives_test2[n],falsePositives_test2[n])
        
    for i in range(len(rlist)):
        if truePositives_test2[i]!= 0 or falsePositives_test2[i]!=0:
            PRECISION_test2[i]=truePositives_test2[i]/(truePositives_test2[i]+falsePositives_test2[i])
        if truePositives_test2[i]!= 0 or falseNegatives_test2[i]!=0:
            RECALL_test2[i]=truePositives_test2[i]/(truePositives_test2[i]+falseNegatives_test2[i])
        if RECALL_test2[i]!= 0 or PRECISION_test2[i]!=0:
            F1_test2[i]=2*RECALL_test2[i]*PRECISION_test2[i]/(RECALL_test2[i]+PRECISION_test2[i])
            
    d = {'r': rlist,'b':blist,'s':slist, 'PRECISION': PRECISION_test2,'RECALL': RECALL_test2,'F1':F1_test2,'false-positives':falsePositives_test2,'false-negatives':falseNegatives_test2}
    df_test2 = pandas.DataFrame(data=d)
    print(df_test2)
    #plt.figure(); df_test2.plot(x = 's', y = 'PRECISION');
    #plt.savefig('Precision_LSH_ratings_605users.png')
    #plt.figure(); df_test2.plot(x = 's', y = 'RECALL');
    #plt.savefig('Recall_LSH_ratings_605users.png')
    #plt.figure(); df_test2.plot(x = 's', y = 'F1');
    #plt.savefig('F1_LSH_ratings_605users.png')
    #plt.figure(); df_test2.plot(x = 's', y = 'false-negatives');
    #plt.savefig('false-negatives_LSH_ratings_605users.png')
    #plt.figure(); df_test2.plot(x = 's', y = 'false-positives');
    #plt.savefig('false-positives_LSH_ratings_605users.png')
    
    
def calculateLists(data,filename):

    #userList
    userId = 1
    while True:
        user = data.loc[data['userId'] == userId]
        if user.empty:
            break
        userList[userId] = user['movieId'].values.tolist()
        userId += 1

    file = open('userList_'+filename,'w')
    writer = csv.writer(file)
    for key,value in userList.items():
        writer.writerow([key,value])
    file.close()

    #movieMap
    i = 0
    for movieId in data['movieId']:
        if movieId not in movieMap:
            movieMap[movieId] = i
            i += 1

    file = open('movieMap_'+filename,'w')
    writer = csv.writer(file)
    for key,value in movieMap.items():
        writer.writerow([key,value])
    file.close()

    #movieList
    users = []
    for movieId in movieMap:
        for userId, movies in userList.items():
            if movieId in movies:
                users = users + [userId]
        movieList[movieId] = users
        users = []

    file = open('movieList_'+filename,'w')
    writer = csv.writer(file)
    for key,value in movieList.items():
        writer.writerow([key,value])
    file.close()

def calculateJaccardSimilarity(movieList1,movieList2):
    s1 = set(movieList1)
    s2 = set(movieList2)
    jaccard = (len(s1.intersection(s2))/len(s1.union(s2)))
    return jaccard

def calculateMinHash(filename):
    N = len(movieMap)
    K = len(userList)
    SIG = numpy.full( (n, N), K )
    for i in range(n):
        hashFunction = createRandomHashFunction(K)
        for movieId, users in movieList.items():
            for userId in users:
                movie = movieMap[movieId]
                sign = hashFunction(userId)
                if sign < SIG[i][movie]:
                    SIG[i][movie] = sign

    with open("SIG_"+filename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(SIG)
    f.close()
    return SIG

def createRandomHashFunction(p=2**33-355, m=100):
    a = random.randint(1,p-1)
    b = random.randint(0, p-1)
    return lambda x: 1 + (((a * x + b) % p) % m)

def signatureSimilarity(movieId1,movieId2,rows,SIG):
    columns = SIG[:,movieMap[movieId1]-1]
    movie1 = columns[0:rows]
    columns = SIG[:,movieMap[movieId2]-1]
    movie2 = columns[0:rows]
    similarity=calculateJaccardSimilarity(movie1,movie2)
    
    return similarity

def LSH(numberOfSignatures,b,r,SIG):
    pairs1 = set()
    pairs2 = set()
    hashFunction = createRandomHashFunction()
    for band in range(b):
        buckets = {}
        for i in range(numberOfSignatures):
            sign = SIG[:,i]
            subsign = sign[band*r:b*r+r]
            valueOfHash = hashFunction(int(''.join(map(str,subsign))))
            if valueOfHash not in buckets:
                buckets[valueOfHash] = [i]
            else:
                buckets[valueOfHash].append(i)

        for bucket in buckets.values():
            for j in range(len(bucket)):
                for l in range(j+1,len(bucket)):
                    pairs1.add((bucket[j],bucket[l]))
        pairs2 = pairs2.union(pairs1)
    return pairs2

if __name__ == "__main__":
    main()
