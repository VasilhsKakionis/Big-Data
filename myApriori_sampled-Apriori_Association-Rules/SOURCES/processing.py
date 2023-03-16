import itertools
import sys
import pandas 
import random
import numpy
import csv
import matplotlib.pyplot as plt
import itertools
import keyboard
import time
import numpy 
import matplotlib.pyplot as plt
import networkx as nx

ruleid=0
itemset=[]
hypothesis=[]
conclusion=[]
rule=[]
frequency=[]
confidence=[]
lift=[]
interest=[]
ruleID=[]
tested={}
item1=[]

def main():

    input_file = "ratings_100users.csv"
    movies_file = "movies.csv"
    minScore=float(input('Give the min score(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0):'))
    minScoreInputs = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
    while(minScore not in minScoreInputs):
        print("Invalid value of min score")
        minScore=float(input('Give the min score(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0):'))

    min_frequency=float(input('Give the min frequency(0,1):'))
    while(min_frequency<0 or min_frequency>1.0):
        print("Invalid value of min frequency")
        min_frequency=float(input('Give the min frequency(0,1):'))

    max_length = int(input('Give the max length (>=2):'))
    while(max_length<2):
        print("Invalid value of min frequency")
        max_length = int(input('Give the max length (>=2):'))

    min_confidence = float(input('Give the min confidence (0,1):'))
    while(min_confidence<0 or min_confidence>1):
        print("Invalid value of min confidence")
        min_confidence = float(input('Give the min confidence (0,1):'))

    MinLift = float(input('Give the min lift (>1 or =-1 (ignore param)):'))
    while(MinLift<=1 and MinLift!=-1):
        print("Invalid value of min lift")
        MinLift = float(input('Give the min lift (>1 or =-1 (ignore param)):'))

    MaxLift = float(input('Give the max lift ((0,1) or =-1 (ignore param)):'))
    while((MaxLift<0 and MaxLift!=-1) or MaxLift>1):
        print("Invalid value of max lift")
        MaxLift = float(input('Give the max lift ((0,1) or =-1 (ignore param)):'))

    userBaskets = CreateMovieBackets(input_file,minScore)

    movies = pandas.read_csv(movies_file)

    movies_df = ReadMovies(movies)

    K = movies.shape[0]

    pairsCounter = TriangularMatrixOfPairsCounter(userBaskets,movies_df,K)

    counterOfPairs = HashedCountersOfPairs(userBaskets) 

    frequent_items1 = myApriori(userBaskets,min_frequency,max_length)
    #createFile(frequent_items1)
    
    #frequent_items2 = sampledApriori(minScore,min_frequency,max_length)
    #createFile(frequent_items2)

    #compute_results(frequent_items1,frequent_items2)

    x = AssociationRulesCreation(min_confidence,MinLift,MaxLift,frequent_items1)

    presentResults(x)
    
def createFile(res1):
    name=str(input("name for file:"))
    items=[tuple(dict(x).keys()) for x in res1]
    it=[j for i in items for j in i]
    
    it1 = []
    for i in it:
    	it1.append([i])

    file = open(name+'.csv','w')
    writer = csv.writer(file)
    for i in it1:
        writer.writerow(i)
    file.close()
    

def CreateMovieBackets(name,minScore):

    data = pandas.read_csv(name)
    userId = 1
    userItems={}
    stars = {}

    while True:
        user = data.loc[data['userId'] == userId]
        if user.empty:
            break
        userItems[userId] = user['movieId'].values.tolist()
        stars[userId] = user['rating'].values.tolist()
        userId += 1

    backets = []
    for x, y in userItems.items():
        backets.append(y)
    
    rating = []
    for x,y in stars.items():
        rating.append(y)
    
    finalBackets = []
    for i in range(len(backets)):
        backet = []
        for j in range(len(backets[i])):
            if(rating[i][j]>=minScore):
                backet.append(backets[i][j])
        finalBackets.append(backet)
        
    file = open('userBaskets.csv','w')
    writer = csv.writer(file)
    for i in range(len(finalBackets)):
        writer.writerow(finalBackets[i])
    file.close()

    return finalBackets

def ReadMovies(data):

     movies_df = {}
     index = 1
     for i in range(data.shape[0]):
        lineOfFrame = data.loc[[i][0]]
        movieId = int(lineOfFrame[0])
        title = lineOfFrame[1]
        genres =  lineOfFrame[2]
        movies_df[movieId] = [index,title,genres]
        index = index + 1

     return movies_df
        
def TriangularMatrixOfPairsCounter(baskets,movies,K):

    numberOfPairs = int(K*(K-1)/2)
    pairs = [0]*numberOfPairs
    for k in range(len(baskets)):
        for i in range(len(baskets[k])-1):
            for j in range(i+1,len(baskets[k])):
                l = movies[baskets[k][i]][0]
                m = movies[baskets[k][j]][0]
                pos = int((l-1)*(K-l/2)+m-l-1)
                pairs[pos] += 1

    return pairs

def HashedCountersOfPairs(baskets):

    countersOfPairs = {}
    for i in range(len(baskets)):
        for pair in itertools.combinations(baskets[i],2):
            if pair in countersOfPairs:
                countersOfPairs[pair] += 1
            else:
                countersOfPairs[pair] = 1

    return countersOfPairs

def myApriori(itemBaskets, min_frequency,max_length):

    N=len(itemBaskets)
    frequent_itemsets=[]
    frequent_itemset=[]
    cand_set={}
    cand_set_number = []
    frequent_itemsets_num = []
    times = []
    global items1
    index = 0

    for k in range(2,max_length+2):

        start = time.time()

        if(k==2):
            for basket in itemBaskets:
                for j in basket:
                    if j in cand_set:
                        cand_set[j] += 1
                    else:
                        cand_set[j] = 1
        else:
            for basket in itemBaskets:
                for i in item1:
                    if set(i)<=set(basket):
                        cand_set[i] += 1
                       
        
        frequent_items={}
        for i,j in cand_set.items():
            if j/N>=min_frequency:
                frequent_items[i] = j
                
        frequent_itemsets_num.append(len(frequent_items))
        frequent_itemsets.append(frequent_items)

        itemset = list(frequent_items)
        cand_set = {}
        item1 = []
        if(k==2):
            for item in itertools.combinations(itemset,k):
                cand_set[item] = 0
                item1.append(item)
        else:
            items = []
            for i in range(len(itemset)):
                for j in range(len(itemset[i])):
                    
                    if itemset[i][j] not in items:
                        items.append(itemset[i][j])
            
            items.sort()
            
            for item in itertools.combinations(items,k):
                cand_set[item] = 0
                item1.append(item)
          
        
        cand_set_number.append(len(cand_set))
        times.append(time.time() - start)
        index += 1
        
        if(len(cand_set)<=0):
            break
    print("")
    for i in range(index):

        print("Apriori's loop number: ",i+1)
        
        if(i==0):
            print("Apriori has ",frequent_itemsets_num[i]," frequent single sets in loop ",i+1)
            print("Apriori has ",cand_set_number[i]," candidate frequent pairs in loop ",i+1)
        else:
            if (i==1):
                print("Apriori has ",frequent_itemsets_num[i]," frequent pairs in loop ",i+1)
            else:
                print("Apriori has ",frequent_itemsets_num[i]," frequent ",i+1,"-plets in loop ",i+1)
            print("Apriori has ",cand_set_number[i]," candidate frequent ",i+2,"-plets in loop ",i+1)

        print("Running time = ",times[i])
        print("")
        
   
    return frequent_itemsets

def sampledApriori(minScore,min_frequency,max_length):
    print("")
    print("Sampled Apriori")
    
    rating_stream = CreateMovieBackets("ratings_100users_shuffled.csv",minScore)
    
    if(len(rating_stream)==100):
        size = 50
    else:
        size = 100
    
    ratings = [[]]*size
    for i in range(len(ratings)):
        ratings[i] = rating_stream[i]

    while(True):
        if keyboard.is_pressed('y') or keyboard.is_pressed('Y') :
            break
        else:
            print('\tDKEEP READING MOVIE RATINGS ... until you press \'Y\' or \'y')
            index = 0
            while(index<len(rating_stream)):
                x = random.randrange(index+1)
                if(x<size):
                    ratings[x] = rating_stream[index]
                index += 1
        time.sleep(0.06)
    newRatings=[]
    for i in ratings:
        #print(i)
        l=i.copy()
        l.sort()
        newRatings.append(l)

    rating=newRatings
            

    new_min_frequency = min_frequency/(len(rating_stream)/len(ratings))
    frequent_itemsets = myApriori(ratings,new_min_frequency,max_length)
    
    second_pass=int(input('For second pass press 2: '))
    if second_pass==2 :
        
        values = []
        for diction in frequent_itemsets:
            keys = []
            keys = diction.keys()
            
            for i in keys:
                if i not in values:
                    values.append(i)

        frequent_item_sets = {}
        for i in values:
            frequent_item_sets[i] = 0

        for basket in ratings:
            for i in values:
                if type(i) == int:
                    if i in basket:
                        frequent_item_sets[i] += 1 
                else:
                    if set(i)<=set(basket):
                        frequent_item_sets[i] += 1
        
        frequent_items = {}
        key = 0
        for i,j in frequent_item_sets.items():
            if j/(len(ratings))>=min_frequency:
                frequent_items[i] = j
                key = i

        items=[]
        for i in range(len(key)):
            temp = {}
            for x,y in frequent_items.items():
                if type(x) == int:
                    if i==0:
                        temp[x]=y
                else:
                    if len(x)==i+1:
                        temp[x]=y
            items.append(temp)
        
        frequent_itemsets = items

    return frequent_itemsets

#Compute precision, recall and f1-score based on myApriori & sampledApriori results    
def compute_results(res1,res2):
    items1=[tuple(dict(x).keys()) for x in res1]
    it1=[j for i in items1 for j in i]
    items2=[tuple(dict(x).keys()) for x in res2]
    it2=[j for i in items2 for j in i]

    tp_set=set(it1).intersection(set(it2))
    fp_set=set(it2)-tp_set
    fn_set=set(it1)-tp_set
    
    tp=len(list(tp_set))
    fp=len(list(fp_set))
    fn=len(list(fn_set))
    
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1 = 2*(precision*recall) / (precision+recall)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1-score:",f1)	
 
def checkInterest(lift,MinLift,MaxLift):

    if MinLift !=-1 and MaxLift!=-1:
        return False 		
    
    if MinLift!=-1:
        if lift>MinLift:
           return True
    
    if MaxLift!=-1:
        if lift<MaxLift:
            return True
    
    if MinLift ==-1 and MaxLift==-1:
        return True
    
    return False

def findB(lst1,lst2):
    return tuple(sorted(set(lst1)-set(lst2)))
 
def findFreq(item,A):

    cnt=0
    for i in range(1,len(item)):
        d=dict(item[i])
        for x in d.keys():
            if A in x:
                cnt+=d[x]
            if isinstance(A, int)==False:
                if set(A) <= set(x):
                    cnt+=d[x]
    return cnt
    
def numOfElements(item):
    
    cnt=0
    for i in item:
        d=dict(i)
        cnt+=len(d.keys())
    
    return cnt
       
def check(splits,j,items,MinLift,MaxLift,min_confidence,numOfElem):
    global ruleid
    global itemset
    global hypothesis 
    global conclusion
    global rule
    global frequency
    global confidence
    global lift
    global interest
    global ruleID
    global tested
    for i in splits:
        size=len(i)
        A=i
        sizeA=len(A)
        B=findB(j,A)
        sizeB=len(B)
        if len(A)==1:
            A=A[0]
            sizeA=1
        if len(B)==1:
            B=B[0]
            sizeB=1
        #print('A,B',A,B)
        
        frA=findFreq(items,A)/numOfElem
        frAB=findFreq(items,j)/numOfElem
            
        confAB=frAB/frA
        frB=findFreq(items,B)/numOfElem
        pB=frB
        interestAB=confAB-pB
        liftAB=confAB/pB
        
        if confAB<min_confidence:
            continue
        else:
            if A in tested.keys() and B in tested.values():
                continue
            tested[A]=B
            if checkInterest(liftAB,MinLift,MaxLift):
                itemset.append(j)
                hypothesis.append(A)
                conclusion.append(B)
                rule.append(str(A)+'-->'+str(B))
                frequency.append(frA)
                confidence.append(confAB)
                lift.append(liftAB)
                interest.append(interestAB)
                ruleid+=1
                ruleID.append(ruleid)
                if size>1:
                    d=[]
                    for pair in itertools.combinations(A,sizeA-1):
                        d.append(pair)
                    check(d,j,items,MinLift,MaxLift,min_confidence,numOfElem)
            else:
                return
   
def AssociationRulesCreation(min_confidence,MinLift,MaxLift,items):

    global ruleid
    global itemset
    global hypothesis 
    global conclusion
    global rule
    global frequency
    global confidence
    global lift
    global interest
    global ruleID
    print("~~~~~~~~~~~~~~~~~")
    
    #begin from pairs
    tested=[]
    numofElem=numOfElements(items)
    for i in range(1,len(items)):
        for j in items[i]:
            size=len(j)
            split=[]
            for pair in itertools.combinations(j,size-1):
                split.append(pair)
            check(split,j,items,MinLift,MaxLift,min_confidence,numofElem)

    data = {'itemset':itemset, 'hypothesis':hypothesis,'conclusion':conclusion,'rule':rule,'frequency':frequency,'confidence':confidence,'lift':lift,'interest':interest,
    'rule ID':ruleID} 
    rules_df = pandas.DataFrame(data) 
    rules_df=rules_df[['itemset','hypothesis','conclusion','rule','frequency','confidence','lift','interest','rule ID']]
    print(rules_df)

    return rules_df

def presentResults(rules_df):
    #name = str(input("name for df: "))  
    #rules_df.to_csv(name+'.csv', index=False)

    while True:
        printChoices()
        choice=input()
        c=choice[0]
        
        if c=='e':
            break
            
        l=['a','b','c','h','m','r','s','v','e','o']
        if c not in l:
            print("Invalid Option")
            continue
            
        if c=='a':
            print(rules_df.rule)

        elif c=='b':   
            choices={'i':'itemset','c':'conclusion','h':'hypothesis'}
            s=choice[5:-1]
            option=choices.get(choice[2])
            movies=s.split(',')
            movies=[int(i) for i in movies]
            if len(movies)!=1:
                movies=tuple(movies)
            else:
                movies=int(movies[0])
            A=rules_df.loc[rules_df[option] == movies]
            print("")
            print(A)

        elif c=='c':
            
            fit = numpy.polyfit(rules_df['lift'], rules_df['confidence'], 1)
            fit_fn = numpy.poly1d(fit)
            plt.plot(rules_df['lift'], rules_df['confidence'], 'yo', rules_df['lift'],
            fit_fn(rules_df['lift']))
            plt.xlabel('Lift')
            plt.ylabel('Confidence')
            plt.title('CONFIDENCE vs LIFT')
            plt.tight_layout()
            #name = str(input("name for c: "))            
            #plt.savefig(name)
            plt.show() 

        elif c=='h':
            option = choice.split(',')
            hist = option[1]
            y = rules_df.loc[:, 'frequency'].tolist()
            x = None

            if hist == 'c':
                x='confidence'

            elif hist == 'l':
                x='lift'

            else:
                print('Invalid Option')
                continue

            rules_df[x].plot.hist(bins=12, alpha=0.5)
            plt.title('Histogram of'+ x +'among discovered rules')
            plt.xlabel(x)
            plt.ylabel('Number of Rules')
            plt.tight_layout()
            #name = str(input("name for h: "))            
            #plt.savefig(name)
            plt.show() 			

        elif c== 'm':
            movieID=int(choice[2:])
            movies = pandas.read_csv("movies.csv")
            movies_df = ReadMovies(movies)
            print("")
            print(movies_df[movieID][1:])

        elif c== 'r':
            ruleID=int(choice[-1])
            print("")
            print(rules_df.loc[ruleID])
            
        elif c == 's':
            option = choice.split(',')
            sortBy= option[1]
            if sortBy == 'c':
                rules_df = rules_df.sort_values( by = ['confidence'] )
                print(rules_df)

            elif sortBy == 'l':
                rules_df = rules_df.sort_values( by = ['lift'] )
                print(rules_df)

            else:
                print("Invalid Option")
                continue  

        elif c == 'v':
            option=choice[-1]
            numRules=choice[2]
            op=['c','r','s']
            
            if option not in op:
                continue

            draw_graph(rules_df,option)            
        else:
            print("Invalid Option")
            continue
    
def draw_graph(rules,draw_choice):

    G = nx.DiGraph()

    color_map = []
    final_node_sizes = []

    color_iter = 0

    NumberOfRandomColors = 3000
    edge_colors_iter = numpy.random.rand(NumberOfRandomColors)

    node_sizes = {}     # larger rule-nodes imply larger confidence
    node_colors = {}    # darker rule-nodes imply larger lift
    
    for index, row in rules.iterrows():
        
        color_of_rule = edge_colors_iter[color_iter]
        rule = row['rule']
        rule_id = row['rule ID']
        confidence = row['confidence']
        lift = row['lift']
        itemset = row['itemset']
        hypothesis=row['hypothesis']
        conclusion=row['conclusion']
        
        
        G.add_nodes_from(["R"+str(rule_id)])

        node_sizes.update({"R"+str(rule_id): float(confidence)})

        node_colors.update({"R"+str(rule_id): float(lift)})
        
        
        #for item in hypothesis:
        G.add_edge(str(hypothesis), "R"+str(rule_id), color=color_of_rule)

        #for item in conclusion:
        G.add_edge("R"+str(rule_id), str(conclusion), color=color_of_rule)

        color_iter += 1 % NumberOfRandomColors

    print("\t++++++++++++++++++++++++++++++++++++++++")
    print("\tNode size & color coding:")
    print("\t----------------------------------------")
    print("\t[Rule-Node Size]")
    print("\t\t5 : lift = max_lilft, 4 : max_lift > lift > 0.75*max_lift + 0.25*min_lift")
    print("\t\t3 : 0.75*max_lift + 0.25*min_lift > lift > 0.5*max_lift + 0.5*min_lift")
    print("\t\t2 : 0.5*max_lift + 0.5*min_lift > lift > 0.25*max_lift + 0.75*min_lift")
    print("\t\t1 : 0.25*max_lift + 0.75*min_lift > lift > min_lift")
    print("\t----------------------------------------")
    print("\t[Rule-Node Color]")
    print("\t\tpurple : conf > 0.9, blue : conf > 0.75, cyan : conf > 0.6, green  : default")
    print("\t----------------------------------------")
    print("\t[Movie-Nodes]")
    print("\t\tSize: 1, Color: yellow")
    print("\t----------------------------------------")

    max_lift = rules['lift'].max()
    min_lift = rules['lift'].min()

    base_node_size = 500
    
    for node in G:

        if str(node).startswith("R"): # these are the rule-nodes...
                
            conf = node_sizes[str(node)]
            lift = node_colors[str(node)]
            
            # rule-node sizes encode lift...
            if lift == max_lift:
                final_node_sizes.append(base_node_size*5*lift)

            elif lift > 0.75*max_lift + 0.25*min_lift:
                final_node_sizes.append(base_node_size*4*lift)

            elif lift > 0.5*max_lift + 0.5*min_lift:
                final_node_sizes.append(base_node_size*3*lift)

            elif lift > 0.25*max_lift + 0.75*min_lift:
                final_node_sizes.append(base_node_size*2*lift)

            else: # lift >= min_lift...
                final_node_sizes.append(base_node_size*lift)

            # rule-node colors encode confidence...
            if conf > 0.9:
                color_map.append('purple')

            elif conf > 0.75:
                color_map.append('blue')

            elif conf > 0.6:
                color_map.append('cyan')

            else: # lift > min_confidence...
                color_map.append('green')

        else: # these are the movie-nodes...
            color_map.append('yellow') 
            final_node_sizes.append(2*base_node_size)

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]

    if draw_choice == 'c': #circular layout
        nx.draw_circular(G, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=True)

    elif draw_choice == 'r': #random layout
        nx.draw_random(G, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=True)

    else: #spring layout...
        pos = nx.spring_layout(G, k=16, scale=1)
        nx.draw(G, pos, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=False)
        nx.draw_networkx_labels(G, pos)    

    #name = str(input("name for v: "))            
    #plt.savefig(name)
    plt.show() 

    # discovering most influential and most influenced movies
    # within highest-lift rules...
    outdegree_rules_sequence = {}
    outdegree_movies_sequence = {}
    indegree_rules_sequence = {}
    indegree_movies_sequence = {}
    
    outdegree_sequence = nx.out_degree_centrality(G)
    indegree_sequence = nx.in_degree_centrality(G)

    for (node, outdegree) in outdegree_sequence.items():
        # Check if this is a rule-node
        if str(node).startswith("R"):
            outdegree_rules_sequence[node] = outdegree
        else:
            outdegree_movies_sequence[node] = outdegree
            
    for (node, indegree) in indegree_sequence.items():
        # Check if this is a rule-node
        if str(node).startswith("R"):
            indegree_rules_sequence[node] = indegree
        else:
            indegree_movies_sequence[node] = indegree

    max_outdegree_movie_node = max(outdegree_movies_sequence, key=outdegree_movies_sequence.get)
    max_indegree_movie_node = max(indegree_movies_sequence, key=indegree_movies_sequence.get)
    print("\tMost influential movie (i.e., of maximum outdegree) wrt involved rules: ",max_outdegree_movie_node)
    print("\tMost influenced movie (i.e., of maximum indegree) wrt involved rules: ",max_indegree_movie_node)

def printChoices():

    print("\n")
    print("(a) List All discovered rules [format:a]")
    print("(b) List all rules containing a BAG of movies in their <ITEMSET|HYPOTHESIS|CONCLUSION> [format:b,<i,h,c>,,<comma-sep. movie IDs>]")
    print("(c) COMPARE rules with <CONFIDENCE,LIFT>  [format: c]")
    print("(h) Print the HISTOGRAM of <CONFIDENCE|LIFT >  [format: h,<c,l >]")
    print("(m) Show details of a MOVIE    [format: m,<movie ID>]")
    print("(r) Show a particular RULE     [format: r,<rule ID>] ")
    print("(s) SORT rules by increasing <CONFIDENCE|LIFT >  [format: s,<c,l >]")
    print("(v) VISUALIZATION of association rules   [format: v,<draw_choice: \n\t(sorted by lift) \t\t[c(ircular),r(andom),s(pring)]>,<num of rules to show>]")
    print("(e) EXIT       [format: e]")   

if __name__ == "__main__":
    main()
