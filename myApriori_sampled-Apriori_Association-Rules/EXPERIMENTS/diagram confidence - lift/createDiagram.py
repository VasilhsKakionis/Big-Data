import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

df1 = pd.read_csv('df1.csv')#0.1
df2 = pd.read_csv('df2.csv')#0.15
df3 = pd.read_csv('df3.csv')#0.2

y1=df1['confidence']
y2=df2['confidence']
y3=df3['confidence']
colors = ['b','g','r']

#plots the Conf histogram
fig, ax1 = plt.subplots()
ax1.hist([y1,y2,y3],color=colors,label=['min freq=0.1', 'min freq=0.15','min freq=0.2'])
ax1.set_xlim(0,1)
ax1.set_ylabel("Count")
ax1.set_xlabel("Confidence")
plt.tight_layout()
plt.legend(loc='upper right')
#plt.savefig("confDiag.png")
plt.show()

y1=df1['lift']
y2=df2['lift']
y3=df3['lift']
#plots the Lift histogram
fig, ax1 = plt.subplots()
ax1.hist([y1,y2,y3],color=colors,label=['min freq=0.1', 'min freq=0.15','min freq=0.2'])
ax1.set_xlim(0,20)
ax1.set_ylabel("Count")
ax1.set_xlabel("Lift")
plt.tight_layout()
plt.legend(loc='upper right')
#plt.savefig("LiftDiag.png")
plt.show()