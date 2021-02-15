#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import sys


# In[2]:
window=tk.Tk()

window.title('Movie Recommendation (SVM)')

canvas1 = tk.Canvas(window, width = 600, height = 600)

ent=tk.Label(window,text='Movie Recommendation - SVM')
canvas1.create_window(290,100,window=ent)

ent=tk.Label(window,text='Enter genre')
canvas1.create_window(220,140,window=ent)

var = tk.StringVar(window)
var.set("Choose Genre") # initial value

option = tk.OptionMenu(window, var, "Action","Adventure","Animation","Biography","Comedy","Crime","Drama","Family","Fantasy","History","Horror","Music","Musical","Mystery","Romance","Sci-Fi","Sport","Thriller","War","Western")
canvas1.create_window(340,140,window=option)


canvas1.pack()

#e1=tk.Entry(window,text='Enter the genre of your choice ')
#canvas1.create_window(200, 140, window=e1)
#list2=['Action',"Adventure","Animation","Biography","Comedy","Crime","Drama","Family","Fantasy","History","Horror","Music","Musical","Mystery","Romance","Sci-Fi","Sport","Thriller","War","Western"]   


# In[3]:


def find():

    window2=tk.Tk()
    window2.title('Movie Recommendation (SVM)')
    canvas2= tk.Canvas(window2, width = 600, height = 600)
    canvas2.pack()
    var.get()
    x_input=var.get()
    dataset = pd.read_csv('IMDB-Movie-Data.csv')
    print(dataset)
   
    # In[4]:


    dataset = dataset.sort_values('Rating',ascending=False)
    x= dataset.iloc[:,[2,8]]
    print(x)


    # In[5]:


    x = pd.concat([x.drop('Genre', 1), x['Genre'].str.get_dummies(sep=",")], 1)
    print(x)


    # In[6]:


    X = x.iloc[:,0:21].values


    # In[7]:


    from sklearn.impute import SimpleImputer
    imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer.fit(X[:,[1]])
    X[:,[1]]=imputer.transform(X[:,[1]])


    # In[8]:


    x1=dataset.iloc[:,[0,2,8]]
    x1 = pd.concat([x1.drop('Genre', 1), x1['Genre'].str.get_dummies(sep=",")], 1)


    # In[9]:


    from sklearn.decomposition import PCA
    pca = PCA(n_components=None)#SInce we do not know how many eigen vectors are needed we keep value of n components=None so that we can get the eigen values of all the eigen vectors to figure out the best one
    X = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)


    # In[10]:


    np.set_printoptions(precision=2)
    print(X)


    # In[11]:


    y= dataset.iloc[:,0].values
    print(y)


    # In[12]:


    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf',random_state=0)
    classifier.fit(X,y)


    # In[13]:


    y_pred = classifier.predict(X)
    i=0
    print('Recommended Movies')
    
    for j in range(0,1000):
        try:
            if(x1[x1.Rank==y_pred[j]][x_input].values[0]==1):
                l1=tk.Label(window2,text=dataset[dataset.Rank==y_pred[j]]['Title'].values[0])
                canvas2.create_window(200, 250+(20*i),window=l1)
                l2=tk.Label(window2,text=dataset[dataset.Rank==y_pred[j]]['Rating'].values[0])
                canvas2.create_window(430, 250+(20*i),window=l2)
                i+=1
            if i==10:
                break
            else:
                pass
        except IndexError:
            break


    def onclick():
        window2.destroy()    
        window.destroy()
        raise SystemExit


    def onclick1():
        window2.destroy()
    
        
    bt2=tk.Button(window2,text='Close And Exit',command=onclick)
    canvas2.create_window(350, 480, window=bt2)

    bt22=tk.Button(window2,text='New Genre',command=onclick1)
    canvas2.create_window(250, 480, window=bt22)



    l3=tk.Label(window2,text='Recommended Movie')
    canvas2.create_window(200,220,window=l3)
    l4=tk.Label(window2,text='Rating')
    canvas2.create_window(430,220,window=l4)

    # In[14]:


    import sklearn.metrics as met
    from sklearn.metrics import accuracy_score
    print(met.r2_score(y,y_pred))
    print(met.mean_squared_error(y,y_pred))
    print(accuracy_score(y,y_pred))

    l6=tk.Label(window2,text='Showing Movies For - '+x_input)
    canvas2.create_window(300,180,window=l6)
    window2.mainloop()


def delay1():
    l5=tk.Label(window,text='Result Shown in New Window')
    canvas1.create_window(290,220,window=l5)
    window.after(3000,find)
    
bt1=tk.Button(window,text='Show Movies',command=delay1)
bt1.pack()
canvas1.create_window(290, 180, window=bt1)


window.mainloop()


	# In[15]:


	#x_new=dataset.iloc[:,[8]].values
	#print(x_new)
	#plt.scatter(x_new,y_pred,color='red')
	#plt.show()


	# In[ ]:

