#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as met
import tkinter as tk
import sys


# In[2]:

window3=tk.Tk()

window3.title('Movie Recommendation (KNN)')

canvas5 = tk.Canvas(window3, width = 600, height = 600)


ent=tk.Label(window3,text='Movie Recommendation - KNN')
canvas5.create_window(290,100,window=ent)

ent=tk.Label(window3,text='Enter genre')
canvas5.create_window(220,140,window=ent)

var1 = tk.StringVar(window3)
var1.set("Choose Genre") # initial value

option = tk.OptionMenu(window3, var1, "Action","Adventure","Animation","Biography","Comedy","Crime","Drama","Family","Fantasy","History","Horror","Music","Musical","Mystery","Romance","Sci-Fi","Sport","Thriller","War","Western")
canvas5.create_window(340,140,window=option)


canvas5.pack()

def find():
    window4=tk.Tk()
    window4.title('Movie Recommendation (KNN)')
    canvas6= tk.Canvas(window4, width = 600, height = 600)
    canvas6.pack()
    #Importing datasets
    var1.get()
    x_input=var1.get()
    dataset = pd.read_csv('IMDB-Movie-Data.csv')
    print(dataset.columns)


    # In[3]:


    #Checking for outliers
    dataset.boxplot(column=['Votes'])


    # In[4]:


    # Checking for Null values
    dataset.count()


    # In[5]:


    # Filling null vales with mean
    dataset['Metascore'].fillna(dataset['Metascore'].mean(), inplace=True)
    dataset['Revenue (Millions)'].fillna(dataset['Revenue (Millions)'].mean(), inplace=True)
    dataset.count()


    # In[6]:


    # Taking Important columns
    df = dataset.drop(['Description','Director','Year','Actors','Runtime (Minutes)','Revenue (Millions)','Metascore','Votes'],1)
    df.count()


    # In[7]:


    # printing new dataset with selective columns
    print(df)


    # In[8]:


    # Creating dataset with movies sorted on basis of rating (descending).
    df_descending = df.copy()
    df_descending = df_descending.sort_values('Rating',ascending=False)
    ref_genre = df_descending.copy() 
    print(df_descending)


    # In[9]:


    # Converting each genre into column
    df_descending = pd.concat([df_descending.drop('Genre', 1), df_descending['Genre'].str.get_dummies(sep=",")], 1)
    print(df_descending)


    # In[10]:


    X = df_descending.drop(['Title'],1)
    X.columns
    print(X)


    # In[11]:


    X = X.values
    print(X)


    # In[12]:


    #Getting Input Genre 
    def get_key(val): 
        for key, value in list1.items(): 
             if val == value: 
                 return key 
      
        return "key doesn't exist"
      
    list1 = {2:"Action",3:"Adventure",4:"Animation",5:"Biography",6:"Comedy",7:"Crime",8:"Drama",9:"Family",10:"Fantasy",11:"History",12:"Horror",13:"Music",14:"Musical",15:"Mystery",16:"Romance",17:"Sci-Fi",18:"Sport",19:"Thriller",20:"War",21:"Western"}  
    #list1 = {3:"Action",4:"Adventure",5:"Animation",6:"Biography",7:"Comedy",8:"Crime",9:"Drama",10:"Family",11:"Fantasy",12:"History",13:"Horror",14:"Music",15:"Musical",16:"Mystery",17:"Romance",18:"Sci-Fi",19:"Sport",20:"Thriller",21:"War",22:"Western"}  
    #x_input=input()
    col=get_key(x_input)
    print(col)


    # In[13]:


    #preparing training data with input Genre
    x_test = []
    y_test = []
    Ranks  = []
    Rating = []
    for i in range(0,len(X)):
        if X[i,col]==1:
            Ranks.append(X[i,0])
            Rating.append(X[i,1])
            x_test.append(X[i,1:])
            y_test.append(X[i,0])
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    y_test=y_test.ravel()


    # In[14]:


    print(x_test)


    # In[15]:


    X_train_predict = x_test.copy()
    Y_train_predict = y_test.copy()


    # In[16]:


    # Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train_predict = sc.fit_transform(X_train_predict)


    # In[17]:


    from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors


    # In[18]:


    classifier_knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
    classifier_knn.fit(X_train_predict,Y_train_predict)


    # In[19]:


    #Predicting
    Y_pred_predict = classifier_knn.predict(X_train_predict)
    print(Y_pred_predict)
    (Y_pred_predict==Y_train_predict).all()


    # In[20]:


    #Checking accuracy
    from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
    accu1 = accuracy_score(Y_train_predict,Y_pred_predict)
    print('Accuracy :',accuracy_score(Y_train_predict,Y_pred_predict))

    r2 = met.r2_score(Y_train_predict,Y_pred_predict)
    print('R-square_score : ',r2)

    mse = met.mean_squared_error(Y_train_predict,Y_pred_predict) 
    print('MSE : ',mse)

    rmse = np.sqrt(mse)
    print('RMSE : ',rmse)


    # In[21]:


    #Plotting predicted Rank with respect to original Rating
    #plt.scatter(X_train_predict[:,0],Y_train_predict,color='magenta')#original rank
    #plt.scatter(X_train_predict[:,0],Y_pred_predict,color='blue')#predicted rank
    #plt.xlabel(x_input)
    #plt.ylabel('Rank')
    #plt.show()


    # In[22]:


    #Plotting predicted Rank with respect to input Genre
    #plt.scatter(X_train_predict[:,col],Y_train_predict,color='red')#original rank
    #plt.scatter(X_train_predict[:,col],Y_pred_predict,color='blue')#predicted rank
    #plt.xlabel(x_input)
    #plt.ylabel('Rank')
    #plt.show()


    # In[23]:


    # Recommendation using Nearest Neighbor
    nbrs = NearestNeighbors(n_neighbors=6,metric='cosine',algorithm='brute').fit(X)


    # In[25]:


    distances , indices = nbrs.kneighbors(X,n_neighbors=100)


    # In[26]:


   # data = list()


    # In[27]:


    #Function to get index of input Movie Title
    def get_index(title):
        return df[df["Title"]==title].index.values.astype(int)[0]

    #Function to print Similar movies based on distances
    def print_similar_movies(query=None):
       data = list()
       if query:
          found_id = get_index(query)          
          for id in indices[found_id][0:]:
              if df_descending.loc[id][x_input] == 1:
                if df_descending.loc[id]["Rating"] > 5:
                  list2=[]
                  list2.append(df_descending.loc[id]["Title"])
                  list2.append(ref_genre.loc[id]["Genre"])
                  list2.append(df_descending.loc[id]["Rating"])
                  list2.append(df_descending.loc[id][x_input]) 
                  data.append(list2)  
                  list2 = None
       return data
           #for id in indices[0]:
            #similar_articles.append(articles[index].title)


    # In[28]:


    rank = Ranks[0]
    movie_input = np.array(df_descending[df_descending['Rank']==rank].Title)
    movie = movie_input[0]
    #print(movie)


    # In[29]:


    #rating = Rating[0]
    title = np.array(df[df['Rank']==rank].index)
    #print(title)


    # In[30]:

    data=list()
    data=print_similar_movies(movie)


    # In[31]:

    exit1=tk.Label(window3,text='Result is shown in New Window')
    canvas5.create_window(290,220,window=exit1)

    def onclick():
        window4.destroy()
        window3.destroy()
        raise SystemExit

    def onclick1():
        window4.destroy()
    
    print('Your Input genre is ',x_input)
    print('Recommending top movies of ',x_input,' Genre : \n\n')
    headers = ['Title', 'Genres', 'Rating', 'Is {}'.format(x_input)]
    data = sorted(data,key=lambda lr:lr[2], reverse=True)
    l3=tk.Label(window4,text='Recommended Movie')
    canvas6.create_window(200,220,window=l3)
    l4=tk.Label(window4,text='Rating')
    canvas6.create_window(430,220,window=l4)


    
    for i in range(0,len(data)):
        if i > 10:
            break
        tab=tk.Label(window4,text=data[i][0])
        canvas6.create_window(200,250+(20*i),window=tab)

        tab=tk.Label(window4,text=data[i][1])

     
        canvas6.create_window(430,250+(20*i),window=tab)


    bt22=tk.Button(window4,text='New Genre',command=onclick1)
    canvas6.create_window(250, 480, window=bt22)
    
    
    bt2=tk.Button(window4,text='Close And Exit',command=onclick)
    canvas6.create_window(350, 480, window=bt2)
   
    window4.mainloop()


bt1=tk.Button(window3,text='Show Movies',command=find)
bt1.pack()
canvas5.create_window(290, 180, window=bt1)


window3.mainloop()




# In[ ]:




