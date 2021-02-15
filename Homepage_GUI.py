choice = 0
while choice !=3:
    choice = int(input("\n\n******Enter Choice:******\n 1.K-Means Clustering of Movies\n 2.Movie Recommendation Using SVM Or KNN\n 3.Exit\n"))
    if choice == 1:
        import Clustering
   
   
    elif choice == 2:
        import tkinter as tk
        
        windowh=tk.Tk()

        windowh.title('Homepage')

        canvash = tk.Canvas(windowh, width = 600, height = 600)

        enth=tk.Label(windowh,text='Enter A Choice')
        canvash.create_window(300,150,window=enth)


        def onclickh2():
            import SVM
        

        def onclickh3():
            import KNN

        def onclickh4():
            windowh.destroy()    
            raise SystemExit   


        bth2=tk.Button(windowh,text='SVM',command=onclickh2)
        canvash.create_window(250, 200, window=bth2)

        bth3=tk.Button(windowh,text='KNN',command=onclickh3)
        canvash.create_window(350, 200, window=bth3)

        bth4=tk.Button(windowh,text='Close And Exit',command=onclickh4)
        canvash.create_window(300, 250, window=bth4)

        canvash.pack()
        windowh.mainloop()

    elif choice == 3:
        pass


    else:
        print("Enter Valid Choice!")
