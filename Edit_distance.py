import numpy as np

def edit_distance(w1,w2):

    # empty matrice
    m,n = len(w1),len(w2)
    df =  np.zeros((m+1,n+1))
    #print(df)

    # initialisation 1st row 1st column
    for i in range(m+1):
        df[i][0] = i
    for j in range(n+1):
        df[0][j] = j
    #print(df)

    # compute the edit-distance

    for i in range(1,m+1): 
        for j in range(1,n+1):
            if w1[i-1] == w2[j-1]: # on doit retirer 1 à i et j parce que les indices de la matrice sont décalés à cause de la colonne et la ligne d'initiation 
                df[i][j] = df[i-1][j-1]
            else :
                add = df[i][j-1] +1 #add
                delete = df[i-1][j] +1 #del
                replace = df[i-1][j-1] +1 #sub
                df[i][j] = min(add,delete,replace)
    print(df)
    (print(f"Edit-distance entre",w1, "et", w2, " est de ", df[m][n]))
    return df[m][n]

cost = edit_distance('arbre','arbitrage')

    