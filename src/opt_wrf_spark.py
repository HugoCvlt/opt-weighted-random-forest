import time
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy import optimize

class Opt_WRF_spark():

    def __init__(self, X, y, Mn, n_min):
        """
            Inputs:
                X : pd.DataFrame, training dataset X
                y : pd.DataFrame, label y
                Mn : int, number of tree in the forest
                n_min : int, min_samples_split of the decision tree

            Attribute:
                n : int, number of data
                array_P_BL : np.array((Mn,n,n)), contains all the matrix P_BL for each tree
                w0 : np.array(Mn), initial weights
                w_star : np.array(Mn), first weights computed by minimizing C_0n
                random_forest: list[DecisionTreeRegressor], list of all our trained decision trees
        """

        self.X = X
        self.y = y
        self.Mn = Mn
        self.n_min = n_min
        self.n = len(X)
        self.array_P_BL = np.array([])
        self.w0 = 1/self.Mn * np.ones(self.Mn)
        self.w_star = np.empty(self.Mn)
        self.random_forest = []
        self.sc = pyspark.SparkContext

    def constraint_sum_w(self, w):
        """
            Constraint function for the optimization
            Sum_i w_i = 1
        """

        return np.sum(w) - 1

    def compute_p_w(self, w):
        """
            Fonction qui utilise spark pour calculer p_w i.e la somme pondérée des matrices P_BL(m)
            Il faut que array_P_BL soit une liste de rdd contenant les triplets de la forme (i,j, m_ij)

            Input: w : np.array(Mn)

            Output: p_w : rdd de paire ((i,j), m_ij)
        """

        mat_to_sum = [self.sc.parallelize([]), self.sc.parallelize([])]
        for m in range(self.Mn):
            mat = self.array_P_BL[m]
            mat_scal = mat.map(lambda t: ((t[0],t[1]), t[2]*w[m]))
            mat_to_sum[1] = mat_scal

            temp_mat = self.sc.union(mat_to_sum)
            sum_mat = temp_mat.reduceByKey(lambda x,y:x+y)

            mat_to_sum[0] = sum_mat

        return mat_to_sum[0]

    def prdt_mat_vect(self, mat, vect):
        """
            Calcule le produit entre une matrice format rdd ((i,j),m_ij) avec un vecteur

            Inputs:
                mat: rdd contenant des paires de la forme ((i,j), m_ij)
                vect: np.array ou list pas besoin de rdd

            Ouput: np.array le résultat du produit
        """

        temp = mat.map(lambda t: (t[0][0], t[1]*vect[t[0][1]]))
        res = temp.reduceByKey(lambda x,y : x+y)
        res = res.sortBy(lambda x: x[0])

        return np.array(res.collect())[:,1]

    def rdd_trace(self, mat):
        """
            Calcule la trace de la matrice

            Inputs:
                mat: rdd contenant des paires de la forme ((i,j), m_ij)

            Ouput: int trace de la matrice
        """

        diag_element = mat.filter(lambda t: t[0][0] == t[0][1])
        trace = diag_element.map(lambda t: t[1]).reduce(lambda x,y:x+y)
        return trace

    def c_0n(self, w):
        """
            Function to minimize to obtain the first set of weights w_star

            Input: w : np.array(Mn)
        """

        #P(w)
        p_w = self.compute_p_w(w)

        pw_y = self.prdt_mat_vect(p_w, self.y)
        first_term = np.linalg.norm(self.y-pw_y)**2

        #Sigma
        p_w0 = self.compute_p_w(self.w0)
        pw0_y = self.prdt_mat_vect(p_w0, self.y)
        sigma = (1/self.n) * np.linalg.norm(self.y-pw0_y)**2

        #P_ii(w)
        p_ii = self.rdd_trace(p_w)

        return first_term + 2*sigma*p_ii
    
    def c_second_n(self, w):
        """
            Function to minimize to obtain the final set of weights w_tilde

            Input: w : np.array(Mn)
        """

        #P(w)
        p_w = self.compute_p_w(w)

        #First norm
        pw_y = self.prdt_mat_vect(p_w, self.y)
        first_term = np.linalg.norm(self.y-pw_y)**2

        #Second term(w)
        second_term = 0
        for i in range(self.n):
            second_term += self.e_tilde[i]**2 * p_w[i,i] #Recup lelement qui faut

        return first_term + 2*second_term

    def two_steps_WRF_opt(self, verbose=True):
        """
            Our implementation of the 2steps WRF_opt, introduce in the paper of Chen,Yu and Zhang

            Input: verbose : Boolean 
        """
  
        list_P_BL = []

        #Create Mn trees, and compute P_BL for each one (m = 1 to Mn)
        for m in range(self.Mn):

            if verbose: print("Creating tree", m+1)
            start_fit = time.time()

            dt = DecisionTreeRegressor(min_samples_split=self.n_min)
            
            #Bootstrap by hand
            bootstrap_indices = np.random.choice(range(self.n), size=self.n, replace=True)
            X_bootstrap = self.X.iloc[bootstrap_indices]
            y_bootstrap = self.y.iloc[bootstrap_indices]

            #Fit the tree on the bootstraped data
            dt.fit(X_bootstrap.values, y_bootstrap.values)

            end_fit = time.time()

            if verbose:
                print("Time to fit tree",m,":",round(end_fit-start_fit,1),end="\n")
                print("Computing P_BL", m+1)
            start_bpl = time.time()

            #Creation of the dataframe containing all the n_l
                #Get the index of the leaf
            leaf_nodes = dt.tree_.children_left == dt.tree_.children_right 
            leaf_indexes = [index for index, value in enumerate(leaf_nodes) if value]
                #Compute n_l for each leaf (number of samples inside the leaf)
            n_l = dt.tree_.n_node_samples[leaf_nodes]
            df_n_l = pd.DataFrame(data={"index_leaf":leaf_indexes,"n_l":n_l})

            
            #Initialisation of P_BL_(m)
            matrix_P_Bl_rdd = self.sc.emptyRDD()

            #For each data (i = 1 to n)
            for i in range(self.n): 
                x_i = self.X.iloc[i,]  

                #Drop the data x_i in the tree m, and compute P_BL(m)(x_i)
                end_leaf = dt.apply(np.array(x_i).reshape(1,-1))[0]
                n_l = int(df_n_l[df_n_l["index_leaf"]==end_leaf]['n_l'].iloc[0])

                #Get the other node present in the same end leaf
                apply_boost = dt.apply(X_bootstrap.values)
                boost_in_end_leaf = [index for index, value in enumerate(apply_boost) if value==end_leaf]

                #Compute x_i_freq which is equivalent to h_i
                x_i_freq = dict(enumerate(np.bincount(bootstrap_indices, minlength=self.n)))

                #Compute P_BL(x_i)
                for node_index in boost_in_end_leaf:
                    triplet = self.sc.parallelize([i, node_index, x_i_freq[node_index]/n_l])
                    matrix_P_Bl_rdd = self.sc.union(matrix_P_Bl_rdd, triplet)

                
            list_P_BL.append(matrix_P_Bl_rdd)
            self.random_forest.append(dt)

            end_pbl = time.time()
            if verbose: print("Time to compute P_BL(",m,") :", round(end_pbl-start_bpl,1), end="\n")
        
        # END OF TREE CREATION

        self.array_P_BL = list_P_BL

    
        #Solve a quad pb and find w_star
            # w_star = argmin_{w\inH} C^0_n = ||y-P(w)y||^2 +2*sum{sigma^2*P_ii(w)} with sigma^2 = ||y-P(w_0)y||^2 w_0=(1_Mn,...,1/Mn)
        
        print()
        if verbose: print("Searching w*")

        w_start = np.random.rand(self.Mn)
        w_start = w_start/sum(w_start)

        start_pb1 = time.time()
        res_w_star = optimize.minimize(self.c_0n, x0=w_start,
                                       constraints={'type':'eq', 'fun':self.constraint_sum_w},
                                       bounds=[(0, 1) for _ in range(self.Mn)],
                                       options = {'maxiter': 10000, 'disp': False},
                                       method='SLSQP')
        end_pb1 = time.time()

        self.w_star = res_w_star.x
        
        if verbose:
            print("w_star =", res_w_star.x)
            print("Solve first pb takes :", round(end_pb1-start_pb1))

        
        #Fix e_tilde = (I-P(w_star))y
        p_w_star = np.empty((self.n,self.n))

        for i in range(self.Mn):
            p_w_star += self.w_star[i]*self.array_P_BL[i]

        self.e_tilde = (np.identity(self.n) - p_w_star)@self.y
        

        #Solve an other pb and find w_tilde
            # w_tilde = argmin_{w\inH} C''_n = ||y-P(w)y||^2 +2*sum{e_tilde^2*P_ii(w)}
        
        print()
        if verbose: print("Searching w_tilde")
        

        w_start = np.random.rand(self.Mn)
        w_start = w_start/sum(w_start)

        start_pb2 = time.time()
        res_w_tilde = optimize.minimize(self.c_second_n, x0=w_start,
                                        constraints={'type':'eq', 'fun':self.constraint_sum_w},
                                        bounds=[(0, 1) for _ in range(self.Mn)],
                                        options = {'maxiter': 10000, 'disp': False},
                                        method='SLSQP')
        end_pb2 = time.time()

        self.w_tilde = res_w_tilde.x
        
        if verbose:
            print("w_tilde = ", res_w_tilde.x)
            print("Solve second pb takes :", round(end_pb2-start_pb2))

    def validate(self, X_validation, y_validation):

        y_rf = 0
        y_opt_weighted = 0
        rf_err = 0
        opt_err = 0

        for i in range(len(X_validation)):
            x_val = X_validation.iloc[i,]
            y_val = y_validation.iloc[i,]

            for m in range(self.Mn):
                tree = self.random_forest[m]
                y_hat = tree.predict(np.array(x_val).reshape(1, -1))

                y_rf += 1/self.Mn *y_hat
                y_opt_weighted += self.w_tilde[m]*y_hat

            #print("y/ y_rf : ", y_val," / ",y_rf, sep=' ')
            #print("y/ y_opt : ", y_val," / ",y_opt_weighted, sep=' ')

            rf_err += np.abs(y_val-y_rf)
            opt_err += np.abs(y_val-y_opt_weighted)

        return rf_err/len(X_validation), opt_err/len(X_validation)
