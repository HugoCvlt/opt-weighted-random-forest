import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from scipy import optimize

class Opt_WRF():

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
                e_tilde : np.array(Mn), estimator of true error
                w0 : np.array(Mn), initial weights
                w_star : np.array(Mn), first weights computed by minimizing C_0n
                w_tilde : np.array(Mn), final weights
                random_forest: list[DecisionTreeRegressor], list of all our trained decision trees
        """

        self.X = X
        self.y = y
        self.Mn = Mn
        self.n_min = n_min
        self.n = len(X)
        self.array_P_BL = np.array([])
        self.e_tilde = np.empty(self.Mn)
        self.w0 = 1/self.Mn * np.ones(self.Mn)
        self.w_star = np.empty(self.Mn)
        self.w_tilde = np.empty(self.Mn)
        self.random_forest = []


    def constraint_sum_w(self, w):
        """
            Constraint function for the optimization
            Sum_i w_i = 1
        """

        return np.sum(w) - 1


    def c_0n(self, w):
        """
            Function to minimize to obtain the first set of weights w_star

            Input: w : np.array(Mn)
        """

        #P(w)
        p_w = np.empty((self.n,self.n))

        for i in range(self.Mn):
            p_w += w[i]*self.array_P_BL[i]

        #Fisrt norm
        first_term = np.linalg.norm(self.y-p_w@self.y)**2 

        #Sigma
        p_w0 = np.sum((1/self.Mn)*self.array_P_BL, axis=0)
        sigma = (1/self.n) * np.linalg.norm(self.y-p_w0@self.y)**2

        #P_ii(w)
        p_ii = np.trace(p_w)

        return first_term + 2*sigma*p_ii
    

    def c_second_n(self, w):
        """
            Function to minimize to obtain the final set of weights w_tilde

            Input: w : np.array(Mn)
        """

        #P(w)
        p_w = np.empty((self.n,self.n))

        for i in range(self.Mn):
            p_w += w[i]*self.array_P_BL[i]

        #First norm
        first_term = np.linalg.norm(self.y-p_w@self.y)**2

        #Second term(w)
        second_term = 0
        for i in range(self.n):
            second_term += self.e_tilde[i]**2 * p_w[i,i]

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

            dt = DecisionTreeRegressor(criterion='absolute_error', min_samples_split=self.n_min)

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
            matrix_P_BL = np.empty((self.n, self.n))

            #For each data (i = 1 to n)
            for i in range(self.n): 
                x_i = self.X.iloc[i,]  

                #Drop the data x_i in the tree m, and compute P_BL(m)(x_i)
                end_leaf = int(dt.apply(np.array(x_i).reshape(1,-1))[0])
                n_l = int(df_n_l[df_n_l["index_leaf"]==end_leaf]['n_l'].iloc[0])

                #Get the other node present in the same end leaf
                apply_boost = dt.apply(X_bootstrap.values)
                boost_in_end_leaf = [index for index, value in enumerate(apply_boost) if value==end_leaf]

                #Compute x_i_freq which is equivalent to h_i
                x_i_freq = dict(enumerate(np.bincount(bootstrap_indices, minlength=self.n)))

                #Compute P_BL(x_i)
                P_BL = np.zeros(self.n)
                for node_index in boost_in_end_leaf:
                    P_BL[node_index-1] = x_i_freq[node_index]/n_l

                #Store the result in our matrix
                matrix_P_BL[i] = P_BL
                
            list_P_BL.append(matrix_P_BL)
            self.random_forest.append(dt)

            end_pbl = time.time()
            if verbose: print("Time to compute P_BL(",m,") :", round(end_pbl-start_bpl,1), end="\n")
        
        # END OF TREE CREATION

        self.array_P_BL = np.array(list_P_BL)

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
                                       options = {'maxiter': 10, 'disp': True},
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

        self.e_tilde = np.array((np.identity(self.n) - p_w_star)@self.y)
        

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
                                        options = {'maxiter': 10, 'disp': True},
                                        method='SLSQP')
        end_pb2 = time.time()

        self.w_tilde = res_w_tilde.x
        
        if verbose:
            print("w_tilde = ", res_w_tilde.x)
            print("Solve second pb takes :", round(end_pb2-start_pb2))

    def validate(self, X_validation, y_validation):
     
        rf_mae = 0
        opt_mae = 0
        rf_rmse = 0
        opt_rmse = 0
        n = len(X_validation)

        for i in range(len(X_validation)):
            x_val = X_validation.iloc[i,]
            y_val = y_validation.iloc[i,]
            y_rf = 0
            y_opt_weighted = 0
            for m in range(self.Mn):
                tree = self.random_forest[m]
                y_hat = tree.predict(np.array(x_val).reshape(1, -1))
                y_rf += (1/self.Mn) *y_hat
                y_opt_weighted += self.w_tilde[m]*y_hat

            rf_mae += np.abs(y_val-y_rf)
            opt_mae += np.abs(y_val-y_opt_weighted)
            rf_rmse += (y_val-y_rf)**2
            opt_rmse += (y_val-y_opt_weighted)**2


        return rf_mae/n, opt_mae/n, np.sqrt(rf_rmse/n), np.sqrt(opt_rmse/n)