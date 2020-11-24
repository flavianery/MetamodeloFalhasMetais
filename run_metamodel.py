#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Escola Politecnica da Universidade de São Paulo
Departamento de Engenharia Mecatrônica e de Sistemas Mecânicos - PMR

@author: Flávia Piñeiro Nery and Matheus Alves Ivanaga
@advisor: Larissa Driemeier

This script contains the functions used to generate a failure analysis metamodel.

Use run() function to configure and call the plotting functions.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve, KFold, cross_val_predict,cross_val_score
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def PolynomialRegression(degree=1,**kwargs):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression(**kwargs))

def ElasticNetRegression(degree=1,alpha=1,l1_ratio=0.5):
    return make_pipeline(PolynomialFeatures(degree),ElasticNet(alpha=alpha,l1_ratio=l1_ratio,normalize=True))
    
       
########################################################################################
# Fitting Polynomial Regression to the dataset
########################################################################################
@ignore_warnings(category=ConvergenceWarning)
def PolyReg(x,y,deg,seed=0,splits_start=2, splits_stop=8, savefig=False, elasticnet=False):
    splits_range = range(splits_start,splits_stop+1)    
    rows = len(splits_range)//2    
    fig, ax = plt.subplots(rows, 2,constrained_layout=True,figsize=(10,10))
    
    if elasticnet:
        poly_reg = ElasticNetRegression(degree=deg,alpha=elasticnet[0],l1_ratio=elasticnet[1])
        if deg == 1:
            fig.suptitle('Regressão Linear (alfa={},beta={})'.format(elasticnet[0],elasticnet[1]), fontsize=16)
        else:
            fig.suptitle('Regressão Polinomial Grau {} (alfa={},beta={})'.format(deg,elasticnet[0],elasticnet[1]), fontsize=16)
    else:
        poly_reg = PolynomialRegression(degree=deg)
        if deg == 1:
            fig.suptitle('Regressão Linear', fontsize=16)
        else:
            fig.suptitle('Regressão Polinomial Grau {}'.format(deg), fontsize=16)
            
    k=0
    pred_color = [1,0.9-(deg%5)/6,(deg%5)/6]
    for i in range(rows):
        for j in range(2):
            cv_splits = splits_range[k]
            k+=1
            cv = KFold(n_splits=cv_splits, random_state=seed, shuffle=True)
            best_score = -1000
            poly_reg_scores = []
            for train_index, test_index in cv.split(x):            
                X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
                poly_reg.fit(X_train, y_train)
                r2_score = poly_reg.score(X_test, y_test)
                poly_reg_scores.append(r2_score)
                
                if r2_score > best_score:
                    X_train_best, X_test_best, y_train_best, y_test_best = (X_train, X_test, y_train, y_test)
                    best_score = r2_score
            
            poly_reg.fit(X_train_best, y_train_best)
            y_pred = poly_reg.predict(X_test_best)
            if elasticnet:
                coef = poly_reg.named_steps['elasticnet'].coef_
                intercept = poly_reg.named_steps['elasticnet'].intercept_
                coef[0] = intercept
            else:
                coef = poly_reg.named_steps['linearregression'].coef_[0]
                coef[0] = poly_reg.named_steps['linearregression'].intercept_
            poly_curve = np.poly1d(np.flip(coef))
            x_range = np.linspace(-0.4,0.4,50)
            ax[i,j].plot(x_range,poly_curve(x_range),color=pred_color,label = "Previsões do modelo")
            ax[i,j].scatter(X_test_best, y_pred, color=pred_color)
            ax[i,j].scatter(X_test_best, y_test_best,color='navy',label = "Dados de Teste")
            ax[i,j].scatter(X_train_best, y_train_best, color='lightblue', label="Dados de Treino")
            ax[i,j].set_title('Divisões: {}  R²: {:.5f}'.format(cv_splits,best_score))
            ax[i,j].set_xlabel('Triaxialidade')
            ax[i,j].set_ylabel('Deformação Plástica Equiv.')
            ax[i,j].legend()
            
    if savefig:
        if elasticnet:
            plt.savefig('polyreg_deg{}_alpha{}_beta{}.png'.format(deg,elasticnet[0],elasticnet[1]), bbox_inches='tight')
        else:
            plt.savefig('polyreg_deg{}.png'.format(deg), bbox_inches='tight')
    plt.show()
    return cv.split(x)


########################################################################################
# Plotting Polynomial Regression validation surface
########################################################################################
def Poly_Validation_Surfaces(x,y,deg_range=np.arange(1, 7),seed=0,savefig=False):
    split_range=np.arange(2,11)
    test_scores=np.zeros((len(split_range),len(deg_range)))
    train_scores=np.zeros((len(split_range),len(deg_range)))
    for cv_splits in split_range:
        cv = KFold(n_splits=cv_splits, random_state=seed, shuffle=True)
        train_out, test_out = validation_curve(PolynomialRegression(), x, y,'polynomialfeatures__degree', deg_range,cv=cv,scoring='r2')
        test_scores[cv_splits-split_range[0]] = np.copy(np.mean(test_out, axis=1))
        train_scores[cv_splits-split_range[0]] = np.copy(np.mean(train_out, axis=1))
        
    for i in range(test_scores.shape[0]):
        for j in range(test_scores.shape[1]):
            test_scores[i,j] = 0 if test_scores[i,j] <=0 else test_scores[i,j]
            
        
    X_plot, Y_plot = np.meshgrid(deg_range,split_range)
        
    max_id= (np.where(test_scores == np.amax(test_scores))[0][0],np.where(test_scores == np.amax(test_scores))[1][0])

    fig = plt.figure(figsize=(15,10))
    ax = fig.gca(projection='3d')
    
    # Plot the surface.
    surf = ax.plot_surface(X_plot, Y_plot, test_scores, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.suptitle('Validation Surface', fontsize=16)
    ax.set_xlabel('graus')
    ax.set_ylabel('nº de divisões')
    ax.set_zlabel('Score médio da validação cruzada')
    ax.set_title('Score Máx.: {:.5f} -> [grau = {}, nº de divisões = {}]'.format(test_scores[max_id],deg_range[max_id[1]],split_range[max_id[0]]))

    if savefig:
        plt.savefig('val_surf.png', bbox_inches='tight')
    plt.show()
    
    return train_scores, test_scores

########################################################################################
# Plotting ElasticNet Regression validation matrix
########################################################################################
@ignore_warnings(category=ConvergenceWarning)
def Regularization_Matrix(x,y,deg_range=np.arange(1, 11),split_range=np.arange(2,9),seed=0,savefig=False):
    alpha_range=np.linspace(0.00001,0.1,11)
    penalty_range=np.linspace(0,1,11)
    lin = len(alpha_range)
    col = len(penalty_range)
    param_matrix=np.zeros((lin,col))
    param_matrix_labels=np.zeros((lin,col), dtype = 'object')
    test_scores=np.zeros((len(split_range),len(deg_range)))
    train_scores=np.zeros((len(split_range),len(deg_range)))
    i,j,max_param=0,0,0
    for alpha in alpha_range:
        for beta in penalty_range:
            for cv_splits in split_range:
                cv = KFold(n_splits=cv_splits, random_state=seed, shuffle=True)
                train_out, test_out = validation_curve(ElasticNetRegression(alpha=alpha,l1_ratio=beta), x, y,'polynomialfeatures__degree', deg_range,cv=cv,scoring='r2')
                test_scores[cv_splits-split_range[0]] = np.copy(np.mean(test_out, axis=1))
                train_scores[cv_splits-split_range[0]] = np.copy(np.mean(train_out, axis=1))
                
            max_id= (np.where(test_scores == np.amax(test_scores))[0][0],np.where(test_scores == np.amax(test_scores))[1][0])
            param_matrix[i,j]=test_scores[max_id]
            param_matrix_labels[i,j]=("{:.4f}\n({}g,{}d)".format(test_scores[max_id],deg_range[max_id[1]],split_range[max_id[0]]))
            
            if max_param < param_matrix[i,j]:
                max_param = param_matrix[i,j]
                max_param_label = "Melhor resultado: {:.5f} -> [alfa = {},beta = {}]".format(max_param,alpha,beta)
            j+=1
            
        j=0
        i+=1
    
    fig, ax = plt.subplots(figsize=(15,10))
    
    mat = ax.imshow(param_matrix, cmap=plt.cm.coolwarm,extent =[0,col,0,lin])
    
    param_matrix_labels = np.flip(np.transpose(param_matrix_labels),1)
    for i in range(col):
        for j in range(lin):
            ax.text(i+0.5,j+0.5, param_matrix_labels[i,j], va='center', ha='center')
            
 
    fig.colorbar(mat).set_label('Score')
    # fig.suptitle(max_param_label, fontsize=16,x=0.6)
    ax.set_xticklabels([("(Ridge)\n{:.1f}".format(x) if x==0 else "(Lasso)\n{:.1f}".format(x) if x==1 else "{:.1f}".format(x)) for x in penalty_range])
    ax.set_yticklabels(["{:.3f}".format(x) for x in alpha_range][::-1])
    ax.set_xticks(np.linspace(0.5,col-0.5,col))
    ax.set_yticks(np.linspace(0.5,lin-0.5,lin))
    ax.set_xlabel(r"$\beta$",fontsize=14)
    ax.set_ylabel(r"$\alpha$",fontsize=14)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_title("(g = grau do polinômio, d = número de divisões)",y=-0.05)
   
    
    if savefig:
        plt.savefig('val_mat.png', bbox_inches='tight')
    
       

########################################################################################
# Fitting Polynomial Regression to the dataset
########################################################################################
@ignore_warnings(category=ConvergenceWarning)
def PolyReg3D(x,y,deg,seed=0,splits_start=2, splits_stop=8, savefig=False,
              elasticnet=False, trd_val_name="Ângulo de Lode",trd_val_lim=(-1,1)):
    splits_range = range(splits_start,splits_stop+1)    
    rows = len(splits_range)//2   
    fig = plt.figure(figsize=(12,10))
    
    if elasticnet:
        poly_reg = ElasticNetRegression(degree=deg,alpha=elasticnet[0],l1_ratio=elasticnet[1])
        if deg == 1:
            fig.suptitle('Regressão Linear (alfa={},beta={})'.format(elasticnet[0],elasticnet[1]), fontsize=16)
        else:
            fig.suptitle('Regressão Polinomial Grau {} (alfa={},beta={})'.format(deg,elasticnet[0],elasticnet[1]), fontsize=16)
    else:
        poly_reg = PolynomialRegression(degree=deg)
        if deg == 1:
            fig.suptitle('Regressão Linear', fontsize=16)
        else:
            fig.suptitle('Regressão Polinomial Grau {}'.format(deg), fontsize=16)
            
    k=0
    pred_color = [1,0.9-(deg%5)/6,(deg%5)/6]
    for i in range(rows):
        for j in range(2):
            cv_splits = splits_range[k]
            k+=1
            cv = KFold(n_splits=cv_splits, random_state=seed, shuffle=True)
            best_score = -10000
            poly_reg_scores = []
            for train_index, test_index in cv.split(x):            
                X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
                poly_reg.fit(X_train, y_train)
                r2_score = poly_reg.score(X_test, y_test)
                poly_reg_scores.append(r2_score)
                
                if r2_score > best_score or (i,j)==(0,0):
                    X_train_best, X_test_best, y_train_best, y_test_best = (X_train, X_test, y_train, y_test)
                    best_score = r2_score
            
            poly_reg.fit(X_train_best, y_train_best)
            y_pred = poly_reg.predict(X_test_best)
            if elasticnet:
                coef = poly_reg.named_steps['elasticnet'].coef_
                coef[0] = poly_reg.named_steps['elasticnet'].intercept_[0]
            else:
                coef = poly_reg.named_steps['linearregression'].coef_[0]
                coef[0] = poly_reg.named_steps['linearregression'].intercept_[0]
                
             
            ax = fig.add_subplot(rows,2,k,projection='3d')
            # m=0
            # n=0
            # for i in range(X_train_best.shape[0]):
            #     if X_train_best[i,0] < 0:
            #         m=i
            #     elif X_train_best[i,0] < 0.4:
            #         n=i
                
            # ax.scatter(X_train_best[n+1:,0], X_train_best[n+1:,1], y_train_best[n+1:], marker='.', color='magenta',s=100, label="Triax. > 0.4")
            # ax.scatter(X_train_best[m+1:n+1,0], X_train_best[m+1:n+1,1], y_train_best[m+1:n+1], marker='.', color='darkviolet',s=100, label="Triax. > 0")
            # ax.scatter(X_train_best[:m+1,0], X_train_best[:m+1,1], y_train_best[:m+1], marker='.', color='red',s=100, label="Dados de Treino")
            ax.scatter(X_train_best[:,0], X_train_best[:,1], y_train_best, marker='.', color='red',s=100, label="Dados do treino")
            ax.scatter(X_test_best[:,0], X_test_best[:,1], y_test_best, marker='.', color='green',s=100,label = "Dados do teste")
            ax.scatter(X_test_best[:,0], X_test_best[:,1], y_pred, marker='.', color='blue',s=100,label = "Previsões do modelo")
            ax.set_xlabel("Triaxialidade")
            ax.set_ylabel(trd_val_name)
            ax.set_zlabel("Deformação Plástica Equiv.")
            xs = np.linspace(-0.6,0.5,50)
            ys = np.linspace(trd_val_lim[0],trd_val_lim[1],50)
            X,Y = np.meshgrid(xs,ys)
            inpt = np.column_stack((X.ravel(),Y.ravel()))
            poly_reg.fit(X_train_best, y_train_best)
            Z = poly_reg.predict(inpt)
            Z[Z<-1]=np.nan
            Z[Z<-0.5]=-0.5
            Z=Z.reshape((50,50))            
            plt.locator_params(axis='y', nbins=5)
            ax.plot_surface(X,Y,Z, alpha=0.5)
            ax.view_init(azim=40)
            ax.legend(loc="upper right",bbox_to_anchor=(0.9, 0.95))
            ax.set_title('Divisões: {}  R²: {:.5f}'.format(cv_splits,best_score))
            
    plt.tight_layout(rect=[0,0,1,0.95])    
    if savefig:
        if elasticnet:
            if trd_val_name=="Ângulo de Lode":
                plt.savefig('polyreg3d_lode_deg{}_alpha{}_beta{}.png'.format(deg,elasticnet[0],elasticnet[1]), bbox_inches='tight',pad_inches=0.2)
            else:    
                plt.savefig('polyreg3d_deg{}_alpha{}_beta{}.png'.format(deg,elasticnet[0],elasticnet[1]), bbox_inches='tight',pad_inches=0.2)
        else:
            if trd_val_name=="Ângulo de Lode":
                plt.savefig('polyreg3d_lode_deg{}.png'.format(deg), bbox_inches='tight',pad_inches=0.2)
            else:
                plt.savefig('polyreg3d_deg{}.png'.format(deg), bbox_inches='tight',pad_inches=0.2)
    plt.show()
    return None

###############################################################################
#RUN PLOTTING FUNCTIONS
#Functions for 2d dataset:
#- PolyReg
#- Poly_Validation_Surfaces
#Functions for 3d dataset:
#- PolyReg3D
#Functions for both datasets:
#- Regularization_Matrix
###############################################################################
def run(s):
    # Fixing random state for reproducibility
    np.random.seed(s)

    #############################	
    #Importing the dataset
    #
    #For 2d model: comment lines with #3d
    #For 3d model: comment lines with #2d 
    #############################
    # dataset=pd.read_csv('PontosMetamodelo_19.csv',sep=';') #2d
    dataset=pd.read_csv('PontosMetamodeloComLode.csv',sep=';') #3d
    # x = dataset['x'].to_numpy().reshape(-1, 1) #2d
    x = np.zeros((len(dataset['x']),2)) #3d
    x[:,0] = dataset['x'].to_numpy() #3d
    x[:,1] = dataset['z'].to_numpy() #3d
    y = dataset['y'].to_numpy().reshape(-1, 1) 
    
    ####################################
    # Plotting triaxiality vs lode angle
    ####################################
    # plt.figure(figsize=(8,6))
    # plt.scatter(x[:,0],x[:,1])
    # plt.title("Triaxialidade x Ângulo de Lode", fontsize=16)
    # plt.xlabel("Triaxialidade")
    # plt.ylabel("Ângulo de Lode")
    # plt.savefig("triax_vs_lode.png")
    # plt.show()
    
    
    PolyReg3D(x,y,deg=2,seed=s,savefig=True)
    PolyReg3D(x,y,deg=3,seed=s,elasticnet=[0.004,0],savefig=True)
    # PolyReg3D(x,y,deg=6,seed=s,elasticnet=[0.01,0],savefig=True,trd_val_name="Tensão de Escoamento",trd_val_lim=(200,300))
    # Regularization_Matrix(x,y,seed=s,savefig=True)
    # PolyReg(x,y,deg=1,seed=s)  
    # PolyReg(x,y,deg=2,seed=s,savefig=True)
    # PolyReg(x,y,deg=2,seed=s,elasticnet=[0.002,0])
    # PolyReg(x,y,deg=5,seed=s)
    # Poly_Validation_Surfaces(x,y,deg_range=np.arange(1,10),seed=s)

    
def main():
    plt.close('all')
    run(0)
    
if __name__=='__main__':
    main()