# THIS is the module for running smote and it's variants on different datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTEN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import random

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



class Binclassification:
    def __init__(self,X,y,numcol,catcol):
        self.X = X
        self.y = y
        self.numcol = numcol
        self.catcol = catcol
        self.simulation_result_n={}
        self.simulation_result_one={}
        self.data_sim_one={}
        self.data_sim_n={}
        self.validate_and_assign_columns()
        
        
    def validate_and_assign_columns(self):
        numeric_cols=[]
        cat_cols=[]
        for Num,Cat,Col in zip(self.numcol,self.catcol,self.X.columns.tolist()):
            if(Num==1 and Cat==1):
                raise Exception("Column could not be both cat and num.")

            if(Num==1):
                numeric_cols.append(Col)
            elif(Cat==1):
                cat_cols.append(Col)
            else:
                raise Exception("A column has to be cat/numeric.")
        self.numcol=numeric_cols
        self.catcol=cat_cols


    
    
    
    def preprocess(self, EncodeCat=False,EncodeLabel=False,seed=42,sampling=2):
        
        '''
        other logic would be apply one hot and std scaler the in last apply smote
        then it would have 1 final pipline 
        
        '''

        '''
        Docstring for preprocess
        
        :param EncodeCat: by default false
        :param EncodeLabel: by default false
        :param seed: 
        :param sampling: 0 for no sampling, 1 for undersampling, 2 for oversampling
        '''
        
        # label encoding
        if EncodeLabel:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(self.y)
        else:
            y_encoded = self.y
        
        X_train,X_test,y_train,y_test=train_test_split(self.X,y_encoded,random_state=seed,test_size=0.3, stratify=y_encoded)

        
        # standardization
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        
        scale_transformer = ColumnTransformer(
            transformers=[
            ("num", numeric_pipeline, self.numcol)
            ],
            remainder="passthrough"   # keep categorical untouched
            )
        
        X_train_scaled = scale_transformer.fit_transform(X_train)
        X_test_scaled  = scale_transformer.transform(X_test)
        
        original_cols = list(self.X.columns)
        remainder_cols = [col for col in original_cols if col not in self.numcol]

        cat_indices = [
            len(self.numcol) + remainder_cols.index(cat_col)
            for cat_col in self.catcol
        ]
        
        print('Before Sampling train_X: {}'.format(X_train_scaled.shape))
        print('Before Sampling train_y: {} \n'.format(y_train.shape))


        if sampling == 0:
            X_train_res, y_train_res = X_train_scaled, y_train

        elif sampling == 1:   # SMOTE
            smote = SMOTE(random_state=seed)
            X_train_res, y_train_res = smote.fit_resample(
                X_train_scaled, y_train
            )
            print("SAMPLING WITH SMOTE")

        elif sampling == 2:   # SMOTETomek
            smote = SMOTETomek(random_state=seed)
            X_train_res, y_train_res = smote.fit_resample(
                X_train_scaled, y_train
            )
            print("SAMPLING WITH Smote Tomek")

        elif sampling == 3:   # SMOTEENN
            smote = SMOTEENN(random_state=seed)
            X_train_res, y_train_res = smote.fit_resample(
                X_train_scaled, y_train
            )
            print("SAMPLING WITH Smote ENN")

        elif sampling == 4:   # BorderlineSMOTE-1
            smote = BorderlineSMOTE(kind="borderline-1", random_state=seed)
            X_train_res, y_train_res = smote.fit_resample(
                X_train_scaled, y_train
            )
            print("SAMPLING WITH BorderlineSMOTE-1")

        elif sampling == 5:   # BorderlineSMOTE-2
            smote = BorderlineSMOTE(kind="borderline-2", random_state=seed)
            X_train_res, y_train_res = smote.fit_resample(
                X_train_scaled, y_train
            )
            print("SAMPLING WITH BorderlineSMOTE-2")

        elif sampling == 6:   # KMeansSMOTE
            smote = KMeansSMOTE(
                random_state=seed,
                k_neighbors=3,
                cluster_balance_threshold=0.01
            )
            X_train_res, y_train_res = smote.fit_resample(
                X_train_scaled, y_train
            )
            print("SAMPLING WITH KMeans SMOTE")

        elif sampling == 7:
            try:
                smote = ADASYN(random_state=seed)
                X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
                print("SAMPLING WITH ADASYN")
            except ValueError:
                print("ADASYN skipped - no samples generated")
                X_train_res, y_train_res = X_train_scaled, y_train

        else:
            raise ValueError("invalid sampling input")
        
        
        print('After Sampling train_X: {}'.format(X_train_res.shape))
        print('After Sampling train_y: {} \n'.format(y_train_res.shape))

        
        # numeric_processor=Pipeline(
        #     steps=[("imputation_mean",SimpleImputer(missing_values=np.nan,strategy="mean")),
        #            ("scaler",StandardScaler())]
        # )

        
        
        # one hot 
        cat_processor=make_pipeline(
            OneHotEncoder(handle_unknown="ignore")            
        )
        cat_transformer = cat_processor if EncodeCat else "passthrough"
        
        Cat_transformation= ColumnTransformer([
            ('cat', cat_transformer, cat_indices)],
            remainder='passthrough')
        
        x_train_final = Cat_transformation.fit_transform(X_train_res)
        x_test_final  = Cat_transformation.transform(X_test_scaled)
        
        return x_train_final,x_test_final,y_train_res,y_test
    
    
    
    def _run_simulation(self, EncodeCat, EncodeLabel, seed, sampling):
            X_train, X_test, y_train, y_test = self.preprocess(EncodeCat,EncodeLabel,seed,sampling)
            
            results = {}

            models = {
                "LogisticRegression": (
                    LogisticRegression(max_iter=5000),
                    {
                        'C': np.logspace(-4, 4, 20),
                        "penalty": ["l1", "l2"],
                        "solver": ["liblinear"]
                    }
                ),
                # "SVC": (
                #     SVC(probability=True,max_iter=1000),
                #     {
                #         'C': [0.1, 1, 10],
                #         'gamma': [0.01, 0.1, 1],
                #         'kernel': ['linear',"rbf"]
                #     }
                # ),
                # "DecisionTree": (
                #     DecisionTreeClassifier(),
                #     {
                #         'max_depth': range(1, 15),
                #         'min_samples_leaf': range(1, 20, 2),
                #         'min_samples_split': range(2, 20, 2),
                #         'criterion': ["entropy", "gini"]
                #     }
                # ),
                "RandomForest": (
                    RandomForestClassifier(),
                    {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 5, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                ),
                "AdaBoost": (
                    AdaBoostClassifier(),
                    {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 1]
                    }
                ),
            }
            

            for name, (model, params) in models.items():
                print("inside the loop 1")

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=StratifiedKFold(10),
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=True
                )

                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_

                y_pred = best_model.predict(X_test)

                if hasattr(best_model, "predict_proba"):
                    y_prob = best_model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = best_model.decision_function(X_test)

                acc = accuracy_score(y_test, y_pred)
                sensitivity = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)

                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                specificity = tn / (tn + fp)

                results[name] = {
                    "best_estimator": grid.best_estimator_,
                    "best_score": grid.best_score_,
                    "best_params": grid.best_params_,
                    "test_result": {
                        "Accuracy": acc,
                        "Sensitivity": sensitivity,
                        "Specificity": specificity,
                        "Precision": precision,
                        "F1-score": f1,
                        "AUC": auc
                    }
                }
                
            data = {
            "X_train":X_train,
            "X_test":X_test,
            "y_train":y_train,
            "y_test":y_test
            }

            return data, results


    def simulate_one(self, EncodeCat=False, EncodeLabel=False, seed=42, sampling=2):

        data, results = self._run_simulation(
            EncodeCat, EncodeLabel, seed, sampling
        )

        self.data_sim_one[f"seed {seed}"] = data
        self.simulation_result_one[f"seed {seed}"] = results

        # return results
        
            


    
    def simulate_ntimes(self, n=5, EncodeCat=False, EncodeLabel=False, sampling=2):

        seeds = 42
        for i in range(1,8):

            data, results = self._run_simulation(
                EncodeCat, EncodeLabel, seed=42, sampling=i
            )

            self.data_sim_n[f"sampling {i}"] = data
            self.simulation_result_n[f"sampling {i}"] = results

        # return self.simulation_result_n

    # def preprocess(self, EncodeCat=False,EncodeLabel=False,seed=42,sampling=2):

    #     '''
    #     Docstring for preprocess
        
    #     :param EncodeCat: by default false
    #     :param EncodeLabel: by default false
    #     :param seed: 
    #     :param sampling: 0 for no sampling, 1 for undersampling, 2 for oversampling
    #     '''
        
        
        
    #     X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,random_state=seed,test_size=0.3, stratify=self.y)
        
    #     numeric_pipeline = Pipeline(steps=[
    #         ("imputer", SimpleImputer(strategy="mean")),
    #         ("scaler", StandardScaler())
    #     ])
    #     # numeric_processor = make_pipeline(
    #     #     SimpleImputer(strategy="mean"),
    #     #     StandardScaler()
    #     #     )
    #     scale_transformer = ColumnTransformer(
    #         transformers=[
    #         ("num", numeric_pipeline, self.numcol)
    #         ],
    #         remainder="passthrough"   # keep categorical untouched
    #         )
        
    #     X_train_scaled = scale_transformer.fit_transform(X_train)
    #     X_test_scaled  = scale_transformer.transform(X_test)

    #     if sampling==0:
            
    #         X_train_res,y_train_res=X_train,y_train
            
    #     if sampling==1:
            
    #         rus = RandomUnderSampler(sampling_strategy='majority')
    #         X_train_res, y_train_res = rus.fit_resample(X_train_scaled, y_train)
            
    #     elif sampling==2:
    #         cat_indices = self.X.columns.get_indexer(self.catcol)
    #         smote = SMOTENC(
    #             categorical_features=cat_indices
    #             )
    #         X_train_res, y_train_res = smote.fit_resample(
    #             X_train_scaled, y_train)
            
    #     else:
    #         raise "invalid sampling input"
        
        
    #     print('After Sampling train_X: {}'.format(X_train_res.shape))
    #     print('After Sampling train_y: {} \n'.format(y_train_res.shape))

        
    #     # numeric_processor=Pipeline(
    #     #     steps=[("imputation_mean",SimpleImputer(missing_values=np.nan,strategy="mean")),
    #     #            ("scaler",StandardScaler())]
    #     # )

        

    #     cat_processor=make_pipeline(
    #         OneHotEncoder(handle_unknown="ignore")            
    #     )
    #     cat_transformer = cat_processor if EncodeCat else "passthrough"
        
    #     Cat_transformation= ColumnTransformer([
    #         ("num",numeric_pipeline,self.numcol)
    #         ('cat', cat_transformer, self.catcol)])
        
    #     x_train_final = Cat_transformation.fit_transform(X_train_res)
    #     x_test_final  = Cat_transformation.transform(X_test_scaled)
    #     return x_train_final,x_test_final,y_train_res,y_test
    
