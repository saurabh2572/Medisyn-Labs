import pandas as pd
import numpy as np
# search thresholds for imbalanced classification
def get_proba_threshold(model, X_val, y_val):
  from numpy import arange, argmax
  from sklearn.metrics import f1_score
  
  # apply threshold to positive probabilities to create labels
  def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')
  
  # keep probabilities for the positive outcome only
  probs = model.predict_proba(X_val)[:,1]
  # define thresholds
  thresholds = arange(0, 1, 0.001)
  # evaluate each threshold
  from sklearn.metrics import f1_score
  scores = [f1_score(y_val, to_labels(probs, t)) for t in thresholds]
  # get best threshold
  ix = argmax(scores)
  best_threshold = thresholds[ix]
#   print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
  return best_threshold

  mlflow.log_artifact("/tmp/positive_rate_trend.html","plots/")

def log_data_used_for_modelling(df,X_train,y_train,X_test,y_test):
  X_train.to_parquet('/tmp/X_train.parquet')
  y_train = pd.DataFrame(y_train)
  y_train.to_parquet('/tmp/y_train.parquet')
  
  X_test.to_parquet('/tmp/X_test.parquet')
  y_test = pd.DataFrame(y_test)
  y_test.to_parquet('/tmp/y_test.parquet')


def is_ks_abnormality_detected(ks_table,verbose=True):
  visits = ks_table.positive.tolist()
  for i in range(len(visits)-1):
    if visits[i]<visits[i+1]:
      if verbose:
        pass
#         print(f'KS abnormality detected in Decile {i+2}')
      return 1, i
  return 0, None

def get_ks_table(df,y_true_col='positive',y_pred_proba_col='proba',verbose=True):
  import numpy as np
  temp_df = df[[y_true_col,y_pred_proba_col]].copy()
  n = len(temp_df)/10
  volume = []
  visited = []
  min_proba=[]
  max_proba=[]
  temp_df = temp_df.sort_values(y_pred_proba_col,ascending=False)
  for g, df in temp_df.groupby(np.arange(len(temp_df)) // n):
    volume.append(df.shape[0])
    visited.append(df[y_true_col].sum())
    min_proba.append(np.round(df[y_pred_proba_col].min(),4))
    max_proba.append(np.round(df[y_pred_proba_col].max(),4))
    
  del temp_df

  ks_table = pd.DataFrame({'decile':[i+1 for i in range(10)],
                           'volume':volume,
                           'positive':visited,
                           'min_proba':min_proba,
                           'max_proba':max_proba,
                          })
  
  ks_table['min_proba']=ks_table['min_proba'].apply(lambda x: np.round(x,4))
  ks_table['max_proba']=ks_table['max_proba'].apply(lambda x: np.round(x,4))
  
  ks_table.insert(2,'cum_volume',ks_table.volume.cumsum())
  ks_table.loc[:,'negative'] = ks_table.volume-ks_table.positive
  ks_table.loc[:,'cum_positive'] = ks_table.positive.cumsum()
  ks_table.loc[:,'cum_negative'] = ks_table.negative.cumsum()
  ks_table.loc[:,'positive_rate'] = round(100*ks_table.positive/ks_table.volume,2)
  ks_table.loc[:,'negative_rate'] = round(100*ks_table.negative/ks_table.volume,2)
  ks_table.loc[:,'cum_positive_rate'] = round(100*ks_table.cum_positive/ks_table.cum_volume,2)
  ks_table.loc[:,'cum_negative_rate'] = round(100*ks_table.cum_negative/ks_table.cum_volume,2)
  ks_table.loc[:,'capture_rate']=round(100*ks_table.cum_positive/ks_table.positive.sum(),2)
  ks_table.loc[:,'non_capture_rate']=round(100*ks_table.cum_negative/ks_table.negative.sum(),2)
  ks_table.loc[:,'ks'] = round(ks_table.capture_rate-ks_table.non_capture_rate,2)
  

  print(f'KS is {ks_table.ks.max()}% at Decile {ks_table[ks_table.ks==ks_table.ks.max()].decile.values[0]}')
  
  if verbose:
    is_ks_abnormality_detected(ks_table)
#     display(ks_table)
  return ks_table

def apply_model_and_get_ks_table(model, X_train, y_train, X_test, y_test, best_proba_threshold, y_true_col='positive', y_pred_proba_col='proba', verbose=True):
    def concat_y_true_y_pred_proba(y_true_col,y_pred_proba_col):
        data = pd.DataFrame(y_true_col.copy())
        data.columns = ['positive']
        data.loc[:,'proba'] = y_pred_proba_col
        return data
    
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba=model.predict_proba(X_test)[:, 1]




    train_data = concat_y_true_y_pred_proba(y_train, y_train_proba)
    test_data = concat_y_true_y_pred_proba(y_test, y_test_proba)


    train_ks_table = get_ks_table(train_data, verbose=verbose)
    test_ks_table = get_ks_table(test_data, verbose=verbose)


    train_ks_table.to_csv('/Users/saurabh.prajapati/Documents/Medisyn-Labs/data/train_ks_table.csv',index=False)
    test_ks_table.to_csv('/Users/saurabh.prajapati/Documents/Medisyn-Labs/data/test_ks_table.csv',index=False)




def log_model_eval_metrics( model, X_train, y_train, X_test, y_test,best_proba_threshold):
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba=model.predict_proba(X_test)[:, 1]

  
    from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score
    y_pred_train = [1 if proba>=best_proba_threshold else 0 for proba in y_train_proba]
    
    y_pred_test = [1 if proba>=best_proba_threshold else 0 for proba in y_test_proba]
    

    
    
    eval_metrics_dict={
        'f1_score': [round(fbeta_score(y_train, y_pred_train, beta=1),2),
                     round(fbeta_score(y_test, y_pred_test, beta=1),2),

                    ],
      
        'precision': [round(precision_score(y_train, y_pred_train),2),
                      round(precision_score(y_test, y_pred_test),2),

                     ],
      
        'recall': [round(recall_score(y_train, y_pred_train),2),
                   round(recall_score(y_test, y_pred_test),2),

                   ],
        
        'roc_auc': [round(roc_auc_score(y_train, y_train_proba),2),
                    round(roc_auc_score(y_test, y_test_proba),2),

                  ]
    }
    print(eval_metrics_dict)
    
    eval_metrics_df=pd.DataFrame(eval_metrics_dict)

    eval_metrics_df.insert(0,'type',['train', 'test'])
    return eval_metrics_df

# class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
#     def __init__(self, model):
#         self.model = model
    
#     def predict(self, context, model_input):
#         return self.model.predict_proba(model_input)[:, 1]

