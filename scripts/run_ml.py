import sys
import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score, log_loss, brier_score_loss, precision_score, recall_score, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from interpret.glassbox import ExplainableBoostingClassifier
from catboost import CatBoostClassifier
import sklearn

sklearn.set_config(transform_output='pandas')

class PandasPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, cont_features, cat_features):
        self.cont_features = cont_features
        self.cat_features = cat_features
        self.medians = {}
    def fit(self, X, y=None):
        for c in self.cont_features: self.medians[c] = X[c].median()
        return self
    def transform(self, X):
        X_out = X.copy()
        for c in self.cont_features: X_out[c] = X_out[c].fillna(self.medians.get(c, 0)).astype(float)
        for c in self.cat_features: X_out[c] = X_out[c].fillna('MISSING').astype(str)
        return X_out

df = pd.read_stata("Morocco-2013-2019-2023.dta", convert_categoricals=False)

def get_state(r):
    y, k17 = r['year'], r['k17']
    if y == 2019:
        k16 = r['_2013_2019_k16']
        if k16 == 1: return 'CREDIT_APPLICANT'
        elif k16 == 2 and k17 == 1: return 'NO_FINANCING_NEED'
        elif k16 == 2 and k17 in [2, 3, 4, 5, 6]: return 'CONSTRAINED_NON_APPLICANT'
        return 'AMBIGUOUS'
    elif y == 2023:
        k162 = r['_2023_k162']
        if pd.notna(k162) and k162 in [1, 2, 3]: return 'CREDIT_APPLICANT'
        elif k162 == 4 and k17 == 1: return 'NO_FINANCING_NEED'
        elif k162 == 4 and k17 in [2, 3, 4, 5, 6]: return 'CONSTRAINED_NON_APPLICANT'
        return 'AMBIGUOUS'

df['STATE'] = df.apply(get_state, axis=1)
df = df[df['STATE'] != 'AMBIGUOUS'].copy()
classes = ['NO_FINANCING_NEED', 'CONSTRAINED_NON_APPLICANT', 'CREDIT_APPLICANT']

def clean_cont(x): return float(x) if not (pd.isna(x) or x < 0) else np.nan
def clean_cat(x): return str(int(x)) if not (pd.isna(x) or x < 0) else 'MISSING'

df['EMPLOYMENT'] = df['l1'].apply(clean_cont)
df['FIRM_AGE'] = df['year'] - df['b5'].apply(clean_cont)
df['EXPORTER'] = df['d3c'].apply(clean_cont)
df['FEMALE_OWNER'] = df['b4'].apply(clean_cat)
df['EXTERNAL_AUDIT'] = df['k21'].apply(clean_cat)
df['ELEC_OBSTACLE'] = df['c30a'].apply(clean_cat)
df['SECTOR'] = df['a4a'].apply(clean_cat)
df['REGION'] = df.apply(lambda r: clean_cat(r['a1']) if r['year'] == 2019 else clean_cat(r['a1a']), axis=1)

cont_features = ['EMPLOYMENT', 'FIRM_AGE', 'EXPORTER']
cat_features = ['FEMALE_OWNER', 'EXTERNAL_AUDIT', 'ELEC_OBSTACLE', 'SECTOR', 'REGION']
all_features = cont_features + cat_features

df['EMPLOYMENT'] = np.log1p(df['EMPLOYMENT'])
df['FIRM_AGE'] = np.log1p(df['FIRM_AGE'])

train = df[df['year'] == 2019].copy()
test = df[df['year'] == 2023].copy()
X_train = train[all_features].reset_index(drop=True)
y_train = train['STATE'].reset_index(drop=True)
X_test = test[all_features].reset_index(drop=True)
y_test = test['STATE'].reset_index(drop=True)

lr_preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), cont_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_features)
])

def brier_multi(y_true, y_prob):
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=classes)
    return np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1))

def get_metrics(y_t, y_p, y_prob):
    res = {'Macro_F1': f1_score(y_t, y_p, average='macro'), 'Balanced_Acc': balanced_accuracy_score(y_t, y_p), 'LogLoss': log_loss(y_t, y_prob, labels=classes), 'Brier': brier_multi(y_t, y_prob)}
    for k in classes:
        res[f'Precision_{k}'] = precision_score(y_t, y_p, labels=[k], average='macro', zero_division=0)
        res[f'Recall_{k}'] = recall_score(y_t, y_p, labels=[k], average='macro', zero_division=0)
        res[f'F1_{k}'] = f1_score(y_t, y_p, labels=[k], average='macro', zero_division=0)
    return res

models_dict = {
    'LR': Pipeline([('prep', lr_preprocessor), ('clf', LogisticRegression(penalty='l2', max_iter=1000, random_state=42))]),
    'EBM': Pipeline([('prep', PandasPreprocessor(cont_features, cat_features)), ('clf', ExplainableBoostingClassifier(random_state=42, n_jobs=1))]),
    'CB': Pipeline([('prep', PandasPreprocessor(cont_features, cat_features)), ('clf', CatBoostClassifier(loss_function='MultiClass', random_seed=42, verbose=False, thread_count=1))])
}

param_grids = {
    'LR': {'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__solver': ['lbfgs']},
    'EBM': {'clf__learning_rate': [0.01, 0.03, 0.05], 'clf__max_bins': [64, 128, 256], 'clf__max_leaves': [3, 5], 'clf__interactions': [0, 5, 10]},
    'CB': {'clf__depth': [4, 6, 8], 'clf__learning_rate': [0.03, 0.1], 'clf__l2_leaf_reg': [3, 10], 'clf__iterations': [300, 600]}
}

mode = sys.argv[1]
m_name = sys.argv[2]
if mode == 'cv':
    rep_i = int(sys.argv[3])
    seed = int(sys.argv[4])
    print(f"[{m_name} CV Rep {rep_i}] Starting...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred_oof = np.zeros(len(y_train), dtype=object)
    y_prob_oof = np.zeros((len(y_train), 3))
    
    for train_idx, test_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[test_idx], y_train.iloc[test_idx]
        
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        grid = GridSearchCV(models_dict[m_name], param_grids[m_name], cv=inner_cv, scoring='f1_macro', n_jobs=1)
        if m_name == 'CB': grid.fit(X_tr, y_tr, clf__cat_features=cat_features)
        else: grid.fit(X_tr, y_tr)
        
        best_mod = grid.best_estimator_
        y_pred_oof[test_idx] = np.array(best_mod.predict(X_val)).flatten()
        y_prob_oof[test_idx] = best_mod.predict_proba(X_val)
        
    met = get_metrics(y_train, y_pred_oof, y_prob_oof)
    met['Model'] = m_name
    met['Repetition'] = rep_i + 1
    with open(f"tmp_cv_{m_name}_{rep_i}.json", 'w') as f: json.dump(met, f)
    print(f"[{m_name} CV Rep {rep_i}] Done.")

elif mode == 'final':
    print(f"[{m_name} Final] Starting...")
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    grid = GridSearchCV(models_dict[m_name], param_grids[m_name], cv=inner_cv, scoring='f1_macro', n_jobs=1)
    if m_name == 'CB': grid.fit(X_train, y_train, clf__cat_features=cat_features)
    else: grid.fit(X_train, y_train)
    
    prm = grid.best_params_
    prm['Model'] = m_name
    with open(f"tmp_param_{m_name}.json", 'w') as f: json.dump(prm, f)
    
    mod = grid.best_estimator_
    y_pred_ext = np.array(mod.predict(X_test)).flatten()
    y_prob_ext = mod.predict_proba(X_test)
    met = get_metrics(y_test, y_pred_ext, y_prob_ext)
    met['Model'] = m_name
    with open(f"tmp_ext_{m_name}.json", 'w') as f: json.dump(met, f)
    
    cm = confusion_matrix(y_test, y_pred_ext, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    cm_df['Model'] = m_name
    cm_df.to_csv(f"ML16_CONFUSION_{m_name}.csv")
    print(f"[{m_name} Final] Done.")
