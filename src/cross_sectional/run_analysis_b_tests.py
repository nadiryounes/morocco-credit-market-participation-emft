import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("B_ANALYSIS_DATA.csv")

tab = pd.crosstab(df['STATE_19'], df['STATE_23'])
states = ['CREDIT_APPLICANT', 'CONSTRAINED_NON_APPLICANT', 'NO_FINANCING_NEED']
tab = tab.reindex(index=states, columns=states)

tab.to_csv("B2_TRANSITION_MATRIX_COUNTS.csv")

row_pct = tab.div(tab.sum(axis=1), axis=0) * 100
row_pct.to_csv("B3_TRANSITION_MATRIX_ROW_PCT.csv")

col_pct = tab.div(tab.sum(axis=0), axis=1) * 100
col_pct.to_csv("B4_TRANSITION_MATRIX_COLUMN_PCT.csv")

print(tab)

# Stuart-Maxwell Test (using statsmodels' marginal_homogeneity)
st = sm.stats.SquareTable(tab.values)
sm_test = st.homogeneity()
print("Stuart-Maxwell p-value:", sm_test.pvalue)

sm_res = pd.DataFrame([{
    'Test': 'Stuart-Maxwell',
    'Statistic': sm_test.statistic,
    'DF': sm_test.df,
    'P_Value': sm_test.pvalue,
    'Implementation': 'statsmodels.stats.contingency_tables.SquareTable'
}])
sm_res.to_csv("B5_STUART_MAXWELL_TEST.csv", index=False)

# Strict discouraged descriptive
df_full = pd.read_stata("Morocco-2013-2019-2023.dta", convert_categoricals=False)
disc = df_full[df_full['k17'] == 6]
disc_res = pd.DataFrame([{
    'STRICT_DISCOURAGED_COUNT': len(disc),
    'Total_Observations': len(df_full),
    'Percentage': len(disc) / len(df_full) * 100,
    'Weighting': 'Unweighted'
}])
disc_res.to_csv("B13_STRICT_DISCOURAGED_DESCRIPTIVE.csv", index=False)

# Panel Inclusion Diagnostic (comparing 176 panel firms vs 2019 firms NOT in panel)
# Actually, the user asked to compare firms IN the matched panel (280) vs NOT in panel.
# "compare available 2019 baseline characteristics between: A. firms observed in the 2019-2023 matched panel B. other 2019 firms"
years = df_full.groupby("panelid")["year"].apply(set)
matched_pids = [pid for pid, y in years.items() if 2019 in y and 2023 in y]

df19_full = df_full[df_full['year'] == 2019].copy()
df19_full['IN_PANEL'] = df19_full['panelid'].isin(matched_pids).astype(int)

# Code variables for comparison
df19_full['EMP'] = df19_full['l1'].replace(-9, np.nan)
df19_full['AGE'] = 2019 - df19_full['b5'].replace(-9, np.nan)
df19_full['EXPORTER'] = (df19_full['d3c'] > 0).astype(int).where(df19_full['d3c'].notna(), np.nan)

def get_state(r):
    k16 = r['_2013_2019_k16']
    k17 = r['k17']
    if k16 == 1: return 'CREDIT_APPLICANT'
    elif k16 == 2 and k17 == 1: return 'NO_FINANCING_NEED'
    elif k16 == 2 and k17 in [2, 3, 4, 5, 6]: return 'CONSTRAINED_NON_APPLICANT'
    return 'AMBIGUOUS'

df19_full['CREDIT_STATE'] = df19_full.apply(get_state, axis=1)

diag = []
for var, vtype in [('EMP', 'cont'), ('AGE', 'cont'), ('EXPORTER', 'prop')]:
    g1 = df19_full[df19_full['IN_PANEL'] == 1][var].dropna()
    g0 = df19_full[df19_full['IN_PANEL'] == 0][var].dropna()
    mean1, mean0 = g1.mean(), g0.mean()
    sd1, sd0 = g1.std(), g0.std()
    # Cohen's d (SMD)
    pooled_sd = np.sqrt(((len(g1)-1)*sd1**2 + (len(g0)-1)*sd0**2) / (len(g1)+len(g0)-2))
    if vtype == 'prop':
        smd = abs(mean1 - mean0) / np.sqrt((mean1*(1-mean1) + mean0*(1-mean0))/2)
    else:
        smd = abs(mean1 - mean0) / pooled_sd
    
    diag.append({
        'Variable': var,
        'Mean_Panel': mean1,
        'Mean_NotPanel': mean0,
        'N_Panel': len(g1),
        'N_NotPanel': len(g0),
        'SMD': smd
    })

pd.DataFrame(diag).to_csv("B11_PANEL_INCLUSION_DIAGNOSTIC.csv", index=False)
print("Python tests done.")
