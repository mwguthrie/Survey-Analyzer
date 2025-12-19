import pandas as pd
from fuzzywuzzy import process

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: put your CSV paths and test type here
PRE_PATH   = 'BEMA Pre - 1502-006 - Fa25_December 15, 2025_09.45.csv'
POST_PATH  = 'BEMA Post - 1502-006 - Fa25_December 15, 2025_09.37.csv'
TEST_TYPE  = 'BEMA'   # "EMCS' or 'BEMA' or 'EBAPS'
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ANSWER KEYS

# EMCS: 25 scored questions + attention Q34, ID Q35
EMCS_KEY = {
    **{f'Q{i}': str(ans) for i, ans in zip(range(1,26),
        [2,5,2,1,4,3,5,3,1,4,5,4,3,4,1,3,2,5,2,1,3,4,2,1,5])}
}
EMCS_ATT_Q, EMCS_ATT_A = 'Q34', '5'
EMCS_ID = 'Q35'

# BEMA: individual keys plus the special rules
# first the straightforward single‐Q answers (numeric: A=1, B=2, ..., G=7):
# Note: some exports duplicate the Q19 header; pandas renames the second to "Q19.1".
# We handle Q13/Q19 explicitly in the grader to be robust to that duplication.
BEMA_SINGLE = {
    **{f'Q{i}': str(ans) for i, ans in zip(
        [4,5,6,7,8,9,10,11,12,14,15,17,18,20,21,22,23,24,25,26,27,30,31],
        [5,1,4,5,2,2, 6, 5, 5, 2, 7, 4, 2, 7, 1, 5, 5, 1, 4, 4, 3, 6, 4]
    )}
}
# Q3 rule mapping Q2→correct Q3
BEMA_Q3_MAP = {str(k):str(v) for k,v in zip([1,2,3,4,5,6],[2,4,3,5,6,7])}
# Q16 rule: same as Q14 AND Q15==7
# Q28+29 combined must be (2,3)
BEMA_ATT_Q, BEMA_ATT_A = 'Q53','5'
BEMA_ID = 'Q54'

# EBAPS: map raw choice ('1'–'5') → score for each question Q1–Q30
# (A=1, B=2, C=3, D=4, E=5 in the exported CSV)
EBAPS_SCORE = {
    'Q1':  {'1':4,'2':3,'3':1,'4':0.5,'5':0},
    'Q2':  {'1':0,'2':1.5,'3':2.5,'4':3.5,'5':4},
    'Q3':  {'1':0,'2':1,'3':2,'4':3.5,'5':4},
    'Q4':  {'1':4,'2':3,'3':2,'4':1,'5':0},
    'Q5':  {'1':0,'2':1,'3':2,'4':3,'5':4},
    'Q6':  {'1':4,'2':4,'3':2,'4':1,'5':0},
    'Q7':  {'1':4,'2':3,'3':2,'4':1,'5':0},
    'Q8':  {'1':4,'2':3,'3':1.5,'4':0.5,'5':0},
    'Q9':  {'1':0,'2':1,'3':2,'4':3,'5':4},
    'Q10': {'1':4,'2':3,'3':2,'4':1,'5':0},
    'Q11': {'1':0,'2':1,'3':2,'4':3,'5':4},
    'Q12': {'1':0,'2':0.5,'3':1,'4':3,'5':4},
    'Q13': {'1':4,'2':3,'3':1,'4':0.5,'5':0},
    'Q14': {'1':4,'2':3,'3':2,'4':1,'5':0},
    'Q15': {'1':4,'2':3,'3':2,'4':1,'5':0},
    'Q16': {'1':0,'2':1,'3':2,'4':3,'5':4},
    'Q17': {'1':4,'2':3,'3':1.5,'4':0.5,'5':0},
    'Q18': {'1':4,'2':3.5,'3':1.5,'4':0.5,'5':0},
    'Q19': {'1':4,'2':0,'3':3,'4':2,'5':1},
    'Q20': {'1':4,'2':0,'3':3,'4':2,'5':1},
    'Q21': {'1':4,'2':3,'3':2,'4':1,'5':0},
    'Q22': {'1':4,'2':3,'3':2,'4':1,'5':0},
    'Q23': {'1':0,'2':4,'3':1,'4':2,'5':3},
    'Q24': {'1':4,'2':4,'3':2,'4':1,'5':0},
    'Q25': {'1':0,'2':1,'3':2,'4':4,'5':4},
    'Q26': {'1':4,'2':4,'3':2,'4':1,'5':0},
    'Q27': {'1':4,'2':4,'3':2,'4':1,'5':0},
    'Q28': {'1':0,'2':1,'3':2,'4':3,'5':4},
    'Q29': {'1':0,'2':2,'3':4,'4':2,'5':0},
    'Q30': {'1':0,'2':1,'3':2,'4':3,'5':4},
}

# maximum possible scores
MAX_EMCS   = len(EMCS_KEY)
MAX_BEMA   = 30  # BEMA totals 30 points (Q28&Q29 count as one question)
MAX_EBAPS  = 4 * len(EBAPS_SCORE)   # each question max = 4

# ─────────────────────────────────────────────────────────────────────────────
# GRADING FUNCTIONS

def grade_emcs(df):
    df = df[df[EMCS_ATT_Q] == EMCS_ATT_A].copy()
    def sc(r):
        return sum(r[q] == a for q, a in EMCS_KEY.items())
    out = df[[EMCS_ID]].copy()
    out['pre_post_score'] = df.apply(sc, axis=1)
    return out

def grade_bema(df):
    df = df[df[BEMA_ATT_Q] == BEMA_ATT_A].copy()
    def sc(r):
        s = 0
        # Q1,2
        if r['Q1']=='1': s+=1
        if r['Q2']=='1': s+=1
        # Q3 rule
        if r['Q2'] in BEMA_Q3_MAP and r['Q3']==BEMA_Q3_MAP[r['Q2']]:
            s+=1
        # Q13 and Q19 (some exports duplicate Q19; pandas renames the second to Q19.1)
        # Official key: Q13=D(4), Q19=B(2).
        q13_val = r.get('Q13')
        if q13_val is None and ('Q19.1' in r.index) and ('Q19' in r.index):
            # If there are duplicate Q19 headers, the first "Q19" sometimes corresponds to Q13.
            q13_val = r.get('Q19')
        if q13_val == '4':
            s += 1

        q19_val = r.get('Q19.1') if ('Q19.1' in r.index) else r.get('Q19')
        if q19_val == '2':
            s += 1

        # single‐Q keys
        for q, a in BEMA_SINGLE.items():
            if r.get(q) == a:
                s += 1
        # Q16 rule
        if r['Q16'] == r['Q14'] and r['Q15']=='7':
            s+=1
        # Q28+29 combined
        if r['Q28']=='2' and r['Q29']=='3':
            s+=1
        return s

    out = df[[BEMA_ID]].copy()
    out['pre_post_score'] = df.apply(sc, axis=1)
    return out

def grade_ebaps(df):
    def sc(r):
        return sum(
            EBAPS_SCORE[q].get(str(r[q]), 0)
            for q in EBAPS_SCORE
        )
    out = pd.DataFrame({ 'idx': df.index })
    out['pre_post_score'] = df.apply(sc, axis=1)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# MATCHING & STATISTICS

def fuzzy_match_ids(pre_ids, post_ids, threshold=70):
    """Return dict pre_id → post_id for those matching above threshold."""
    
    def _valid(x):
        return pd.notna(x) and str(x).strip() != ''
    
    
    # Filter out any empty or whitespace-only IDs
    valid_pre  = [str(x) for x in pre_ids  if _valid(x)]
    valid_post = [str(x) for x in post_ids if _valid(x)]
    
    
    matches = {}
    used_post = set()
    for pre in valid_pre:
        best, score = process.extractOne(pre, valid_post)
        if score >= threshold and best not in used_post:
            matches[pre] = best
            used_post.add(best)
    return matches

def compute_summary(df_pairs, max_score, all_pre, all_post):
    """df_pairs has columns: pre_score, post_score"""
    df = df_pairs.copy()
    df['gain_norm'] = (
        (df['post_score'] - df['pre_score']) /
        (max_score - df['pre_score'])
    )
    mean_pre_total  = all_pre.mean()
    mean_post_total = all_post.mean()
    class_norm_gain = (
            (mean_post_total - mean_pre_total)
            / (max_score - mean_pre_total)
        )
    
    summary = {
        'n_matched':      len(df),
        'mean_pre':       df['pre_score'].mean(),
        'mean_post':      df['post_score'].mean(),
        'mean_norm_gain': df['gain_norm'].mean(),

        'n_pre_total':   len(all_pre),
        'n_post_total':  len(all_post),
        'mean_pre_total':  all_pre.mean(),
        'mean_post_total': all_post.mean(),
        'class_norm_gain':  class_norm_gain,
        }
        
        
    return df, summary

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS

def analyze(pre_path, post_path, test_type):
    pre_df  = pd.read_csv(pre_path)
    post_df = pd.read_csv(post_path)
    
    # print(list(pre_df))
    # print(list(post_df))
    
    
    #Filter out short responses
    for name, df in (('pre', pre_df), ('post', post_df)):
        # convert to numeric, coercing errors → NaN
        df['Duration (in seconds)'] = pd.to_numeric(
            df['Duration (in seconds)'],
            errors='coerce'
        )
        # drop NaNs
        df.dropna(subset=['Duration (in seconds)'], inplace=True)
        # keep only ≥60 s
        filtered = df.loc[df['Duration (in seconds)'] >= 60]
        if name == 'pre':
            pre_df = filtered.copy()
        else:
            post_df = filtered.copy()

    if test_type == 'EMCS':
        pre_sc  = grade_emcs(pre_df)
        post_sc = grade_emcs(post_df)
        id_col  = EMCS_ID
        max_sc  = MAX_EMCS

    elif test_type == 'BEMA':
        pre_sc  = grade_bema(pre_df)
        post_sc = grade_bema(post_df)
        id_col  = BEMA_ID
        max_sc  = MAX_BEMA

    elif test_type == 'EBAPS':
        pre_sc  = grade_ebaps(pre_df)
        post_sc = grade_ebaps(post_df)
        id_col  = 'idx'          # use row‐index
        max_sc  = MAX_EBAPS

    else:
        raise ValueError(f"Unknown TEST_TYPE {test_type}")

    # match IDs (for EBAPS these are just 0..n-1)
    pre_ids  = pre_sc[id_col].astype(str).tolist()
    post_ids = post_sc[id_col].astype(str).tolist()
    matches  = fuzzy_match_ids(pre_ids, post_ids)
    
    matches = {
    k: v for k, v in matches.items()
    if pd.notna(k) and pd.notna(v)
       and str(k).strip().lower() != 'nan'
       and str(v).strip().lower() != 'nan'
       }
    
    

    # assemble matched‐pair DataFrame
    rows = []
    for pre_id, post_id in matches.items():
        pre_score  = pre_sc.loc[ pre_sc[id_col].astype(str)==pre_id, 
                                'pre_post_score'].iloc[0]
        post_score = post_sc.loc[post_sc[id_col].astype(str)==post_id, 
                                 'pre_post_score'].iloc[0]
        rows.append({'pre_id':pre_id,'post_id':post_id,
                     'pre_score':pre_score,'post_score':post_score})
    df_pairs = pd.DataFrame(rows)
    
    
    # df_pairs = df_pairs.dropna()
    
    all_pre_scores = pre_sc['pre_post_score']
    all_post_scores = post_sc['pre_post_score']

    # compute statistics
    df_results, summary = compute_summary(df_pairs, 
                                          max_sc, 
                                          all_pre_scores,
                                          all_post_scores)
    
    
    
    return df_results, summary

if __name__ == '__main__':
    df_results, summary = analyze(PRE_PATH, POST_PATH, TEST_TYPE)
    print(f"\n--- {TEST_TYPE} RESULTS ---")
    print(f"Matched students: {summary['n_matched']}")
    print(f"Mean matched pre-test score : {summary['mean_pre']:.2f} / {MAX_EMCS if TEST_TYPE=='EMCS' else MAX_BEMA if TEST_TYPE=='BEMA' else MAX_EBAPS}")
    print(f"Mean matched post-test score: {summary['mean_post']:.2f}")
    print(f"Mean matched normalized gain: {summary['mean_norm_gain']:.3f}\n")
    
    # Individual data
    print("\nIndividual gains:")
    print(df_results[['pre_id','post_id','pre_score','post_score','gain_norm']])
    
    # Class-wide stats
    print(f"\nClass-wide pre-test mean :       {summary['mean_pre_total']:.2f}  (n={summary['n_pre_total']})")
    print(f"Class-wide post-test mean:       {summary['mean_post_total']:.2f}  (n={summary['n_post_total']})")
    print(f"Class-wide normalized gain:      {summary['class_norm_gain']:.3f}")
    

