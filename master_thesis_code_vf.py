
# ---------------------------
# 1. Setup
# ---------------------------

# Import of packages 
#%%
# ================================
# TANDARD LIBRARIES
# ================================
import os
import sys
import time
import json
import glob
import shutil
import tempfile
import traceback
from datetime import datetime
from itertools import product
from collections import Counter, defaultdict
from random import sample
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import UserWarning, FutureWarning, ConvergenceWarning, FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

# ================================
# DATA HANDLING
# ================================
import numpy as np
import pandas as pd
import joblib

# ================================
# VISUALIZATION
# ================================
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

# ================================
#  PARALLELISM & UTILITIES
# ================================
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

# ================================
# DATA PREPROCESSING
# ================================
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ================================
# STATISTICAL TESTING
# ================================
import scipy.stats as stats
from scipy.stats import (
    spearmanr, ks_2samp, kruskal, shapiro, mannwhitneyu
)
import scikit_posthocs as sp

# ================================
# MACHINE LEARNING MODELS
# ================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

# ================================
# 📈 MODEL SELECTION
# ================================
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    HalvingRandomSearchCV,
    ParameterSampler
)
from sklearn.experimental import enable_halving_search_cv  # required before HalvingRandomSearchCV
from sklearn.base import clone

# ================================
# EVALUATION METRICS
# ================================
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, auc
)

# ================================
# CLASS IMBALANCE HANDLING
# ================================
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

# ================================
#  INTERPRETABILITY & ANALYSIS
# ================================
import shap
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import hdbscan

#%%
# Set Pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#%%
# ---------------------------
# 2. Load Raw Data 
# ---------------------------
# Base directories
base_dir = r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data"
raw_dir = os.path.join(base_dir, "RawData")
prep_dir = os.path.join(base_dir, "PrepData")
results_dir = os.path.join(base_dir, "ResultsData")
models_dir = r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Models"

# File paths
file_paths = {
    "events": os.path.join(raw_dir, "events_England.json"),
    "tags": os.path.join(raw_dir, "tags2name.csv"),
    "events_name": os.path.join(raw_dir, "eventid2name.csv"),
    "matches": os.path.join(raw_dir, "matches_England.json"),
    "players": os.path.join(raw_dir, "players.json"),
    "teams": os.path.join(raw_dir, "teams.json"),
    "salary": os.path.join(prep_dir, "salary_pl_2017_2018_wyscoutnames.csv")
}

# Unified data loading function
def load_data(paths):
    with open(paths["events"], 'r', encoding='utf-8') as f:
        events = json.load(f)
    tags_df = pd.read_csv(paths["tags"])
    events_df = pd.read_csv(paths["events_name"])
    matches_df = pd.read_json(paths["matches"])
    players_df = pd.read_json(paths["players"])
    teams_df = pd.read_json(paths["teams"])
    salary_df = pd.read_csv(paths["salary"])
    
    return events, tags_df, events_df, matches_df, players_df, teams_df, salary_df

# Load all datasets
events_raw, tags_df, events_names_df, matches_df, players_df, teams_df, df_salary = load_data(file_paths)
# %%

players_df['role_code'] = players_df['role'].apply(lambda x: x.get('code2') if isinstance(x, dict) else np.nan)
players_df['role_name'] = players_df['role'].apply(lambda x: x.get('name') if isinstance(x, dict) else np.nan)
players_df.drop(columns=['role'], inplace=True)
players_df['firstName'] = players_df['firstName'].apply(lambda x: x.encode('utf-8').decode('unicode_escape') if isinstance(x, str) else x)
players_df['lastName'] = players_df['lastName'].apply(lambda x: x.encode('utf-8').decode('unicode_escape') if isinstance(x, str) else x)
players_df['fullName'] =  players_df['firstName'].str.strip() + ' ' + players_df['lastName'].str.strip()
# %%
df_events = pd.DataFrame(events_raw)
data = df_events.copy()
data['subEventId'] = pd.to_numeric(data['subEventId'], errors='coerce')
data['subEventId'].fillna(60, inplace=True)
data['tags'] = data['tags'].apply(lambda x: [{'id': int(d['id'])} for d in x])


#%%
# ---------------------------
# 3. Feature enginerring functions 
# ---------------------------
def add_tag_columns(df):
    df = df.copy()

    # Clean malformed tag entries
    def clean_tags_entry(x):
        if isinstance(x, list):
            flat = [d for d in x if isinstance(d, dict) and 'id' in d]
            return [{'id': int(d['id'])} for d in flat]
        return []

    df['tags'] = df['tags'].apply(clean_tags_entry)

    # Extract tag IDs
    tag_lists = df['tags'].apply(lambda taglist: [tag['id'] for tag in taglist])

    # Binary encoding
    mlb = MultiLabelBinarizer()
    tag_array = mlb.fit_transform(tag_lists)
    tag_df = pd.DataFrame(tag_array, columns=[f"tag_{i}" for i in mlb.classes_], index=df.index)

    # Manual override for tag_1501 (clearance)
    tag_df['tag_1501'] = ((df['eventId'] == 7) & (df['subEventId'] == 71)).astype(int)

    # Combine
    df = pd.concat([df, tag_df], axis=1)
    return df
data = add_tag_columns(data)

# %%
def extract_coordinates(pos, index, key, fallback=None):
    """
    Extract coordinate from Wyscout 'positions' field.
    
    - pos: list of dicts with 'x'/'y' keys.
    - index: 0 for origin, 1 for destination.
    - key: 'x' or 'y'
    - fallback: value to use if coordinate is missing or ambiguous.
    
    Returns coordinate or fallback (or np.nan if fallback is None).
    """
    try:
        value = pos[index][key]
        return value
    except (IndexError, KeyError, TypeError):
        return fallback if fallback is not None else np.nan

# %%
# Origin: always assumed valid
data['coordinates_x'] = data['positions'].apply(lambda pos: extract_coordinates(pos, 0, 'x'))
data['coordinates_y'] = data['positions'].apply(lambda pos: extract_coordinates(pos, 0, 'y'))

# Destination: fallback to origin to avoid NaNs
data['end_coordinates_x'] = data.apply(
    lambda row: extract_coordinates(row['positions'], 1, 'x', fallback=row['coordinates_x']), axis=1
)

data['end_coordinates_y'] = data.apply(
    lambda row: extract_coordinates(row['positions'], 1, 'y', fallback=row['coordinates_y']), axis=1
)

# %%
def calculate_distance_to_goal(pos):
    """Compute Euclidean distance from event origin to goal center (100,50)."""
    goal_x, goal_y = 100, 50
    if len(pos) > 0:
        x, y = pos[0]['x'], pos[0]['y']
        return np.hypot(goal_x - x, goal_y - y)
    return np.nan


# %%
def calculate_angle_to_goal(pos, event_id, subevent_id):
    """
    Calculate angle to the goal mouth based on vectors to left and right posts.
    Only for shots or FK shots.
    """
    if not ((event_id == 10) or (event_id == 3 and subevent_id in [33, 35])):
        return np.nan  # better than -1 for ML

    goal_post_left = np.array([100, 36])
    goal_post_right = np.array([100, 64])

    if len(pos) > 0:
        x, y = pos[0]['x'], pos[0]['y']
        ball_pos = np.array([x, y])
        vec_left = goal_post_left - ball_pos
        vec_right = goal_post_right - ball_pos

        norm_left = np.linalg.norm(vec_left)
        norm_right = np.linalg.norm(vec_right)

        if norm_left == 0 or norm_right == 0:
            return np.nan

        cos_angle = np.clip(np.dot(vec_left, vec_right) / (norm_left * norm_right), -1.0, 1.0)
        return np.arccos(cos_angle)  # in radians
    return np.nan

# %%
distance_imputer = SimpleImputer(strategy='median')
angle_imputer = SimpleImputer(strategy='mean')
data['distance_to_goal'] = data['positions'].apply(calculate_distance_to_goal)
data['angle_to_goal'] = data.apply(lambda row: calculate_angle_to_goal(row['positions'], row['eventId'], row['subEventId']), axis=1)
data['distance_to_goal'] = distance_imputer.fit_transform(data[['distance_to_goal']])
data['angle_to_goal'] = angle_imputer.fit_transform(data[['angle_to_goal']])

# %%
##Only real angles
##Set the imputed mean angle 
imputed_angle = 1.2527171992227035

## Remove the imputed values from bin calculation
real_angles = data.loc[data['angle_to_goal'] != imputed_angle, 'angle_to_goal']

##Now get proper quantiles 
quantile_bins = real_angles.quantile([0, 0.25, 0.5, 0.75, 1.0]).values

## Assign non-shot first
data['angle_category_quantile'] = np.where(data['angle_to_goal'] < 0, 'non-shot', np.nan)

## Apply cut with safety for duplicates
data.loc[data['angle_to_goal'] >= 0, 'angle_category_quantile'] = pd.cut(
    data.loc[data['angle_to_goal'] >= 0, 'angle_to_goal'],
    bins=quantile_bins,
    labels=['Q1', 'Q2', 'Q3', 'Q4'],
    include_lowest=True,
    duplicates='drop'
)
# %%
data.loc[data['angle_to_goal'] >= 0, 'angle_category_quantile'] = pd.cut(
    data.loc[data['angle_to_goal'] >= 0, 'angle_to_goal'],
    bins=quantile_bins,
    labels=['Q1', 'Q2', 'Q3', 'Q4'],
    include_lowest=True,
    duplicates='drop'
)

# %%
def add_contextual_features(df_events, players_df):
    df = df_events.copy()

    # --- Merge player roles ---
    role_info = players_df[['wyId', 'role_code']].rename(columns={'wyId': 'playerId'})
    df = df.merge(role_info, on='playerId', how='left')
    df['role_code'] = df['role_code'].fillna('UNKNOWN')

    # --- Temporal Features ---
    df['time_since_last_action'] = df.groupby('matchId')['eventSec'].diff().fillna(0)
    df['prev_teamId'] = df.groupby('matchId')['teamId'].shift(1)
    df['possession_change'] = (df['teamId'] != df['prev_teamId']).astype(int)

    # Proper possession change time handling
    df['poss_change_time'] = df['eventSec'].where(df['possession_change'] == 1)
    df['last_poss_change_time'] = df.groupby('matchId')['poss_change_time'].ffill()
    df['time_since_possession_change'] = (
        df['eventSec'] - df['last_poss_change_time']
    ).clip(lower=0).fillna(0)

    # --- Shot Features ---
    df['is_shot'] = (
        (df['eventId'] == 10) |
        ((df['eventId'] == 3) & (df['subEventId'].isin([33, 35])))
    ).astype(int)
    df['shot_time'] = df['eventSec'].where(df['is_shot'] == 1)

    # FIX: Forward-fill last shot time per team, not per match
    df['last_shot_time'] = df.groupby(['matchId', 'teamId'])['shot_time'].ffill()
    df['time_since_last_shot'] = (
        df['eventSec'] - df['last_shot_time']
    ).clip(lower=0).fillna(0)

    # --- Movement Features ---
    df['prev_x'] = df.groupby('matchId')['coordinates_x'].shift(1)
    df['prev_y'] = df.groupby('matchId')['coordinates_y'].shift(1)
    df['distance_covered'] = np.sqrt(
        (df['coordinates_x'] - df['prev_x'])**2 +
        (df['coordinates_y'] - df['prev_y'])**2
    ).fillna(0)

    # --- Team Spatial Averages (rolling mean per team) ---
    df['avg_team_x'] = (
        df.groupby(['matchId', 'teamId'])['coordinates_x']
        .rolling(window=5, min_periods=1).mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df['avg_team_y'] = (
        df.groupby(['matchId', 'teamId'])['coordinates_y']
        .rolling(window=5, min_periods=1).mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # --- Sequence Statistics ---
    df['is_pass'] = (df['eventId'] == 8).astype(int)
    df['num_passes_last_5'] = (
        df.groupby('matchId')['is_pass']
        .rolling(window=5, min_periods=1).sum()
        .reset_index(level=0, drop=True)
    )

    # --- Dribble Feature ---
    df['is_dribble'] = (
        (df['eventId'] == 1) &
        (df['subEventId'] == 11) &
        (df.get('tag_703', 0) == 1)
    ).astype(int)
    df['num_dribbles_last_5'] = (
        df.groupby('matchId')['is_dribble']
        .rolling(window=5, min_periods=1).sum()
        .reset_index(level=0, drop=True)
    )

    # --- Contextual Awareness ---
    df['same_team_as_prev'] = (df['teamId'] == df['prev_teamId']).astype(int)
    df['prev_eventId'] = df.groupby('matchId')['eventId'].shift(1).fillna(-1).astype(int)
    df['prev_subEventId'] = df.groupby('matchId')['subEventId'].shift(1).fillna(-1).astype(int)
    df['is_in_final_third'] = (df['coordinates_x'] > 66).astype(int)
    df['is_central_channel'] = ((df['coordinates_y'] >= 33) & (df['coordinates_y'] <= 66)).astype(int)

    # --- Defensive Indicator ---
    defensive_roles = ['GK', 'DF']
    df['is_defensive_player'] = df['role_code'].isin(defensive_roles).astype(int)

    # --- Cleanup ---
    df.drop(columns=['shot_time', 'poss_change_time', 'last_poss_change_time'], inplace=True)

    return df


# %%
data= add_contextual_features(data, players_df)

# %%
def scoring(df_events):
    df = df_events.copy()

    # Filtrer les périodes valides
    df = df[df['matchPeriod'].isin(['1H', '2H'])].copy()

    # Temps absolu
    period_order = {'1H': 1, '2H': 2}
    period_offset = {'1H': 0, '2H': 45 * 60}
    df['period_order'] = df['matchPeriod'].map(period_order)
    df['absoluteSec'] = df.apply(
        lambda row: period_offset[row['matchPeriod']] + row['eventSec'], axis=1
    )

    # Initialisation
    df['team_score_before_event'] = 0
    df['opponent_score_before_event'] = 0
    df['score_difference_before_event'] = 0
    df['team_game_state'] = 'drawing'
    df['is_team_losing_before_event'] = False
    df['score_diff_from_focal_team'] = 0  # reste pour compatibilité downstream

    all_rows = []

    for match_id in df['matchId'].unique():
        match_df = df[df['matchId'] == match_id].sort_values(['period_order', 'eventSec', 'eventId']).copy()
        team_ids = match_df['teamId'].unique()
        if len(team_ids) != 2:
            continue

        score_by_team = {team_ids[0]: 0, team_ids[1]: 0}

        team_score_list = []
        opp_score_list = []
        score_diff_list = []
        game_state_list = []
        is_losing_list = []
        diff_from_focal_list = []

        for _, row in match_df.iterrows():
            team = row['teamId']
            opponent = [t for t in team_ids if t != team][0]

            team_score = score_by_team[team]
            opponent_score = score_by_team[opponent]
            diff = team_score - opponent_score

            # Game state from perspective of team acting
            if diff > 0:
                state = 'winning'
            elif diff < 0:
                state = 'losing'
            else:
                state = 'drawing'

            team_score_list.append(team_score)
            opp_score_list.append(opponent_score)
            score_diff_list.append(diff)
            game_state_list.append(state)
            is_losing_list.append(diff < 0)

            # Pour compatibilité : perspective focal_team (1er team du match)
            focal_team = team_ids[0]
            if team == focal_team:
                diff_from_focal = diff
            else:
                diff_from_focal = -diff
            diff_from_focal_list.append(diff_from_focal)

            # Mise à jour du score
            is_goal = row.get('tag_101', 0) == 1
            is_shot = (
                row['eventId'] == 10 or
                (row['eventId'] == 3 and row['subEventId'] in [33, 35])
            )
            if is_goal and is_shot:
                score_by_team[team] += 1
            elif row.get('tag_102', 0) == 1:
                score_by_team[opponent] += 1

        match_df['team_score_before_event'] = team_score_list
        match_df['opponent_score_before_event'] = opp_score_list
        match_df['score_difference_before_event'] = score_diff_list
        match_df['team_game_state'] = game_state_list
        match_df['is_team_losing_before_event'] = is_losing_list
        match_df['score_diff_from_focal_team'] = diff_from_focal_list

        all_rows.append(match_df)

    df = pd.concat(all_rows).sort_index()

    # Clutch moment (après 75e min, score serré)
    df['is_clutch_moment'] = (
        (df['matchPeriod'] == '2H') &
        (df['eventSec'] >= 2700) &
        (df['score_diff_from_focal_team'].abs() <= 1)
    )

    df.drop(columns=['period_order'], inplace=True)
    return df



# %%
data = scoring(data)

#%%

salary_df_cleaned = df_salary[[
    'Player', 'Team',
    'Weekly GrossBase Salary(IN GBP)',
    'Annual GrossBase Salary(IN GBP)'
]].rename(columns={
    'Player': 'fullName',
    'Team': 'club',
    'Weekly GrossBase Salary(IN GBP)': 'weekly_wage_gbp',
    'Annual GrossBase Salary(IN GBP)': 'annual_wage_gbp'
})

#%%

def infer_missing_player_ids(df):
    """
    Infer missing playerId values (where playerId == 0) using neighboring events,
    based on event/subEvent and tag_ columns generated by `add_tag_columns()`.
    """
    df = df.copy()

    # Define conditions likely associated with missing playerId rows
    event_ids = {1, 2, 3, 7, 8, 9, 10}
    sub_event_names = {
        'Air duel', 'Ground loose ball duel', 'Free Kick', 'Ground defending duel',
        'Ground attacking duel', 'Throw in', 'Corner', 'Goal kick', 'Touch', 'Foul',
        'Simple pass', 'Launch', 'Hand foul', 'Free kick cross', 'Shot', 'Clearance', 'Reflexes'
    }
    event_names = {
        'Duel', 'Free Kick', 'Others on the ball', 'Foul', 'Pass', 'Shot', 'Save attempt'
    }

    # List of binary tag columns to consider as indicators of real game actions
    relevant_tag_cols = [col for col in df.columns if col.startswith('tag_') and df[col].sum() > 0]

    # Define condition mask
    def is_likely_missing(row):
        tag_hit = any(row.get(col, 0) == 1 for col in relevant_tag_cols)
        return (
            (row['playerId'] == 0) and (
                row['eventId'] in event_ids or
                row['subEventName'] in sub_event_names or
                row['eventName'] in event_names or
                tag_hit
            )
        )

    mask_missing = df.apply(is_likely_missing, axis=1)

    print(f"🕵️ Attempting inference for {mask_missing.sum()} missing playerId rows...")

    df = df.reset_index(drop=True)
    inferred_players = []

    for idx in df[mask_missing].index:
        match_id = df.at[idx, 'matchId']
        team_id = df.at[idx, 'teamId']

        player_backward = None
        for b in range(idx - 1, max(idx - 6, -1), -1):
            if df.at[b, 'matchId'] != match_id:
                break
            if df.at[b, 'teamId'] == team_id and df.at[b, 'playerId'] != 0:
                player_backward = df.at[b, 'playerId']
                break

        player_forward = None
        for f in range(idx + 1, min(idx + 6, len(df))):
            if df.at[f, 'matchId'] != match_id:
                break
            if df.at[f, 'teamId'] == team_id and df.at[f, 'playerId'] != 0:
                player_forward = df.at[f, 'playerId']
                break

        if player_forward and player_forward == player_backward:
            inferred_players.append(player_forward)
        elif player_forward:
            inferred_players.append(player_forward)
        elif player_backward:
            inferred_players.append(player_backward)
        else:
            inferred_players.append(0)

    df.loc[mask_missing, 'playerId'] = inferred_players
    print(f"✅ Filled {sum(p != 0 for p in inferred_players)} playerId values (out of {len(inferred_players)})")

    return df

# %%
data = infer_missing_player_ids(data)
# %%
def compute_is_quality_shot(df):
    """
    Vectorized version of is_quality_shot.
    Flags quality shots using spatial and tag-based logic.
    Adds a new column: 'is_quality_shot' (1 if quality, else 0)
    """

    df = df.copy()

    # 1. Identify valid shots
    is_shot = (df['eventId'] == 10)
    is_fk_shot = (df['eventId'] == 3) & (df['subEventId'].isin([33, 35]))
    is_valid_shot = is_shot | is_fk_shot

    # 2. Tag-based flags
    accurate_shot = df.get('tag_1801', 0) == 1
    big_chance = df.get('tag_201', 0) == 1
    key_pass = df.get('tag_302', 0) == 1
    dribble_before = df.get('tag_703', 0) == 1

    high_quality_tag = accurate_shot | big_chance | key_pass | dribble_before

    low_defensive_pressure = ~(
        (df.get('tag_701', 0) == 1) |
        (df.get('tag_1401', 0) == 1) |
        (df.get('tag_1501', 0) == 1)
    )

    # 3. Spatial filters
    inside_penalty_area = df['distance_to_goal'] <= 20
    good_angle = df['angle_to_goal'] >= 0.6
    central_shot = (
        (df['coordinates_x'] >= 80) &
        (df['coordinates_y'] >= 30) & (df['coordinates_y'] <= 70)
    )

    spatially_good = inside_penalty_area | good_angle | central_shot

    # 4. Combine conditions
    quality_from_shot = (
        is_shot &
        (spatially_good & (high_quality_tag | low_defensive_pressure))
    )

    quality_from_fk = (
        (df['eventId'] == 3) &
        (
            (df['subEventId'] == 35) |  # Penalty = always quality
            ((df['subEventId'] == 33) & (df['distance_to_goal'] <= 30))
        )
    )

    df['is_quality_shot'] = ((quality_from_shot | quality_from_fk).astype(int))

    return df

#%%

#data.to_pickle(os.path.join(prep_dir, "data_process.pkl"))
data = pd.read_pickle(os.path.join(prep_dir, "data_process.pkl"))

# %%
data2 = data.copy()
#%%
data2 = compute_is_quality_shot(data2)
# %%
def compute_is_key_action(df):
    """
    Stricter version: Flags only high-signal offensive and defensive actions,
    avoiding over-flagging from common tags like 'accurate'.
    """
    df = df.copy()

    # High-value tags
    assist = df.get('tag_301', 0) == 1
    key_pass = df.get('tag_302', 0) == 1
    duel_win = df.get('tag_703', 0) == 1
    counter_attack = df.get('tag_1901', 0) == 1
    retained = df.get('tag_702', 0) == 1
    interception_tag = df.get('tag_1401', 0) == 1
    clearance_tag = df.get('tag_1501', 0) == 1

    # Event types
    is_pass = df['eventId'] == 8
    is_duel = (df['eventId'] == 1) & (df['subEventId'].isin([10, 11, 12]))
    is_interception = (df['eventId'] == 7) & (df['subEventId'] == 72)
    is_clearance = (df['eventId'] == 1) & (df['subEventId'] == 1)

    # Location & direction
    df['delta_x'] = df['end_coordinates_x'] - df['coordinates_x']
    in_final_third = df['coordinates_x'] >= 70
    in_deep_final_third = df['coordinates_x'] >= 80
    progressive = df['delta_x'] > 5

    # Pass types
    smart_pass = df['subEventId'] == 83
    cross_or_through = df['subEventId'].isin([84, 85])

    # --- Pass conditions ---
    high_value_pass = is_pass & (
        assist |
        key_pass |
        smart_pass |
        (cross_or_through & in_deep_final_third & progressive)
    )

    # --- Defensive / Transition actions ---
    duel_condition = is_duel & (duel_win | (retained & in_final_third))
    interception_condition = is_interception & interception_tag
    clearance_condition = is_clearance & clearance_tag
    counter_condition = counter_attack & (df['eventId'] != 3)  # ignore FK with counter

    # --- Combine all ---
    df['is_key_action'] = (
        high_value_pass |
        duel_condition |
        interception_condition |
        clearance_condition |
        counter_condition
    ).astype(int)

    return df


#%%

data2 = compute_is_key_action(data2)

#%%

# Recheck proportion of passes marked as key
pass_key_rate =data2[data2['eventId'] == 8]['is_key_action'].mean()
print(f"Key action rate on passes: {pass_key_rate:.2%}")


# %%
def flag_contributing_actions_advanced(
    df, 
    window=3, 
    possession_threshold=2
):
    """
    Flags contributions to quality shots using a fixed number of same-team actions (window),
    and flags defensive disruptions using backward possession-aware logic.

    Parameters:
        df (pd.DataFrame): Wyscout event data with 'is_quality_shot' and 'is_key_action'.
        window (int): Max number of same-team actions to evaluate (forward).
        possession_threshold (int): Max number of opponent actions before breaking sequence.

    Returns:
        pd.DataFrame with:
            - contributes_to_quality (1/0)
            - steps_to_shot (int)
            - disrupts_opponent_quality (1/0)
    """
    df = df.copy()
    df['contributes_to_quality'] = 0
    df['steps_to_shot'] = np.nan
    df['disrupts_opponent_quality'] = 0

    df = df.sort_values(['matchId', 'eventSec']).reset_index(drop=True)

    for match_id in df['matchId'].unique():
        match_df = df[df['matchId'] == match_id]
        match_indices = match_df.index.tolist()
        team_ids = match_df['teamId'].unique()
        if len(team_ids) < 2:
            continue

        for idx in match_indices:
            current_team = df.at[idx, 'teamId']
            same_team_count = 0
            opponent_count = 0

            # ---------- Forward: Offensive Contribution ----------
            for step in range(1, len(df) - idx):
                next_idx = idx + step
                if next_idx >= len(df):
                    break
                if df.at[next_idx, 'matchId'] != match_id:
                    break

                next_team = df.at[next_idx, 'teamId']
                if next_team == current_team:
                    same_team_count += 1
                    if df.at[next_idx, 'is_quality_shot'] == 1:
                        df.at[idx, 'contributes_to_quality'] = 1
                        df.at[idx, 'steps_to_shot'] = step
                        break
                    if same_team_count >= window:
                        break
                else:
                    opponent_count += 1
                    if opponent_count >= possession_threshold:
                        break

            # Always mark the shot itself
            if df.at[idx, 'is_quality_shot'] == 1:
                df.at[idx, 'contributes_to_quality'] = 1
                df.at[idx, 'steps_to_shot'] = 0

            # ---------- Backward: Defensive Disruption ----------
            if df.at[idx, 'is_key_action'] == 1:
                opponent_team = [t for t in team_ids if t != current_team][0]
                opponent_sequence = 0
                for step in range(1, idx + 1):
                    prev_idx = idx - step
                    if prev_idx < 0 or df.at[prev_idx, 'matchId'] != match_id:
                        break
                    if df.at[prev_idx, 'teamId'] != opponent_team:
                        break
                    opponent_sequence += 1
                    if df.at[prev_idx, 'is_quality_shot'] == 1:
                        break
                if opponent_sequence >= possession_threshold:
                    df.at[idx, 'disrupts_opponent_quality'] = 1

    return df

# %%
#data2.to_pickle(os.path.join(prep_dir, "data_bfc.pkl"))
data2= pd.read_pickle(os.path.join(prep_dir, "data_bfc.pkl"))

#%%
# Apply fixed-action window variant
data2 = flag_contributing_actions_advanced(data2)

#%%
#data2.to_pickle(os.path.join(prep_dir, "data_contributions_flag.pkl"))
data2= pd.read_pickle(os.path.join(prep_dir, "data_contributions_flag.pkl"))


# ---------------------------
# 4. Experiment 1
# ---------------------------
#%%
# Vectorized Quality Shot Function Generator
def make_quality_shot_fn(distance_threshold=20, angle_threshold=0.6, free_kick_distance_threshold=30,
                         tags_required=None):
    tags_required = tags_required or ['tag_1801', 'tag_201', 'tag_302', 'tag_703']
    def compute_is_quality_shot(df):
        df = df.copy()
        is_shot = (df['eventId'] == 10)
        is_fk_shot = (df['eventId'] == 3) & (df['subEventId'].isin([33, 35]))

        tag_ok = pd.Series(False, index=df.index)
        for tag in tags_required:
            tag_ok |= (df.get(tag, 0) == 1)

        low_pressure = ~((df.get('tag_701', 0) == 1) | (df.get('tag_1401', 0) == 1) | (df.get('tag_1501', 0) == 1))

        dist = df['distance_to_goal']
        angle = df['angle_to_goal']
        x, y = df['coordinates_x'], df['coordinates_y']

        inside_box = dist <= distance_threshold
        good_angle = angle >= angle_threshold
        central = (x >= 80) & (y >= 30) & (y <= 70)
        spatial_ok = inside_box | good_angle | central

        shot_qual = is_shot & (spatial_ok & (tag_ok | low_pressure))
        fk_qual = is_fk_shot & ((df['subEventId'] == 35) | ((df['subEventId'] == 33) & (dist <= free_kick_distance_threshold)))

        df['is_quality_shot'] = (shot_qual | fk_qual).astype(int)
        return df
    return compute_is_quality_shot

# Vectorized Key Action Generator
def make_key_action_fn(tags_used=None):
    tags_used = tags_used or ['tag_301', 'tag_302', 'tag_703']
    def compute_is_key_action(df):
        df = df.copy()
        df['is_key_action'] = df[tags_used].sum(axis=1) > 0
        return df
    return compute_is_key_action

# Compute Player Contribution Stats
def compute_player_contrib_stats(df):
    df = df[df['playerId'] != 0].copy()

    # Offensive and defensive flags
    df_game_ratios = df.groupby(['playerId', 'matchId']).agg(
        total=('playerId', 'count'),
        contrib=('contributes_to_quality', 'sum'),
        disrupts=('disrupts_opponent_quality', 'sum')
    ).reset_index()

    df_game_ratios['contrib_ratio'] = df_game_ratios['contrib'] / (df_game_ratios['total'] + 1e-6)
    df_game_ratios['disrupt_ratio'] = df_game_ratios['disrupts'] / (df_game_ratios['total'] + 1e-6)

    summary = df_game_ratios.groupby('playerId').agg(
        avg_contrib_ratio=('contrib_ratio', 'mean'),
        std_contrib_ratio=('contrib_ratio', 'std'),
        avg_disrupt_ratio=('disrupt_ratio', 'mean'),
        std_disrupt_ratio=('disrupt_ratio', 'std'),
        iqr_contrib_ratio=('contrib_ratio', lambda x: x.quantile(0.75) - x.quantile(0.25)),
        iqr_disrupt_ratio=('disrupt_ratio', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    ).reset_index()

    return summary, df_game_ratios

#%%
# Single Experiment Runner
def run_single_experiment(df, tracked_players, params):
    # Label generation
    compute_shot_fn = make_quality_shot_fn(
        distance_threshold=params['distance'],
        angle_threshold=params['angle'],
        free_kick_distance_threshold=params['fk_dist'],
        tags_required=params['quality_tags']
    )
    df = compute_shot_fn(df)
    df = compute_is_key_action(df)
    df = flag_contributing_actions_advanced(
        df,
        window=params['window'],
        possession_threshold=params['possession']
    )

    # Contribution stats
    stats, df_game_ratios = compute_player_contrib_stats(df)
    stats = stats[stats['playerId'].isin(tracked_players)]
    df_game_ratios = df_game_ratios[df_game_ratios['playerId'].isin(tracked_players)]

    # Add hyperparameter metadata
    for k, v in params.items():
        stats[k] = str(v) if isinstance(v, list) else v
        df_game_ratios[k] = str(v) if isinstance(v, list) else v

    return stats, df_game_ratios


#%%
# Full Parallel Experiment Runner (with CSV export)
def run_experiment_grid(df, tracked_players, param_grid, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    partial_path = os.path.join(results_dir, "experiment1_results_partial_opt.csv")
    final_path = os.path.join(results_dir, "experiment1_results_opt.csv")

    results = []
    is_first_write = True

    for params in tqdm(param_grid, desc="Running experiment grid"):
        try:
            result = run_single_experiment(df.copy(), tracked_players, params)
            results.append(result)

            # Append to partial CSV (ensures headers only once)
            result.to_csv(
                partial_path,
                mode='a',
                header=is_first_write,
                index=False
            )
            is_first_write = False  # After first write, don't write header again

        except Exception as e:
            print(f"⚠️ Skipping param set {params} due to error: {e}")
            continue

    # Combine full results and save
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(final_path, index=False)
        return final_df
    else:
        print("❌ No results generated. Check your experiment logic.")
        return pd.DataFrame()

#%%
# Param Grid Setup
## --- Parameters to explore (structure of labeling logic) ---
distances = [16, 18, 20]
angles = [0.6]
fk_distances = [25, 30]
windows = [3, 5, 7, 10]
possessions = [1, 2, 3]
# Most commonly accepted in sport analytics and useful for baseline
quality_tag_sets = [
    ['tag_201'],                        # Big Chance
    ['tag_101'],                        # Goal
    ['tag_201', 'tag_101'],            # Big Chance + Goal
]  # Accurate shot + Big Chance
key_tag_sets = [
    ['tag_301'],                        # Assist
    ['tag_302'],                        # Key Pass
    ['tag_703'],                        # Duel Won
]
# --- Grid building ---
param_grid = [
    {
        'distance': d,
        'angle': a,
        'fk_dist': fk,
        'window': w,
        'possession': p,
        'quality_tags': qt,
        'key_tags': kt
    }
    for d, a, fk, w, p, qt, kt in product(
        distances, angles, fk_distances,
        windows, possessions,
        quality_tag_sets, key_tag_sets
    )
]
#%%
# Top 100 most active players by volume
top_players_by_volume = (
    data2.loc[data2['playerId'] != 0, 'playerId']
    .value_counts()
    .nlargest(100)
    .index
)

print(top_players_by_volume)
#%%
# Top 20 goal scorers
is_goal = data2['tag_101'] == 1
is_shot = data2['eventId'].isin([10])  # tirs dans le jeu
is_set_piece = data2['subEventId'].isin([33, 35])  # CF direct, penalty

top_goal_scorers = (
    data2.loc[(is_goal & (is_shot | is_set_piece)), 'playerId']
    .value_counts()
    .nlargest(20)
    .index
)
print(top_goal_scorers)
#%%
# Top 20 assists provider
top_assist_providers = (
    data2.loc[(data2['tag_301'] == 1), 'playerId']
    .value_counts()
    .nlargest(20)
    .index
)

#%%
# Top 20 defenders
top_defenders = (
    data2.loc[
        (data2['tag_1401'] == 1) | 
        (data2['tag_701'] == 1) | 
        (data2['tag_703'] == 1) | 
        (data2['tag_702'] == 1),
        'playerId'
    ]
    .value_counts()
    .nlargest(20)
    .index
)
#%%
# Subset of players
tracked_players = (
    top_players_by_volume
    .union(top_goal_scorers)
    .union(top_assist_providers)
    .union(top_defenders)
)
#%%
# Lookup player info
player_lookup = (
    players_df
    .loc[players_df['wyId'].isin(tracked_players), ['wyId', 'shortName', 'fullName', 'role_name']]
    .rename(columns={'wyId': 'playerId'})
    .copy()
)
# Clean shortName encoding
player_lookup['shortName'] = player_lookup['shortName'].apply(
    lambda x: x.encode('utf-8').decode('unicode_escape') if isinstance(x, str) and '\\u' in x else x
)
#%%
# run Experiment 1
exp1_dir = os.path.join(results_dir, "ResultsDataExp1")
os.makedirs(exp1_dir, exist_ok=True)
results = run_experiment_grid(data2, tracked_players, param_grid, results_dir)

#%%
#Save results in csv
results.to_csv(os.path.join(exp1_dir, "results_raw.csv"), index=False)

#%%
#add names of players
results_w_names = results.merge(player_lookup, on = 'playerId', how = 'left')
#%%
results_w_names.to_csv(os.path.join(exp1_dir, "results_w_names.csv"), index=False)

#%%
# creation of the composite score
def add_composite_score(df, alpha=0.7):
    df = df.copy()
    if 'avg_contrib_ratio' in df.columns and 'avg_disrupt_ratio' in df.columns:
        df['composite_score'] = (
            alpha * df['avg_contrib_ratio'] +
            (1 - alpha) * df['avg_disrupt_ratio']
        )
    else:
        raise ValueError("Both avg_contrib_ratio and avg_disrupt_ratio must be in the DataFrame.")
    return df

#%%
results_w_names = add_composite_score(results_w_names, alpha=0.7)

#%%
alpha = 0.7
compositefilename = f"results_w_composite_alpha_{alpha:.1f}.csv"
results_w_names.to_csv(os.path.join(exp1_dir, compositefilename), index=False)

#%%
# best param from Experiment 1
best_params = {
    'distance': 20,
    'angle': 0.6,
    'fk_dist': 30,
    'window': 10,
    'possession': 3,
    'quality_tags': ['tag_201'],
    'key_tags': ['tag_703']
}


# %%
# Filter for window=7 and possession=2
subset = results_w_names[
    (results_w_names['window'] == 7) &
    (results_w_names['possession'] == 2)
]

# Compute average metrics across players
window7_pos2_metrics = subset[['avg_contrib_ratio', 'avg_disrupt_ratio', 'composite_score']].mean()
print("Metrics for window=7, possession=2:\n", window7_pos2_metrics)

# Optionally: compare with best overall config (e.g., row 647 from your previous message)
best_config_metrics = {
    'avg_contrib_ratio': 0.080557,
    'avg_disrupt_ratio': 0.024211,
    'composite_score': 0.063653
}
print("\nBest Config Metrics:\n", best_config_metrics)

# Difference (optional)
diff = window7_pos2_metrics - pd.Series(best_config_metrics)
print("\nDifference:\n", diff)

# %%
# Best param choose knowing sports knowledge
params_f = {
    'distance': 20,
    'angle': 0.6,
    'fk_dist': 30,
    'window': 7,
    'possession': 2,
    'quality_tags': ['tag_201'],
    'key_tags': ['tag_703']
}

# Run just the best
df_stats, df_game_ratios = run_single_experiment(data2.copy(), tracked_players, best_params)

#%%
df_stats.to_csv(os.path.join(exp1_dir, "df_stats.csv"), index=False)
df_game_ratios.to_csv(os.path.join(exp1_dir, "df_game_ratios.csv"), index=False)

#%%
df_stats = add_composite_score(df_stats, alpha=0.7)
# %%
df_stats_named = df_stats.merge(player_lookup, on='playerId', how='left')
df_game_ratios_named = df_game_ratios.merge(player_lookup, on='playerId', how='left')

#%%
df_stats_named.to_csv(os.path.join(exp1_dir, f"df_stats_named_alpha_{alpha:.1f}.csv"), index=False)
df_game_ratios_named.to_csv(os.path.join(exp1_dir, f"df_game_ratios_named_alpha_{alpha:.1f}.csv"), index=False)

#%%
player_with_salary = player_lookup.merge(
    salary_df_cleaned,
    on='fullName',
    how='left'
)
# %%
#missing_salary = player_with_salary[player_ary['weekly_wage_gbp'].isna()]with_sal

# %%
#print(missing_salary[['fullName', 'club']])

# %%
# missing salary data from Capologies
manual_salaries = {
    "Christian Benteke Liolo": {
        "weekly_wage_gbp": "£120,000",
        "annual_wage_gbp": "£6,240,000",
        "club": "Crystal Palace"
    },
    "Granit Xhaka": {
        "weekly_wage_gbp": "£80,000",
        "annual_wage_gbp": "£4,160,000",
        "club": "Arsenal"
    },
    "Johann Berg Guðmunds­son": {
        "weekly_wage_gbp": "£25,000",
        "annual_wage_gbp": "£1,300,000",
        "club": "Burnley"
    },
    "Onyinye Wilfred Ndidi": {
        "weekly_wage_gbp": "£30,000",
        "annual_wage_gbp": "£1,560,000",
        "club": "Leicester"
    },
    "José Salomón Rondón Giménez": {
        "weekly_wage_gbp": "£60,000",
        "annual_wage_gbp": "£3,120,000",
        "club": "West Bromwich"
    },
    "Jamal Lascelles": {
        "weekly_wage_gbp": "£40,000",
        "annual_wage_gbp": "£2,080,000",
        "club": "Newcastle"
    },
    "Mathias Jattah-Njie Jørgensen": {
        "weekly_wage_gbp": "£30,000",
        "annual_wage_gbp": "£1,560,000"
    },
    "Ahmed Hegazy": {
        "weekly_wage_gbp": "£25,000",
        "annual_wage_gbp": "£1,300,000"
    },
    "Cédric Ricardo Alves Soares": {
        "weekly_wage_gbp": "£50,000",
        "annual_wage_gbp": "£2,600,000"
    }
}

#%%
for name, values in manual_salaries.items():
    mask = player_with_salary['fullName'] == name
    if 'weekly_wage_gbp' in values:
        player_with_salary.loc[mask, 'weekly_wage_gbp'] = values.get('weekly_wage_gbp')
    if 'annual_wage_gbp' in values:
        player_with_salary.loc[mask, 'annual_wage_gbp'] = values.get('annual_wage_gbp')
    if 'club' in values:
        player_with_salary.loc[mask, 'club'] = values.get('club')

# %%
#duplicated_names = player_with_salary['fullName'][player_with_salary['fullName'].duplicated(keep=False)]
# %%
#duplicated_names
# %%
rows_to_remove = [
    ('Virgil van Dijk', 'Southampton'),
    ('Alexis Alejandro Sánchez Sánchez', 'Arsenal'),
    ('Henrikh Mkhitaryan', 'Manchester United'),
    ('Kurt Happy Zouma', 'Chelsea')
]

player_with_salary= player_with_salrary = player_with_salary[~player_with_salary[['fullName', 'club']].apply(tuple, axis=1).isin(rows_to_remove)]
# %%
merged_stats = df_stats.merge(
    player_with_salary[['playerId', 'fullName', 'role_name', 'annual_wage_gbp', 'club']],
    on='playerId',
    how='left'
)

# %%
merged_stats['annual_wage_gbp'] = (
    merged_stats['annual_wage_gbp']
    .astype(str)
    .str.replace('£', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
    .replace('', np.nan)
    .astype(float)
)


# %%
# premier league rankings
pl_ranks = {
    'Manchester City': 1, 'Manchester United': 2, 'Tottenham': 3,
    'Liverpool': 4, 'Chelsea': 5, 'Arsenal': 6, 'Burnley': 7,
    'Everton': 8, 'Leicester': 9, 'Newcastle': 10,
    'Crystal Palace': 11, 'Bournemouth': 12, 'West Ham': 13,
    'Watford': 14, 'Brighton': 15, 'Huddersfield': 16,
    'Southampton': 17, 'Swansea': 18, 'Stoke City': 19, 'West Bromwich': 20
}

# %%
merged_stats['club_rank'] = merged_stats['club'].map(pl_ranks)
merged_stats['club_rank'] = merged_stats['club_rank'].astype(int)

#%%
merged_stats = add_composite_score(merged_stats, alpha=alpha)
#%%
merged_stats.to_csv(os.path.join(exp1_dir, f"merged_stats_alpha_{alpha:.1f}.csv"), index=False)
# %%
# subrole definition from football knowledge
center_backs = [
    "Ahmed Hegazy",
    "Harry Maguire",
    "Shane Duffy",
    "Kurt Happy Zouma",
    "Michael Keane",
    "Jamal Lascelles",
    "Lewis Dunk",
    "Jan Vertonghen",
    "Laurent Koscielny",
    "Virgil van Dijk",
    "Nicolás Hernán Otamendi",
    "Christopher Schindler",
    "Steve Cook",
    "Wesley Hoedt",
    "Dejan Lovren",
    "Mathias Jattah-Njie Jørgensen",
    "Shkodran Mustafi",
    "Joël Andre Job Matip",
    "Simon Francis",
    "Alfie Mawson",
    "Nathan Aké",
    "Antonio Rüdiger",  
    "Davinson Sánchez Mina",
    "César Azpilicueta Tanco" 
]

full_backs = [
    "Marcos Alonso Mendoza",
    "Andrew Robertson",
    "Héctor Bellerín Moruno",
    "Sead Kolašinac",
    "Kieran Trippier",
    "Ignacio Monreal Eraso",
    "Ben Davies",
    "Ryan Bertrand",
    "Ashley Young",  
    "Pablo Javier Zabaleta Girod",
    "Kyle Walker",
    "Aaron Cresswell",
    "Luis Antonio Valencia Mosquera",
    "Kieran Gibbs",
    "Cédric Ricardo Alves Soares",
    "José Holebas",
    "Ben Chilwell",
    "Kyle Naughton",
    "Martin Olsson",
    "Charlie Daniels",
    "DeAndre Yedlin",
    "Joe Gomez",
      "Erik Pieters"  
]

def_mid = [
    "Onyinye Wilfred Ndidi",
    "Idrissa Gana Gueye",
    "N'Golo Kanté",
    "Eric Dier",
    "Jordan Brian Henderson",
    "Nemanja Matić",
    "Oriol Romeu Vidal",
    "Abdoulaye Doucouré",
    "Chris Brunt",
    "Mark Noble",
    "Lewis Cook",
    "Dale Stephens",
    "Tom Carroll",
    "Jonathan Hogg"
]
central_midfield = [
    "Kevin De Bruyne",
    "Mesut Özil",
    "Christian Dannemann Eriksen",
    "Bamidele Alli",
    "David Josué Jiménez Silva",
    "Francesc Fàbregas i Soler",
    "Paul Pogba",
    "Jonjo Shelvey",
    "Emre Can",
    "Granit Xhaka",
    "James Milner",
    "Fernando Luiz Rosa",  
    "Yohan Cabaye",
    "Joe Allen",
    "Fabian Delph",
    "Pascal Groß",
    "Davy Pröpper",
    "Luka Milivojević",
    "Aaron Mooy",
    "Jack Cork",
    "Moussa Sidi Yaya Dembélé"
]

winger_mid = [
    "Wilfried Zaha",
    "Leroy Sané",
    "Andros Townsend",
    "Matt Ritchie",
    "Dušan Tadić",
    "Henrikh Mkhitaryan",
    "Riyad Mahrez",
    "Xherdan Shaqiri",
    "Johann Berg Guðmunds­son",
    "James McArthur",  
    "Marc Albrighton"
]

winger_forwards = [
    "Mohamed Salah Ghaly",
    "Heung-Min Son",
    "Raheem Shaquille Sterling",
    "Eden Hazard",
    "Sadio Mané",
    "Richarlison de Andrade",
    "Jordan Ayew",
    "Alexis Alejandro Sánchez Sánchez"  
]

central_strikers = [
    "Sergio Leonel Agüero del Castillo",
    "Harry Kane",
    "Alexandre Lacazette",
    "Gabriel Fernando de Jesus",
    "Álvaro Borja Morata Martín",
    "Romelu Lukaku Menama",
    "Jamie Vardy",
    "José Salomón Rondón Giménez",
    "Christian Benteke Liolo",
    "Chris Wood",
    "Ashley Barnes",
    "Glenn Murray",
    "Roberto Firmino Barbosa de Oliveira", 
    "Marko Arnautović",  
    "Joshua King",
    "Pierre-Emerick Aubameyang" 
]

#%%
sub_roles_map = {
    'Defender': {
        'Full-back': full_backs,          
        'Centre-back': center_backs
    },
    'Midfielder': {
        'Winger': winger_mid,
        'Defensive Midfielder': def_mid,
        'Central/Attacking Midfielder': central_midfield
    },
    'Forward': {
        'Winger Forward': winger_forwards,
        'Central Striker': central_strikers
    }
}

# ---------------------------
# 5. Analyze Experiment 1
# ---------------------------

#%%
#%%
def get_top_players(results_df, metric='avg_contrib_ratio', top_k=10):
    """Return top players by max value of chosen metric across all configurations."""
    top_players = (
        results_df.loc[results_df.groupby('playerId')[metric].idxmax()]
        .sort_values(by=metric, ascending=False)
        .head(top_k)
    )
    return top_players


def plot_top_players_bar(results_df, metric='avg_contrib_ratio', top_k=10, title=None):
    top_players = get_top_players(results_df, metric, top_k)
    metric_label = metric.replace('_', ' ').title()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_players, x=metric, y='shortName', palette='viridis')
    plt.title(title or f'Top {top_k} Players by {metric_label}')
    plt.xlabel(metric_label)
    plt.ylabel("Player")
    plt.tight_layout()
    plt.show()

def plot_role_distribution(results_df, metric='avg_contrib_ratio'):
    best_params_per_player = results_df.loc[
        results_df.groupby('playerId')[metric].idxmax()
    ]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=best_params_per_player, x='role_name', y=metric, palette='Set2')
    plt.title(f'{metric.replace("_", " ").title()} Distribution by Role')
    plt.xlabel("Role")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def summarize_labeling_configs(results_df, top_k=10, alpha=0.7, use_composite=True):
    """Summarize best labeling configs using both metrics."""
    group_cols = ['distance', 'angle', 'fk_dist', 'window', 'possession', 'quality_tags', 'key_tags']
    summary = (
        results_df
        .groupby(group_cols)
        .agg(
            avg_contrib_ratio=('avg_contrib_ratio', 'mean'),
            avg_disrupt_ratio=('avg_disrupt_ratio', 'mean')
        )
        .reset_index()
    )
    if use_composite:
        summary['composite_score'] = (
            alpha * summary['avg_contrib_ratio'] +
            (1 - alpha) * summary['avg_disrupt_ratio']
        )
        return summary.sort_values(by='composite_score', ascending=False).head(top_k)
    else:
        return summary.sort_values(by='avg_contrib_ratio', ascending=False).head(top_k)

def plot_config_performance_scatter(summary_df):
    """Visualize trade-off between offensive and defensive labeling scores."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=summary_df,
        x='avg_contrib_ratio',
        y='avg_disrupt_ratio',
        size='composite_score',
        hue='composite_score',
        palette='viridis',
        legend='brief'
    )
    plt.title("Offense vs. Defense Trade-Off Across Labeling Configs")
    plt.xlabel("Avg Contribution Ratio")
    plt.ylabel("Avg Disruption Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_midfield_subrole_comparison(df_profiles, midfield_type_col='midfield_type', metric='avg_contrib_ratio'):
    """Compare contribution metric across central and wide midfielders."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_profiles, x=midfield_type_col, y=metric, palette='muted')
    plt.title(f'{metric.replace("_", " ").title()}: Central vs Wide Midfielders')
    plt.xlabel("Midfielder Type")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%

# Create full summary of all configs
param_summary = (
    results_w_names
    .groupby(['distance', 'angle', 'fk_dist', 'window', 'possession', 'quality_tags', 'key_tags'])
    .agg(
        avg_contrib_ratio=('avg_contrib_ratio', 'mean'),
        avg_disrupt_ratio=('avg_disrupt_ratio', 'mean'),
        composite_score=('composite_score', 'mean')  # already exists
    )
    .reset_index()
)

#%%
param_summary.to_csv(os.path.join(exp1_dir, f"param_summary_alpha_{alpha:.1f}.csv"), index=False)
#%% Top 5 Labeling Configs Table

top_configs = param_summary.sort_values(by='composite_score', ascending=False).head(5)


#%%
metrics = {
    "avg_contrib_ratio": "Top 10 Configs by Avg Contribution Ratio",
    "avg_disrupt_ratio": "Top 10 Configs by Avg Disruption Ratio",
    "composite_score": "Top 10 Configs by Composite Score"
}
dfs = []
for metric in ['avg_contrib_ratio', 'avg_disrupt_ratio', 'composite_score']:
    df = param_summary.sort_values(by=metric, ascending=False).head(10).copy()
    df['metric_ranked_by'] = metric
    dfs.append(df)

combined_table = pd.concat(dfs, ignore_index=True)

cols = ['metric_ranked_by', 'distance', 'angle', 'fk_dist', 'window', 'possession', 'quality_tags', 'key_tags', 
        'avg_contrib_ratio', 'avg_disrupt_ratio', 'composite_score']
combined_table = combined_table[cols]

# Export to LaTeX
print(
    combined_table.to_latex(
        index=False,
        longtable=True,
        caption='Top 10 Labeling Configs Ranked by Each Metric Separately',
        label='tab:top-configs-combined'
    )
)
#%% 
# B. Scatter Plot – Trade-Off Space
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=param_summary,
    x='avg_contrib_ratio',
    y='avg_disrupt_ratio',
    size='composite_score',
    hue='composite_score',
    palette='viridis'
)
plt.title("Offense vs. Defense Trade-Off Across Labeling Configs")
plt.xlabel("Avg Contribution Ratio")
plt.ylabel("Avg Disruption Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Top Players Barplots (for all 3 metrics)
for metric in ['avg_contrib_ratio', 'avg_disrupt_ratio', 'composite_score']:
    top_players = results_w_names.loc[results_w_names.groupby('playerId')[metric].idxmax()]
    top_players = top_players.sort_values(by=metric, ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_players, x=metric, y='shortName', palette='viridis')
    plt.title(f'Top 10 Players by {metric}')
    plt.tight_layout()
    plt.show()
# %%
top_disruptors = results_w_names.loc[results_w_names.groupby('playerId')['avg_disrupt_ratio'].idxmax()]
top10_disruptors = top_disruptors.sort_values(by='avg_disrupt_ratio', ascending=False).head(20)
print(top10_disruptors[['shortName', 'role_name', 'avg_disrupt_ratio']])
# %%
best_defenders = top_disruptors[top_disruptors['role_name'] == 'Defender'].head(10)
# %%
print(best_defenders.sort_values(by='avg_disrupt_ratio', ascending=False)[['shortName', 'role_name', 'avg_disrupt_ratio']])
# %% Role-Based Distribution Boxplots

for metric in ['avg_contrib_ratio', 'avg_disrupt_ratio', 'composite_score']:
    best_params_per_player = results_w_names.loc[results_w_names.groupby('playerId')[metric].idxmax()]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=best_params_per_player, x='role_name', y=metric, palette='Set2')
    plt.title(f'{metric} Distribution by Role')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %%
best_param_row = param_summary.sort_values(by='composite_score', ascending=False).iloc[0]
# %%
def generate_flagged_df(df_raw, best_params):
    quality_fn = make_quality_shot_fn(
        distance_threshold=best_params['distance'],
        angle_threshold=best_params['angle'],
        free_kick_distance_threshold=best_params['fk_dist'],
        tags_required=eval(best_params['quality_tags'])  # Important: handle stringified lists
    )
    key_fn = make_key_action_fn(tags_used=eval(best_params['key_tags']))

    df = df_raw.copy()
    df = quality_fn(df)
    df = key_fn(df)
    df = flag_contributing_actions_advanced(
        df,
        window=int(best_params['window']),
        possession_threshold=int(best_params['possession'])
    )
    return df
# %%
print(best_param_row)
# %% Distribution of Composite Score
sns.histplot(merged_stats['composite_score'], kde=True)
plt.title("Distribution of Composite Score (Per Player)")
plt.xlabel("Composite Score")
plt.show()

# %% Composite Score by Role
sns.boxplot(data=merged_stats, x='role_name', y='composite_score')
plt.title("Composite Score by Role")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
# %%
from scipy.stats import spearmanr
corr_salary, p_salary = spearmanr(merged_stats['composite_score'], merged_stats['annual_wage_gbp'])
# %% Scatter: Salary vs Composite Score
sns.scatterplot(data=merged_stats, x='annual_wage_gbp', y='composite_score', hue='role_name')
plt.title(f'Composite Score vs Salary\n(Spearman r={corr_salary:.2f}, p={p_salary:.4f})')
plt.show()
# %%
corr_rank, p_rank = spearmanr(merged_stats['composite_score'], merged_stats['club_rank'])
# %%
print(corr_rank, p_rank)
# %%
sns.boxplot(data=merged_stats, x='club_rank', y='composite_score')
plt.title(f'Composite Score vs Club Ranking\n(Spearman r={corr_rank:.2f}, p={p_rank:.4f})')
plt.xlabel("Premier League Rank (1 = Best)")
plt.show()
# %%
best_params_per_player = results_w_names.loc[
    results_w_names.groupby('playerId')['composite_score'].idxmax()
].copy()

df_profiles = best_params_per_player.merge(
    player_with_salary[['playerId', 'fullName', 'role_name', 'club', 'annual_wage_gbp']],
    on='playerId',
    how='left'
)

# Prefer values from the salary file when available
df_profiles['fullName'] = df_profiles['fullName_y'].combine_first(df_profiles['fullName_x'])
df_profiles['role_name'] = df_profiles['role_name_y'].combine_first(df_profiles['role_name_x'])

# Drop old columns
df_profiles = df_profiles.drop(columns=[
    'fullName_x', 'fullName_y', 'role_name_x', 'role_name_y'
])
# %%
top_defenders = df_profiles[df_profiles['role_name'] == "Defender"].sort_values(by='composite_score', ascending=False)
top_forwards = df_profiles[df_profiles['role_name'] == "Forward"].sort_values(by='composite_score', ascending=False)
top_midfielders = df_profiles[df_profiles['role_name'] == "Midfielder"].sort_values(by='composite_score', ascending=False)

# %%
print(top_midfielders['fullName'])
# %%
#%%
print(top_forwards['fullName'])


# %%
print(top_defenders['fullName'])


#%%
def assign_sub_role(row):
    name = row['fullName']
    role = row['role_name']

    if role == 'Defender':
        if name in full_backs:
            return 'Full-back'
        elif name in center_backs:
            return 'Centre-back'
    elif role == 'Midfielder':
        if name in winger_mid:
            return 'Winger'
        elif name in def_mid:
            return 'Defensive Midfielder'
        elif name in central_midfield:
            return 'Central/Attacking Midfielder'
    elif role == 'Forward':
        if name in winger_forwards:
            return 'Winger Forward'
        elif name in central_strikers:
            return 'Central Striker'
    return 'Unknown'

# %%
df_profiles['sub_role'] = df_profiles.apply(assign_sub_role, axis=1)
# %%

# Aggregate statistics by sub_role
subrole_stats = (
    df_profiles.groupby('sub_role')
    .agg(
        avg_composite=('composite_score', 'mean'),
        std_composite=('composite_score', 'std'),
        iqr_composite=('composite_score', lambda x: x.quantile(0.75) - x.quantile(0.25)),
        n_players=('playerId', 'nunique')
    )
    .reset_index()
    .sort_values(by='avg_composite', ascending=False)
)

print(subrole_stats)

# Plot boxplot by subrole
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_profiles, x='sub_role', y='composite_score', palette='coolwarm')
plt.title("Composite Score Distribution by Player Sub-Role")
plt.ylabel("Composite Score")
plt.xlabel("Player Sub-Role")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
df_profiles[['fullName', 'role_name', 'sub_role']]
# %%
df_profiles_subset = df_profiles[['fullName', 'role_name', 'sub_role']]
# %%
df_profiles_subset.to_csv('df_subset.txt', sep='\t', index=False)
# %%
top15_df = (
    merged_stats[['fullName', 'club', 'role_name', 'composite_score', 'annual_wage_gbp']]
    .dropna(subset=['composite_score', 'annual_wage_gbp'])
    .sort_values(by='composite_score', ascending=False)
    .head(15)
    .copy()
)
# %%
top15_df.to_csv('df_top15.txt', sep='\t', index=False)
# %%
plot_top_players_bar(df_profiles, metric='composite_score', top_k=10)
# %%
# Define the players to highlight
merged_stats['fullName'] = merged_stats['fullName_x']
merged_stats.drop(columns=['fullName_x', 'fullName_y'], errors='ignore', inplace=True)
#%%
highlight_players = [
    'Harry Kane',
    'Mohamed Salah Ghaly',
    'Sergio Leonel Agüero del Castillo',
    'Richarlison de Andrade',
    'Alexis Alejandro Sánchez Sánchez',
    'Mesut Özil',
    'Paul Pogba',
    'Kevin De Bruyne',
    'Virgil van Dijk'
]

#%%
plt.figure(figsize=(8, 6))  # un peu plus large pour faire de la place à droite

sns.scatterplot(
    data=merged_stats,
    x='annual_wage_gbp',
    y='composite_score',
    hue='role_name',
    alpha=0.7,
    s=60
)

# Affichage de la légende à droite
plt.legend(
    title='Role',
    bbox_to_anchor=(1.02, 1),   # déplace la légende à droite
    loc='upper left',
    borderaxespad=0
)

# Annoter les joueurs (avec nettoyage si besoin)
for _, row in merged_stats[merged_stats['fullName'].isin(highlight_players)].iterrows():
    plt.text(
        row['annual_wage_gbp'],
        row['composite_score'],
        row['fullName'],
        fontsize=8,
        weight='bold',
        color='black'
    )

# Titres et axes
plt.title(f'Composite Score vs Salary\n(Spearman r={corr_salary:.2f})')
plt.xlabel("Annual Wage (GBP)")
plt.ylabel("Composite Score")
plt.grid(True)

plt.tight_layout()
plt.show()
# %%
# Filter De Bruyne's full season data
kdb_stats = df_profiles[df_profiles['fullName'].str.contains("Kevin De Bruyne", case=False)]

# Print or inspect his composite score values 
print(kdb_stats[['fullName', 'composite_score', 'avg_contrib_ratio', 'avg_disrupt_ratio']])

# %%
midfielders = df_profiles[df_profiles['role_name'] == 'Midfielder']
plt.figure(figsize=(8, 6))
sns.boxplot(data=midfielders, y='composite_score', color='lightblue')
plt.scatter(x=0, y=kdb_stats['composite_score'].values[0], color='red', s=100, label='K. De Bruyne')
plt.title("Composite Score of Midfielders (Red = Kevin De Bruyne)")
plt.ylabel("Composite Score")
plt.xticks([])
plt.legend()
plt.show()

#%%
# Define players to highlight and assign colors
highlight_players = {
    "Kevin De Bruyne": "red",
    "Paul Pogba": "green",
    "David Josué Jiménez Silva": "blue",
    "N'Golo Kanté": "orange",
    "Mesut Özil": "purple"
}

# Filter for midfielders and plot boxplot
midfielders = df_profiles[df_profiles['role_name'] == 'Midfielder']
plt.figure(figsize=(9, 6))
sns.boxplot(data=midfielders, y='composite_score', color='lightgray')

# Plot each highlighted player
for player, color in highlight_players.items():
    player_score = df_profiles[df_profiles['fullName'].str.contains(player, case=False, na=False)]
    if not player_score.empty:
        plt.scatter(
            x=0,
            y=player_score['composite_score'].values[0],
            color=color,
            s=100,
            label=player
        )

# Final touches
plt.title("Composite Score of Midfielders with Key Profiles Highlighted")
plt.ylabel("Composite Score")
plt.xticks([])
plt.legend()
plt.tight_layout()
plt.show()
#%%
df_profiles['sub_role'].unique()
# %%

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=merged_stats,                
    x='annual_wage_gbp',
    y='composite_score',
    hue='role_name',                 
    alpha=0.8,
    s=70
)
plt.title("Composite Score vs. Annual Wage")
plt.xlabel("Annual Wage (£)")
plt.ylabel("Composite Score")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
clean_df = merged_stats[['annual_wage_gbp', 'composite_score']].dropna()

#  Spearman correlation
corr, p_value = spearmanr(clean_df['annual_wage_gbp'], clean_df['composite_score'])

print(f"📈 Spearman correlation: r = {corr:.2f}, p = {p_value:.4f}")
# %%
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=results_w_names,
    x='window',
    y='composite_score',
    palette='coolwarm'
)
plt.title("Distribution of Composite Score by Window Size")
plt.xlabel("Window Size")
plt.ylabel("Composite Score")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
window_summary = (
    results_w_names.groupby('window')
    .agg(
        mean_contrib=('avg_contrib_ratio', 'mean'),
        mean_disrupt=('avg_disrupt_ratio', 'mean'),
        mean_composite=('composite_score', 'mean')
    )
    .reset_index()
)

# Lineplot
plt.figure(figsize=(10, 6))
sns.lineplot(data=window_summary, x='window', y='mean_contrib', label='Avg Contribution Ratio')
sns.lineplot(data=window_summary, x='window', y='mean_disrupt', label='Avg Disruption Ratio')
sns.lineplot(data=window_summary, x='window', y='mean_composite', label='Composite Score')
plt.title("Average Metrics by Window Size")
plt.xlabel("Window")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
# players list
highlight_players = [
    "Harry Kane",
    "Mohamed Salah Ghaly",
    "Kevin De Bruyne",
    "Eden Hazard",
    "Marcos Alonso Mendoza"
]

df_highlight = results_w_names[results_w_names['fullName'].isin(highlight_players)].copy()


plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_highlight,
    x='window',
    y='composite_score',
    hue='shortName',  # ou 'fullName'
    marker='o'
)
plt.title("Composite Score vs. Window Size (Highlighted Players)")
plt.xlabel("Window")
plt.ylabel("Composite Score")
plt.grid(True)
plt.tight_layout()
plt.legend(title="Player")
plt.show()
# %%
pivot = (
    results_w_names.groupby(['window', 'possession'])['composite_score']
    .mean()
    .reset_index()
    .pivot(index='window', columns='possession', values='composite_score')
)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, cmap='viridis', fmt=".3f")
plt.title("Composite Score by Window and Possession Threshold")
plt.xlabel("Possession Threshold")
plt.ylabel("Window Size")
plt.tight_layout()
plt.show()
# %%
player_sensitivity = (
    results_w_names.groupby('playerId')['composite_score']
    .agg(['mean', 'std', lambda x: x.max() - x.min()])
    .rename(columns={'<lambda_0>': 'range'})
)

player_sensitivity = player_sensitivity.merge(player_lookup, on='playerId')
player_sensitivity.sort_values(by='std', ascending=False).head(10)

#%%
player_sensitivity_sorted = player_sensitivity.sort_values(by='std', ascending=False)

# 
with open("player_sensitivity_sorted.tex", "w") as f:
    f.write(player_sensitivity_sorted.to_latex(index=False, float_format="%.2f"))
# %%
top10_sensitivity = player_sensitivity.sort_values(by='std', ascending=False).head(10)

with open("player_sensitivity_top10.tex", "w") as f:
    f.write(top10_sensitivity.to_latex(index=False, float_format="%.2f"))

#%%
alphas = np.linspace(0, 1, 11)  # e.g. 0.0 to 1.0
correlations = []

for a in alphas:
    df_temp = merged_stats.copy()
    df_temp['composite_score'] = a * df_temp['avg_contrib_ratio'] + (1 - a) * df_temp['avg_disrupt_ratio']
    df_temp = df_temp.dropna(subset=['composite_score', 'annual_wage_gbp'])

    r, _ = spearmanr(df_temp['composite_score'], df_temp['annual_wage_gbp'])
    correlations.append(r)

# Plot
plt.figure(figsize=(8, 5))
sns.lineplot(x=alphas, y=correlations, marker='o')
plt.title("Spearman Correlation: Composite Score vs Salary by Alpha")
plt.xlabel("Alpha (weight of contribution)")
plt.ylabel("Spearman Correlation (ρ)")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
role_alpha_matrix = []

for a in alphas:
    df_temp = merged_stats.copy()
    df_temp['composite_score'] = a * df_temp['avg_contrib_ratio'] + (1 - a) * df_temp['avg_disrupt_ratio']
    mean_scores = df_temp.groupby('role_name')['composite_score'].mean()
    role_alpha_matrix.append(mean_scores)

df_role_alpha = pd.DataFrame(role_alpha_matrix, index=alphas)

plt.figure(figsize=(10, 6))
sns.heatmap(df_role_alpha.T, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Mean Composite Score per Role across Alpha values")
plt.xlabel("Alpha")
plt.ylabel("Role")
plt.tight_layout()
plt.show()
# %%
target_players = ['Kevin De Bruyne', 'N\'Golo Kanté', 'Harry Kane']
player_rankings = {name: [] for name in target_players}

for a in alphas:
    df_temp = df_profiles.copy()
    df_temp['composite_score'] = a * df_temp['avg_contrib_ratio'] + (1 - a) * df_temp['avg_disrupt_ratio']
    df_temp = df_temp.sort_values(by='composite_score', ascending=False).reset_index(drop=True)

    for name in target_players:
        rank = df_temp[df_temp['fullName'] == name].index.values
        player_rankings[name].append(rank[0] + 1 if len(rank) > 0 else np.nan)

# Plot
plt.figure(figsize=(10, 6))
for name, ranks in player_rankings.items():
    sns.lineplot(x=alphas, y=ranks, label=name, marker='o')

plt.title("Player Ranking vs Alpha (Lower = Better)")
plt.xlabel("Alpha")
plt.ylabel("Ranking")
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
vandijk_results = results_w_names[
    results_w_names['fullName'].str.contains("Virgil van Dijk", case=False, na=False)
]
# %%
vandijk_window_scores = (
    vandijk_results
    .groupby('window')['composite_score']
    .mean()
    .reset_index()
)
# %%
plt.figure(figsize=(8, 5))
sns.lineplot(data=vandijk_window_scores, x='window', y='composite_score', marker='o')
plt.title("Évolution du Composite Score de Van Dijk selon la taille de fenêtre (window)")
plt.xlabel("Window")
plt.ylabel("Composite Score")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
# a rerun
players_def_test = ["Virgil van Dijk", "Marcos Alonso Mendoza", "Harry Maguire"]
df_focus = results_w_names[results_w_names['fullName'].isin(players_def_test)]

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_focus, x='window', y='composite_score', hue='shortName', marker='o')
plt.title("Composite Score vs Window – Center Back")
plt.xlabel("Window")
plt.ylabel("Composite Score")
plt.grid(True)
plt.tight_layout()
plt.legend(title="Joueur")
plt.show()
# %%
sns.boxplot(data=results_w_names, x='role_name', y='avg_disrupt_ratio')
# %%
alphas = np.linspace(0, 1, 11)
scores = []

for a in alphas:
    temp = df_profiles.copy()
    temp['composite_score'] = a * temp['avg_contrib_ratio'] + (1 - a) * temp['avg_disrupt_ratio']
    vvd_score = temp[temp['fullName'].str.contains("Virgil van Dijk", case=False)]['composite_score'].values[0]
    scores.append(vvd_score)

plt.plot(alphas, scores, marker='o')
plt.title("Composite Score de Van Dijk selon alpha")
plt.xlabel("Alpha (poids offensif)")
plt.ylabel("Composite Score")
plt.grid(True)
plt.show()
# %%
players = {
    "Virgil van Dijk": "blue",
    "N'Golo Kanté": "orange",
    "Kevin De Bruyne": "green",
    "Harry Kane": "red"
}

alphas = np.linspace(0, 1, 11)
player_scores = {name: [] for name in players}

for alpha in alphas:
    df_temp = df_profiles.copy()
    df_temp['composite_score'] = (
        alpha * df_temp['avg_contrib_ratio'] +
        (1 - alpha) * df_temp['avg_disrupt_ratio']
    )
    for player in players:
        score = df_temp[df_temp['fullName'].str.contains(player, case=False)]['composite_score'].values[0]
        player_scores[player].append(score)

# Plot
plt.figure(figsize=(9, 6))
for player, color in players.items():
    plt.plot(alphas, player_scores[player], label=player, marker='o', color=color)

plt.title("Evolution of the composite score according to $\alpha$ (offensive weight)")
plt.xlabel("Alpha")
plt.ylabel("Composite Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%
df_profiles['club_rank'] = df_profiles['club'].map(pl_ranks)
#%%
center_backs_df = df_profiles[df_profiles['sub_role'] == 'Centre-back']
top_disruptors_cb = center_backs_df.sort_values(by='avg_disrupt_ratio', ascending=False).head(10)
#%%
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_disruptors_cb,
    y='fullName',
    x='avg_disrupt_ratio',
    palette='Blues_d'
)
plt.xlabel("Average Disrupt Ratio")
plt.ylabel("Centre-back")
plt.title("Top 10 Centre-Backs by Disruptive Defensive Actions")
plt.tight_layout()
plt.show()
# %%
# only central def
cb_df = df_profiles[df_profiles['sub_role'] == "Centre-back"].copy()
cb_df = cb_df.dropna(subset=['club_rank'])

plt.figure(figsize=(10, 6))

sns.scatterplot(data=cb_df, x='club_rank', y='avg_disrupt_ratio', hue='fullName')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="fullName", prop={'size': 8})

# Akey players annotation
highlight_players = ['Virgil van Dijk', 'Nicolás Hernán Otamendi', 'Jan Vertonghen']
for _, row in cb_df[cb_df['fullName'].isin(highlight_players)].iterrows():
    plt.text(row['club_rank'] + 0.2, row['avg_disrupt_ratio'], row['fullName'], fontsize=9)

plt.title("Disrupt Ratio vs Club Rank (Centre-backs)")
plt.xlabel("Premier League Rank (1 = Best Team)")
plt.ylabel("Average Disrupt Ratio")
plt.xticks(np.arange(int(cb_df['club_rank'].min()), int(cb_df['club_rank'].max()) + 1, 1))

plt.gca().invert_xaxis()  # Rank 1 = best team
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
cb_df['rank_quartile'] = pd.qcut(cb_df['club_rank'], q=4, labels=["Top clubs", "Q2", "Q3", "Bottom clubs"])

sns.boxplot(data=cb_df, x='rank_quartile', y='avg_disrupt_ratio')
plt.title("Disrupt Ratio by Quartile of Club Ranking (Centre-backs)")
plt.xlabel("Quartile du classement EPL")
plt.ylabel("Disrupt Ratio moyen")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=results_w_names,
    x='possession',
    y='avg_disrupt_ratio',
    marker='o',
    estimator='mean',
    ci=None
)
plt.title("Average Disruption Ratio by Possession Threshold")
plt.xlabel("Possession Threshold")
plt.ylabel("Average Disrupt Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".3f")
plt.title("Composite Score by Window and Possession Threshold")
plt.ylabel("Window Size")
plt.xlabel("Possession Threshold")
plt.tight_layout()
plt.savefig("heatmap_window_possession_composite.png")
plt.show()
# %%
if 'quality_tags_str' not in param_summary.columns:
    param_summary['quality_tags_str'] = param_summary['quality_tags'].astype(str)

plt.figure(figsize=(10, 6))
sns.barplot(data=param_summary, x='quality_tags_str', y='composite_score', palette='pastel')
plt.title("Composite Score by Quality Tags Used")
plt.ylabel("Composite Score")
plt.xlabel("Quality Tags")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("barplot_composite_by_tags.png")
plt.show()
# %%
latex_table = param_summary[[
    'window', 'possession', 'distance', 'angle', 'fk_dist',
    'quality_tags_str', 'key_tags', 'avg_contrib_ratio',
    'avg_disrupt_ratio', 'composite_score'
]].rename(columns={
    'quality_tags_str': 'quality_tags'
}).to_latex(index=False, longtable=True, caption="All tested labeling configurations", label="tab:labeling-full-grid")

with open("full_configurations_table.tex", "w") as f:
    f.write(latex_table)
# %%
print(param_summary['quality_tags'].unique())
# %%
print(
    param_summary
    .groupby('quality_tags_str')['composite_score']
    .describe()
)
# %%
param_summary = (
    results_w_names
    .groupby(['distance', 'angle', 'fk_dist', 'window', 'possession', 'quality_tags', 'key_tags'])
    .agg(
        avg_contrib_ratio=('avg_contrib_ratio', 'mean'),
        avg_disrupt_ratio=('avg_disrupt_ratio', 'mean'),
        composite_score=('composite_score', 'mean')
    )
    .reset_index()
)

param_summary['quality_tags_str'] = param_summary['quality_tags'].astype(str)
# %%
plt.figure(figsize=(10, 6))
sns.barplot(data=param_summary, x='quality_tags_str', y='composite_score', palette='pastel')
plt.title("Composite Score by Quality Tags Used")
plt.ylabel("Composite Score")
plt.xlabel("Quality Tags")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%


# ---------------------------
# 6. Experiment 2
# ---------------------------

base_dir = "C:/Users/loicm/Documents/2024-2025/MasterThesis"
results_dir = os.path.join(base_dir, "ResultsData")
models_dir = os.path.join(base_dir, "Models")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
results_path = os.path.join(results_dir, "results_exp2.csv")

#%% Load data
data3 = pd.read_pickle(os.path.join(prep_dir, "data_bfc.pkl"))

#%% Prepare match metadata
matches_info = matches_df.rename(columns={'wyId': 'matchId'})
matches_info['dateutc'] = pd.to_datetime(matches_info['dateutc'])
matches_info = matches_info[['matchId', 'gameweek', 'dateutc', 'teamsData']]

#%% Prepare player features
player_features = players_df[['wyId', 'role_code', 'foot', 'height', 'weight']].rename(columns={'wyId': 'playerId'})

# Metadata for visualization only (e.g., names)
player_metadata = players_df[['wyId', 'shortName', 'role_code']].rename(columns={'wyId': 'playerId'})

#%% Define role counting function (for formation complexity)
def count_roles(player_list):
    roles = player_features[player_features['playerId'].isin(player_list)]['role_code']
    num_fw = (roles == 'FW').sum()
    num_md = (roles == 'MD').sum()
    num_df = (roles == 'DF').sum()
    return num_fw, num_md, num_df

#%%
def extract_home_away_flag(events_df, matches_df):
    df = events_df.copy()
    home_team_dict = {
        row['matchId']: list(row['teamsData'].keys())[0]
        for _, row in matches_df.iterrows()
    }
    df['is_home_team'] = df.apply(
        lambda row: int(str(row['teamId']) == str(home_team_dict.get(row['matchId'], -1))),
        axis=1
    )
    return df

#%%
def compute_substitution_load(events_df, matches_df):
    df = events_df.copy()
    load = []

    for idx, row in df.iterrows():
        match_id = row['matchId']
        team_id = str(row['teamId'])
        event_sec = row['eventSec']

        teams_data = matches_df[matches_df['matchId'] == match_id]['teamsData'].values[0]
        subs = teams_data.get(team_id, {}).get('formation', {}).get('substitutions', [])
        count = sum(1 for sub in subs if sub['minute'] * 60 <= event_sec)
        load.append(count)

    df['substitution_load'] = load
    return df
#%%
def flag_fresh_substitutes(events_df, matches_df):
    df = events_df.copy()
    fresh = []

    for idx, row in df.iterrows():
        match_id = row['matchId']
        player_id = row['playerId']
        event_sec = row['eventSec']

        teams_data = matches_df[matches_df['matchId'] == match_id]['teamsData'].values[0]
        all_subs = []
        for team in teams_data.values():
            all_subs.extend(team.get('formation', {}).get('substitutions', []))

        sub_time = next((sub['minute'] * 60 for sub in all_subs if sub['playerIn'] == player_id), None)

        if sub_time is not None and (event_sec - sub_time) <= 300:
            fresh.append(1)
        else:
            fresh.append(0)

    df['is_fresh_substitute'] = fresh
    return df

#%% Create formation complexity for each match/team
formation_data = []

for _, match in matches_info.iterrows():
    match_id = match['matchId']
    teams = match['teamsData']

    for team_id_str, team_info in teams.items():
        team_id = int(team_id_str)
        lineup = team_info.get('formation', {}).get('lineup', [])
        player_ids = [p['playerId'] for p in lineup]

        num_FW, num_MD, num_DF = count_roles(player_ids)
        formation_complexity = (num_FW * 2 + num_MD) - (num_DF * 2)

        formation_data.append({
            'matchId': match_id,
            'teamId': team_id,
            'num_FW': num_FW,
            'num_MD': num_MD,
            'num_DF': num_DF,
            'formation_complexity': formation_complexity
        })

#%% Convert to DataFrame
formation_df = pd.DataFrame(formation_data)

#%%
formation_df['formation_complexity'].value_counts()

#%%
bins_form = [-8, -3, 0, 3, 8]
labels_form = ['Defensive', 'Balanced', 'Attacking', 'Very Attacking']
formation_df['formation_bin'] = pd.cut(formation_df['formation_complexity'], bins=bins_form, labels=labels_form)

#%%

# Build once: match_team_subs[(matchId, teamId)] = [sub_minutes]
match_team_subs = {}

for _, row in matches_info.iterrows():
    match_id = row['matchId']
    teams_data = row['teamsData']

    if isinstance(teams_data, str):
        teams_data = json.loads(teams_data) 

    for team_id_str, team_info in teams_data.items():
        team_id = int(team_id_str)
        subs = team_info.get('formation', {}).get('substitutions', [])
        sub_minutes = [sub['minute'] * 60 for sub in subs if isinstance(sub, dict)]
        match_team_subs[(match_id, team_id)] = sub_minutes

#%%
player_sub_times = {}

for _, row in matches_info.iterrows():
    match_id = row['matchId']
    teams_data = row['teamsData']

    if isinstance(teams_data, str):
        teams_data = json.loads(teams_data)

    for team_data in teams_data.values():
        subs = team_data.get('formation', {}).get('substitutions', [])
        for sub in subs:
            if isinstance(sub, dict):
                player_sub_times[(match_id, sub['playerIn'])] = sub['minute'] * 60
#%%
def fast_sub_load(row):
    subs = match_team_subs.get((row['matchId'], row['teamId']), [])
    return sum(1 for s in subs if s <= row['eventSec'])

def fast_fresh_sub(row):
    sub_time = player_sub_times.get((row['matchId'], row['playerId']), None)
    if sub_time is not None and (row['eventSec'] - sub_time) <= 300:
        return 1
    return 0

#%% Merge with main event data later using:
# data_exp2 = data_exp2.merge(formation_df, on=['matchId', 'teamId'], how='left')

#data_exp2 = flag_contributing_actions_advanced(data3, window=7, possession_threshold=2)
#data_exp2.to_pickle(os.path.join(prep_dir, "data_contributions_exp2.pkl"))
data_exp2 =pd.read_pickle(os.path.join(prep_dir, "data_contributions_exp2.pkl"))

#%%

data_exp2['steps_to_shot'].value_counts()

#%%

data_exp2['steps_to_shot'].isna().value_counts()

#%%
data_exp2['contributes_to_quality'].value_counts(normalize=True)

#%%
data_exp2['is_quality_shot'].sum()

(data_exp2['steps_to_shot'] == 0).sum()
#%%
print(matches_info.columns)
print(formation_df.columns)
#%% Merge match metadata first (gameweek, date, teamsData)

#%%
data_exp2 = data_exp2.drop(columns=['gameweek', 'dateutc', 'formation_complexity', 'formation_bin'], errors='ignore')

#%%
data_exp2 = data_exp2.merge(
    matches_info[['matchId', 'gameweek', 'dateutc']],
    on='matchId', how='left'
)


#%%
data_exp2 = data_exp2.merge(
    formation_df[['matchId', 'teamId', 'formation_complexity', 'formation_bin']],
    on=['matchId', 'teamId'], how='left'
)
#%%
#data_exp2.to_pickle(os.path.join(prep_dir, "data_bft.pkl"))
data_exp2 = pd.read_pickle(os.path.join(prep_dir, "data_bft.pkl"))
#%%
print(data_exp2.columns.tolist())

#%%
#%% for extend data
#data_exp2['substitution_load'] = data_exp2.apply(fast_sub_load, axis=1)
#%%
#data_exp2['is_fresh_substitute'] = data_exp2.apply(fast_fresh_sub, axis=1)
#%%
#data_exp2 = extract_home_away_flag(data_exp2, matches_info)

#%%
#data_exp2.to_pickle(os.path.join(prep_dir, "data_contributions_exp2_extend.pkl"))
#data_ext = pd.read_pickle(os.path.join(prep_dir, "data_contributions_exp2_extend.pkl"))

#%% Features
target_col = 'contributes_to_quality'

leakage_cols = [
    'eventId', 'subEventId', 'is_quality_shot', 'is_key_action', 
    'steps_to_shot', 'disrupts_opponent_quality', 'id', 'matchId',
    'playerId', 'matchPeriod', 'prev_eventId', 'prev_subEventId'
]

categorical_features = [
     'role_code', 'team_game_state',
    'is_team_losing_before_event', 'is_clutch_moment'
]

numerical_features = [
    'distance_to_goal', 'angle_to_goal', 'eventSec', 'coordinates_x', 'coordinates_y',
    'end_coordinates_x', 'end_coordinates_y', 'time_since_last_action',
    'possession_change', 'time_since_possession_change',
    'time_since_last_shot', 'prev_x', 'prev_y', 'distance_covered',
    'avg_team_x', 'avg_team_y', 'num_passes_last_5', 'num_dribbles_last_5',
    'score_difference_before_event', 'score_diff_from_focal_team',
    'delta_x'
]

binary_features = [
    'is_shot', 'is_pass', 'is_dribble', 'same_team_as_prev',
    'is_in_final_third', 'is_central_channel', 'is_defensive_player'
] + [col for col in data_exp2.columns if col.startswith("tag_")]
# %%
simple_features = categorical_features + numerical_features + binary_features

# Extended feature set (adds context-aware variables)
contextual_categorical = ['is_home_team', 'angle_category_quantile']
contextual_numerical = ['formation_complexity', 'substitution_load']
contextual_binary = ['is_fresh_substitute']

extended_features = (
    categorical_features + contextual_categorical +
    numerical_features + contextual_numerical +
    binary_features + contextual_binary
)

#%% chronological split

unique_gws = sorted(data_exp2['gameweek'].dropna().unique())
train_gws = unique_gws[:30]
test_gws = unique_gws[30:]

train_df_chrono = data_exp2[data_exp2['gameweek'].isin(train_gws)].copy()
test_df_chrono = data_exp2[data_exp2['gameweek'].isin(test_gws)].copy()

X_train_simple = train_df_chrono[simple_features].copy()
y_train_simple = train_df_chrono[target_col].copy()

X_test_simple = test_df_chrono[simple_features].copy()
y_test_simple = test_df_chrono[target_col].copy()
# %%
# 2. Random Stratified Split
# ------------------------------
X_all_simple = data_exp2[simple_features].copy()
y_all_simple = data_exp2[target_col].copy()

# Random stratified split (80/20)
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    X_all_simple, y_all_simple,
    test_size=0.2,
    stratify=y_all_simple,
    random_state=619
)

# %%
print("Chronological split:", X_train_simple.shape, X_test_simple.shape)
print("Random split:", X_train_rand.shape, X_test_rand.shape)

print("Chrono class balance:", y_train_simple.value_counts(normalize=True))
print("Random class balance:", y_train_rand.value_counts(normalize=True))
# %%
sns.histplot(data_exp2['time_since_last_shot'], bins=50)
# %%
missing_report = data_exp2.isna().sum().sort_values(ascending=False)

# Keep only variables with missing values
missing_report = missing_report[missing_report > 0]

# Display
print("🔍 Columns with missing values:\n")
print(missing_report)
# %%
# ---------------------------
# 7. Save data for Experiment 2
# ---------------------------
# %%
exp2_dir = r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\DataExp2"
os.makedirs(exp2_dir, exist_ok=True)

# Save random split data
joblib.dump(X_train_rand, os.path.join(exp2_dir, "X_train_rand.pkl"))
joblib.dump(X_test_rand, os.path.join(exp2_dir, "X_test_rand.pkl"))
joblib.dump(y_train_rand, os.path.join(exp2_dir, "y_train_rand.pkl"))
joblib.dump(y_test_rand, os.path.join(exp2_dir, "y_test_rand.pkl"))

# Save chronological split data (optional but recommended)
joblib.dump(X_train_simple, os.path.join(exp2_dir, "X_train_simple.pkl"))
joblib.dump(X_test_simple, os.path.join(exp2_dir, "X_test_simple.pkl"))
joblib.dump(y_train_simple, os.path.join(exp2_dir, "y_train_simple.pkl"))
joblib.dump(y_test_simple, os.path.join(exp2_dir, "y_test_simple.pkl"))

# Save feature lists
joblib.dump(categorical_features, os.path.join(exp2_dir, "categorical_features.pkl"))
joblib.dump(numerical_features, os.path.join(exp2_dir, "numerical_features.pkl"))
joblib.dump(binary_features, os.path.join(exp2_dir, "binary_features.pkl"))

# Save complete feature list used for modeling
simple_features = categorical_features + numerical_features + binary_features
joblib.dump(simple_features, os.path.join(exp2_dir, "simple_features.pkl"))

#%%
# ---------------------------
# 8. Run Experiment 2 on GCP
# This step is run on GCP on a vm instance with 64 gb 
# ---------------------------

# GCS paths
bucket_path = "gs://thesis-loic-data/DataExp2"
local_path = "/tmp/DataExp2"
OUTPUT_PATH = os.path.join(local_path, "results")
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(local_path, exist_ok=True)
os.system(f"gsutil -m cp {bucket_path}/*.pkl {local_path}/")

# Load data
X_train_rand = joblib.load(f"{local_path}/X_train_rand.pkl")
X_test_rand = joblib.load(f"{local_path}/X_test_rand.pkl")
y_train_rand = joblib.load(f"{local_path}/y_train_rand.pkl")
y_test_rand = joblib.load(f"{local_path}/y_test_rand.pkl")

X_train_simple = joblib.load(f"{local_path}/X_train_simple.pkl")
X_test_simple = joblib.load(f"{local_path}/X_test_simple.pkl")
y_train_simple = joblib.load(f"{local_path}/y_train_simple.pkl")
y_test_simple = joblib.load(f"{local_path}/y_test_simple.pkl")

for df in [X_train_rand, X_test_rand, X_train_simple, X_test_simple]:
    df.drop(columns=['last_shot_time'], inplace=True)

categorical_features = joblib.load(f"{local_path}/categorical_features.pkl")
binary_features = joblib.load(f"{local_path}/binary_features.pkl")
numerical_features = [
    'distance_to_goal', 'angle_to_goal', 'eventSec', 'coordinates_x', 'coordinates_y',
    'end_coordinates_x', 'end_coordinates_y', 'time_since_last_action',
    'possession_change', 'time_since_possession_change',
    'time_since_last_shot', 'prev_x', 'prev_y', 'distance_covered',
    'avg_team_x', 'avg_team_y', 'num_passes_last_5', 'num_dribbles_last_5',
    'score_difference_before_event', 'score_diff_from_focal_team',
    'delta_x'
]
simple_features = categorical_features + numerical_features + binary_features

splits = {
    "random": (X_train_rand, y_train_rand, X_test_rand, y_test_rand),
    "chrono": (X_train_simple, y_train_simple, X_test_simple, y_test_simple)
}
print("✅ files OK")

#%%
def log_step(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def save_model_to_gcs(model, model_name, bucket_path="gs://thesis-loic-data/Models/"):
    tmp = os.path.join(local_path, model_name)
    joblib.dump(model, tmp)
    gcs_dest = os.path.join(bucket_path, f"{model_name}.pkl")
    os.system(f"gsutil cp {tmp} {gcs_dest}")
    log_step(f"✅ Modèle {model_name} sauvegardé dans {gcs_dest}")

def save_csv_to_gcs(df, filename, bucket_path="gs://thesis-loic-data/Results/"):
    tmp = os.path.join(local_path, filename)
    df.to_csv(tmp, index=False)
    print(f"DEBUG:{tmp} exist? {os.path.exists(tmp)}") 
    gcs_dest = os.path.join(bucket_path, filename)
    print(f"DEBUG: moving {tmp} in {gcs_dest}")
    os.system(f"gsutil cp {tmp} {gcs_dest}")
    log_step(f"✅ Résultats CSV sauvegardés dans {gcs_dest}")

# Training function
def train_single_model(X_train, y_train, X_test, y_test,
                       model_name, model_proto, param_grid,
                       categorical_features, numerical_features, all_features,
                       split_type, imputer_name, imputer,
                       bucket_path, scoring,
                       seed_base=42,
                       repeat_idx=0,
                       verbose=False):
    try:
        start_time = time.time()
        random_seed = seed_base + repeat_idx
        np.random.seed(random_seed)

        log_step(f"➡️ Start | Split={split_type} | Model={model_name} | Imputer={imputer_name} | Seed={random_seed}")

        model = clone(model_proto)
        if hasattr(model, "random_state"):
            model.set_params(random_state=random_seed)

        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline([
                ('imputer', imputer),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', model)
        ])
        
        n_iter_by_model = {
        'LogisticRegression': 500,#30,
        'RandomForest': 500,#40,
        'XGBoost': 500,#60,
        'LightGBM': 500,#60,
        'CatBoost': 500,#40
        }

        nb_total_combis = 1
        for param, values in param_grid.items():
            nb_total_combis *= len(values)

    
        n_iter = min(n_iter_by_model.get(model_name, 30), nb_total_combis)

        log_step(f"🔧 {model_name} | n_iter={n_iter} / {nb_total_combis} combinaisons possibles")
        
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed_base + repeat_idx)

        search = HalvingRandomSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            #n_iter=n_iter,
            n_candidates=n_iter,
            min_resources=100,
            scoring=scoring,
            n_jobs=1,
            cv=cv,
            random_state=random_seed,
            refit='f1'
        )


        X_resampled, y_resampled = X_train, y_train


        log_step("🔍 RandomizedSearchCV tuning")
        fit_start = time.time()
        search.fit(X_resampled, y_resampled)
        fit_duration = time.time() - fit_start
        
        best_model = search.best_estimator_

        log_step(f"🕒 Fit time = {fit_duration:.2f} sec")
        log_step(f"🔍 Best params for {model_name}: {search.best_params_}")

        model_filename = f"{model_name}_{imputer_name}_{split_type}_{repeat_idx}_new"

        save_model_to_gcs(best_model, model_filename, bucket_path + "/Models_v4/")
      
        if model_name == 'LightGBM':
            try:
                # Classifieur seul
                lgbm_clf = best_model.named_steps['clf']
                lgbm_filename = model_filename + "_lgbm_only.pkl"
                local_lgbm_path = os.path.join("/tmp", lgbm_filename)
                joblib.dump(lgbm_clf, local_lgbm_path)
                gcs_path_lgbm = os.path.join(bucket_path + "/Models_v4/", lgbm_filename)
                os.system(f"gsutil cp {local_lgbm_path} {gcs_path_lgbm}")
                log_step(f" LightGBM classifier saved in : {gcs_path_lgbm}")

                # Booster natif
                booster = lgbm_clf.booster_
                booster_txt_filename = model_filename + "_booster.txt"
                local_booster_path = os.path.join("/tmp", booster_txt_filename)
                booster.save_model(local_booster_path)
                gcs_path_booster = os.path.join(bucket_path + "/Models_v4/", booster_txt_filename)
                os.system(f"gsutil cp {local_booster_path} {gcs_path_booster}")
                log_step(f"Native booster saved in: {gcs_path_booster}")

                # Preprocessor seul (optionnel mais recommandé pour SHAP)
                preprocessor = best_model.named_steps['preprocessor']
                prep_filename = model_filename + "_preprocessor.pkl"
                local_prep_path = os.path.join("/tmp", prep_filename)
                joblib.dump(preprocessor, local_prep_path)
                gcs_path_prep = os.path.join(bucket_path + "/Models_v4/", prep_filename)
                os.system(f"gsutil cp {local_prep_path} {gcs_path_prep}")
                log_step(f"🧪 Preprocessor saved in: {gcs_path_prep}")

            except Exception as e:
                log_step(f"⚠️ Error when saving LightGBM components : {str(e)}")

                
        

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
        
        # Saving predictions for PR/AUC curves
        try:
            # 1. Local save
            npz_filename = f"{model_name}_{imputer_name}_{split_type}_{repeat_idx}_proba_v2.npz"
            npz_path_local = os.path.join("/tmp/DataExp2", npz_filename)
            np.savez_compressed(npz_path_local, y_true=y_test, y_proba=y_proba)

            log_step(f"💾 file NPZ saved locally : {npz_path_local}")

            # 2. GCS Upload
            gcs_proba_dir = os.path.join(bucket_path, "ProbaData")
            gcs_dest_npz = os.path.join(gcs_proba_dir, npz_filename)

            upload_result = os.system(f"gsutil cp {npz_path_local} {gcs_dest_npz}")
            if upload_result == 0:
                log_step(f"NPZ file uploaded to GCS : {gcs_dest_npz}")
            else:
                log_step(f" GCS upload fails for : {npz_path_local}")

            os.remove(npz_path_local)

        except Exception as e:
            log_step(f"Error while saving/uploading NPZ : {str(e)}")
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calcul des métriques avant le return
        f1 = f1_score(y_test, y_pred, zero_division= 0)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_proba) if y_proba is not None else float('nan')


        log_step(f" F1={f1:.4f} | ROC_AUC={roc_auc:.4f} | Model={model_name}")
        
        log_step(f"end training model {model_name} | split={split_type} | seed={random_seed}")

        # Return final
        
        return {
        'model_id': f"{model_name}_{imputer_name}_{split_type}_{repeat_idx}",
        'model': model_name,
        'imputer': imputer_name,
        'split_type': split_type,
        'repeat_idx': repeat_idx,
        'seed': random_seed,
        'prob_name': npz_filename,

        # performance score
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'cv_f1_mean': search.best_score_,

        # 🧮 confusion matrix
        'confusion_tp': tp,
        'confusion_fp': fp,
        'confusion_fn': fn,
        'confusion_tn': tn,

        # training information
        'n_iter': n_iter,
        'fit_time_sec': fit_duration,
        'duration_sec': time.time() - start_time,
        'status': 'success',


        'n_estimators': search.best_params_.get('clf__n_estimators'),
        'max_depth': search.best_params_.get('clf__max_depth'),
        'learning_rate': search.best_params_.get('clf__learning_rate'),

        'best_params': search.best_params_,
        }

    except Exception as e:
        full_error = traceback.format_exc()
        log_step(f"❌ ERREUR | Model={model_name} | Imputer={imputer_name} | Split={split_type}")
        log_step(full_error)
        return {
            'split_type': split_type,
            'model': model_name,
            'imputer': imputer_name,
            'status': 'error',
            'error': str(e),
            'trace': full_error
        }

#%%
def train_all_parallel(
    splits,
    categorical_features, numerical_features, all_features,
    bucket_path="gs://thesis-loic-data",
    results_path="gs://thesis-loic-data/Results/results_exp2.csv",
    save_path="/tmp/DataExp2/results",
    n_jobs=12, 
    n_repeats=1,
    seed_base=1000,
    verbose=True,
    return_summary=False
):
    os.makedirs(save_path, exist_ok=True)
    f= open(os.devnull, 'w')
    sys.stderr = f
    log_step("START EVAL ALL PARAM")
    imputers = {
        'mean': SimpleImputer(strategy='mean'),
        'knn': KNNImputer(n_neighbors=5)
    }

    models = {
        'LogisticRegression': LogisticRegression(max_iter=3000, solver='liblinear'),
        'RandomForest': RandomForestClassifier(n_jobs=1),
        'XGBoost': XGBClassifier(eval_metric='logloss', objective='binary:logistic', n_jobs=1),
        'LightGBM': LGBMClassifier(objective='binary', n_jobs=1, verbosity=-1),
        'CatBoost': CatBoostClassifier(verbose=0, allow_writing_files=False)
    }
    log_step("Build param grid")
    param_grids = {
        'LogisticRegression': {
            'clf__penalty': ['l1', 'l2'],
            'clf__C': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
            'clf__solver': ['liblinear', 'saga']
        },
        'RandomForest': {
            'clf__n_estimators': [100, 300, 500],
            'clf__max_depth': [None, 10, 20, 40],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__max_features': ['sqrt', 'log2', None]
        },
        'XGBoost': {
            'clf__n_estimators': [100, 300, 500],
            'clf__max_depth': [3, 6, 10],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__subsample': [0.6, 0.8, 1],
            'clf__colsample_bytree': [0.6, 0.8, 1],
            'clf__reg_alpha': [0, 0.5, 1],
            'clf__reg_lambda': [0, 0.5, 1],
            'clf__min_child_weight': [1, 3, 5],
            'clf__gamma': [0, 1],
            'clf__max_delta_step': [0]
        },
        'LightGBM': {
            'clf__n_estimators': [100, 300, 500],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__max_depth': [-1, 6, 10],
            'clf__num_leaves': [31, 63, 127],
            'clf__min_data_in_leaf': [20, 50, 100],
            'clf__subsample': [0.6, 0.8, 1.0],
            'clf__colsample_bytree': [0.6, 0.8, 1.0],
            'clf__reg_alpha': [0, 0.1, 1.0],
            'clf__reg_lambda': [0, 0.1, 1.0]
        },
        'CatBoost': {
            'clf__iterations': [100],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__depth': [4, 6, 8],
            'clf__l2_leaf_reg': [1, 3, 5],
            'clf__border_count': [32, 64, 128],
            'clf__bagging_temperature': [0, 1, 5]
        }
    }
    
    log_step("build parallel tasks")
    tasks = []
    for split_type, (X_train, y_train, X_test, y_test) in splits.items():
        scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
        catboost_class_weights = {0: 1.0, 1: scale_pos_weight}

        for imputer_name, imputer in imputers.items():
            for model_name, model_proto in models.items():
                for repeat_idx in range(n_repeats):
                    random_seed = seed_base + repeat_idx
                    model = clone(model_proto)
                    if hasattr(model, "random_state"):
                        model.set_params(random_state=random_seed)
                    if model_name == 'XGBoost':
                        model.set_params(scale_pos_weight=scale_pos_weight)
                    elif model_name == 'LightGBM':
                        model.set_params(scale_pos_weight=scale_pos_weight)
                    elif model_name == 'CatBoost':
                        model.set_params(class_weights=catboost_class_weights)

                    tasks.append((
                        X_train, y_train, X_test, y_test,
                        model_name, model_proto, param_grids[model_name],
                        categorical_features, numerical_features, all_features,
                        split_type, imputer_name, imputer,
                        bucket_path, 'f1',
                        seed_base, repeat_idx, verbose
                    ))

    log_step(f"Launch {len(tasks)} tasks with {n_repeats} repetitions...")
    check_point_list = os.listdir(save_path)
    checkpoint_path = os.path.join(save_path,  f"{model_name}_{split_type}_{imputer_name}_{random_seed}_local_results_checkpoint.csv")
    if os.path.exists(checkpoint_path):
        df_checkpoint = pd.read_csv(checkpoint_path)
        done_ids = set(df_checkpoint["model_id"])
        log_step(f" Checkpoint detected: {checkpoint_path}")
        log_step(f"{len(done_ids)} trained models found")
    else:
        df_checkpoint = pd.DataFrame()
        done_ids = set()
        log_step("No checkpoints detected. All models will be trained.")

    filtered_tasks = []
    for args in tasks:
        model_name = args[4]
        imputer_name = args[12]
        split_type = args[11]
        repeat_idx = args[13]
        model_id = f"{model_name}_{imputer_name}_{split_type}_{repeat_idx}"
        if model_id not in done_ids:
            filtered_tasks.append(args)
        else:
            log_step(f" Skipped model : {model_id}")

    log_step(f"🔄 {len(filtered_tasks)} tasks remaining to be completed (on {len(tasks)} total)")

   
    
    def log_and_run(*args):
        
        split_type = args[11]
        model_name = args[4]
        imputer_name = args[12]
        random_seed = args[16]

        log_step(f"start of task: Split={split_type} | Model={model_name} | Imputer={imputer_name} | Seed={random_seed}")
        run_results_filename = f"{model_name}_{split_type}_{imputer_name}_{random_seed}_local_results_checkpoint.csv"

        result = train_single_model(*args)

        # Local incremental backup
        checkpoint_path = os.path.join(save_path, run_results_filename )
        write_header = not os.path.exists(checkpoint_path)
        pd.DataFrame([result]).to_csv(checkpoint_path, mode='a', index=False, header=write_header)

        # Saving the updated CSV on GCS
        try:
            save_csv_to_gcs(
                pd.read_csv(checkpoint_path),
                run_results_filename,
                bucket_path=os.path.dirname(results_path)
            )
        except Exception as e:
            log_step(f"Error uploading CSV GCS (non-blocking) : {str(e)}")

        print(f"[{datetime.now()}]  End of task: {result.get('model_id', 'unknown')} | status = {result.get('status')}", flush=True)
        return result


    overall_start = time.time()
    try:
        new_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(log_and_run)(*args) for args in filtered_tasks
        )
    except Exception as e:
        log_step(f" Error during Parallel : {e}")
        new_results = []

    # robust save
    try:
        df_new = pd.DataFrame(new_results)
        df_all = pd.concat([df_checkpoint, df_new], ignore_index=True)

        final_filename = os.path.basename(results_path)
        successes = df_all[df_all["status"] == "success"]
        errors = df_all[df_all["status"] == "error"]

        if not successes.empty:
            local_file = os.path.join(save_path, final_filename)
            log_step(f" {len(successes)} successfully models trained.")
        else:
            local_file = os.path.join(save_path, "local_results_debug.csv")
            log_step("No model successfully trained. Debug file generated.")

        df_all.to_csv(local_file, index=False)
        df_all.to_csv(checkpoint_path, index=False)

        log_step(f"Results saved locally in :{local_file}")
        log_step(f"Complete checkpoint in : {checkpoint_path}")

        if not successes.empty:
            save_csv_to_gcs(df_all, filename=final_filename, bucket_path=os.path.dirname(results_path))
            log_step("CSV results uploaded to GCS.")
        else:
            log_step("No data uploaded to GCS (no valid models).")

    except Exception as e:
        full_error = traceback.format_exc()
        log_step("ERROR when saving the final results.")
        log_step(str(e))
        log_step(full_error)

    log_step(f"⏱️ Total training time = {time.time() - overall_start:.2f} sec")

    if return_summary:
        return summarize_repeats(df_all)
    return df_all

#%%
# Run Experiment 2
df_results = train_all_parallel(
    splits=splits,
    categorical_features=categorical_features,
    numerical_features=numerical_features,
    all_features=simple_features,
    bucket_path="gs://thesis-loic-data",  
    results_path="gs://thesis-loic-data/Results/results_exp2.csv", 
    save_path=OUTPUT_PATH,           
    n_jobs=24,
    n_repeats=3,                           
    seed_base=1000,
    verbose=True,
    return_summary=False
)
#%%
# ---------------------------
# 9. Analyze Experiement 2 
# part shap and features importance 
# on gcp 
# ---------------------------
# issue for save correctly the information with lightgbm so we need to rebuild manually
booster = lgb.Booster(model_file=local_booster_path)
print("Booster loaded from GCS bucket (via /tmp)")

#%%
booster_feature_names = booster.feature_name()
importances = booster.feature_importance(importance_type="gain")
#%%
# Recreate the preprocessor with exactly the right parameters
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ]), numerical_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])
# Re-fit on X_train
X_train = X_train_simple.copy()
preprocessor.fit(X_train)

# Get the names of transformed features
true_feature_names = preprocessor.get_feature_names_out()

#%%
# Quick visualization
importance_df = pd.DataFrame({
    "Feature": true_feature_names,
    "Importance": booster.feature_importance(importance_type="gain")
}).sort_values("Importance", ascending=False)

print(importance_df.head(10))

#%%
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"][:20], importance_df["Importance"][:20])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances (Gain)")
plt.tight_layout()

# ✅ save plot
plt.savefig("/tmp/feature_importances_top20.png", dpi=300, bbox_inches="tight")
print("✅ plot saved in /tmp/feature_importances_top20.png")

plt.show()

#%%
# Shap
X_transformed = preprocessor.fit_transform(X_train)
feature_names = preprocessor.get_feature_names_out()
#%%
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X_transformed)
print("SHAP values calculated successfully.")

#%%
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
plt.savefig("/tmp/shap_summary_plot.png", dpi=300, bbox_inches="tight")
print("SHAP graph saved in /tmp/shap_summary_plot.png")

#%%
# ---------------------------
# 10. Analyze Experiement 2 
# return on local machine
# ---------------------------
#%%
plots_exp2_dir = r"C:\Users\loicm\Documents\2024-2025\MasterThesis\plots\Experiment_2"
#%%
# local path to NPZ files
proba_dir = r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\DataExp2\ProbModels"

npz_files = [f for f in os.listdir(proba_dir) if f.endswith('.npz')]

all_proba_data = []

for filename in npz_files:
    path = os.path.join(proba_dir, filename)
    with np.load(path) as data:
        y_true = data['y_true']
        y_proba = data['y_proba']
        all_proba_data.append({
            'filename': filename,
            'y_true': y_true,
            'y_proba': y_proba
        })

print(f"✅ {len(all_proba_data)} files NPZ loaded.")
# %%
models_dir_exp2 = r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\DataExp2\DataModels"
model_files = [f for f in os.listdir(models_dir_exp2) if f.endswith('.pkl')]

all_models = {}

for filename in model_files:
    path = os.path.join(models_dir_exp2, filename)
    try:
        model = joblib.load(path)
        all_models[filename] = model
    except Exception as e:
        print(f"❌ Loading error {filename}: {e}")

print(f"✅ {len(all_models)} successfully loaded models.")

# %%
results_csv_path_exp2 = r"C:\Users\loicm\Documents\2024-2025\MasterThesis\ResultsData\ResultsDataExp2\results_exp2.csv"

df_results = pd.read_csv(results_csv_path_exp2)
print(f"✅ CSV loaded with {df_results.shape[0]} rows and {df_results.shape[1]} columns.")
# %%
# RQ2
## Step 1 Descriptive statistics by model
grouped = df_results.groupby('model').agg({
    'f1': ['mean', 'std'],
    'precision': ['mean', 'std'],
    'recall': ['mean', 'std'],
    'roc_auc': ['mean', 'std'],
    'pr_auc': ['mean', 'std'],
    'accuracy': ['mean', 'std'],
    'fit_time_sec': ['mean'],

    'cv_f1_mean': ['mean']
}).round(4)
# %%
grouped.columns = ['_'.join(col) for col in grouped.columns]
grouped = grouped.reset_index()

#%%
print("Comparison fo models :")
print(grouped)  
#%%
## Step 2
### PR-AUC aggregation by model
pr_auc_summary = df_results.groupby("model").agg(
    pr_auc_mean=("pr_auc", "mean"),
    pr_auc_std=("pr_auc", "std")
).round(4)

#%%
print("Comparison of PR-AUC models:")
display(pr_auc_summary)
## Step 3 Kruskal-Wallison PR-AUC
### shapiro test (normality of pr auc)
#%%
for model in df_results['model'].unique():
    stat, p = shapiro(df_results[df_results['model'] == model]['pr_auc'].dropna())
    print(f"Shapiro-Wilk pour {model}: W = {stat:.3f}, p-value = {p:.4f}")

#%%
pr_auc_groups = [df_results[df_results['model'] == model]['pr_auc'].dropna() for model in df_results['model'].unique()]
### Kruskal-Wallis test
kw_stat, kw_pval = kruskal(*pr_auc_groups)
print("Kruskal-Wallis sur PR-AUC")
print(f"H = {kw_stat:.4f}, p-value = {kw_pval:.4f}")

#%%
###  Dunn with Holm correction test
df_dunn = df_results[["model", "pr_auc"]].dropna()
dunn_result = sp.posthoc_dunn(
    df_dunn, 
    val_col='pr_auc', 
    group_col='model', 
    p_adjust='holm'
)
#%%
print("Dunn's post-hoc test (adjusted p-values)")
print(dunn_result.round(4))
#%%
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".4f", cmap="coolwarm", cbar=True,
            square=True, linewidths=0.5)
plt.title("Dunn post-hoc test (PR-AUC per model) - adjusted p-values (Holm)")
plt.tight_layout()
plt.show()

#%%
# Step 1 – Extract all scores per seed
pr_auc_results = []
roc_auc_results = []

for item in all_proba_data:
    filename = item["filename"].replace("_proba.npz", "")
    y_true = item["y_true"]
    y_proba = item["y_proba"]

    try:
        pr_auc = average_precision_score(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        pr_auc_results.append({"model_id": filename, "pr_auc": pr_auc})
        roc_auc_results.append({"model_id": filename, "roc_auc": roc_auc})
    except Exception as e:
        print(f"⚠️ Skipped {filename} due to error: {e}")

# Step 2 – Convert to DataFrames
df_pr = pd.DataFrame(pr_auc_results)
df_roc = pd.DataFrame(roc_auc_results)

# Step 3 – Merge PR and ROC scores per model_id
df_detailed = pd.merge(df_pr, df_roc, on="model_id", suffixes=("_pr_auc", "_roc_auc"))

# Step 4 – Extract base_model (remove seed)
df_detailed["base_model"] = df_detailed["model_id"].apply(lambda x: "_".join(x.split("_")[:3]))

# Step 5 – Aggregate by base_model
df_summary = df_detailed.groupby("base_model").agg({
    "pr_auc": ["mean", "std"],
    "roc_auc": ["mean", "std"]
}).round(4).reset_index()

# Step 6 – Rename columns
df_summary.columns = ["base_model", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "std_roc_auc"]

# Step 7 – Visualize or export
print(df_summary)
#%%

####option1
# Boxplot of PR-AUC for each model (3 seeds)
order = (
    df_detailed.groupby("base_model")["pr_auc"]
    .mean()
    .sort_values()
    .index
)

plt.figure(figsize=(22, 10)) 
sns.boxplot(data=df_detailed, x="base_model", y="pr_auc", order=order, width=0.6)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title("Distribution of PR-AUC across Seeds per Model Configuration", fontsize=14)
plt.xlabel("Model Configuration", fontsize=12)
plt.ylabel("PR-AUC", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("boxplot_pr_auc_by_model_wide.png", dpi=300)
plt.show()

#%%
#### option2
plt.figure(figsize=(12, 20))

# Tri optionnel par moyenne décroissante
order = df_detailed.groupby("base_model")["pr_auc"].mean().sort_values(ascending=False).index

sns.boxplot(
    data=df_detailed,
    y="base_model",
    x="pr_auc",
    order=order,
    width=0.6
)

plt.yticks(fontsize=10)
plt.title("Distribution of PR-AUC across Seeds per Model Configuration", fontsize=14)
plt.xlabel("PR-AUC", fontsize=12)
plt.ylabel("Model Configuration", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("boxplot_pr_auc_horizontal.png", dpi=300)
plt.show()
#%%
def plot_grouped_pr_curves_by_model(proba_data, save_dir="figures/exp2/pr_curves_grouped"):
    """
    Generate one PR curve per model configuration (model + split + imputer),
    with one curve per seed in the same plot.

    Parameters:
        proba_data (list): list of dicts with keys 'filename', 'y_true', 'y_proba'
        save_dir (str): path to save PR plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Group curves by model configuration (excluding seed)
    grouped = defaultdict(list)

    for item in proba_data:
        filename = item['filename'].replace("_proba.npz", "")
        
        # Extract base ID without seed (e.g., LightGBM_knn_random)
        parts = filename.split("_")
        model_base = "_".join(parts[:3])  # model, imputer, split
        seed = parts[3] if len(parts) > 3 else "0"

        grouped[model_base].append({
            "seed": seed,
            "y_true": item["y_true"],
            "y_proba": item["y_proba"],
            "full_id": filename
        })

    # Plot one figure per group
    for model_base, entries in grouped.items():
        plt.figure(figsize=(7, 5))

        for entry in entries:
            precision, recall, _ = precision_recall_curve(entry["y_true"], entry["y_proba"])
            pr_auc = average_precision_score(entry["y_true"], entry["y_proba"])
            plt.plot(recall, precision, label=f"Seed {entry['seed']} (AUC={pr_auc:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve\n{model_base}")
        plt.grid(True)
        plt.legend()

        save_path = os.path.join(save_dir, f"{model_base}_grouped.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved grouped PR curve: {save_path}")
#%%
plot_grouped_pr_curves_by_model(all_proba_data,
                                save_dir=r"C:\Users\loicm\Documents\2024-2025\MasterThesis\plots\Experiment_2\Precision_recall_curve_models")
## Etape 4 Comparaison random vs chrono split
#%%
#%%
df3 = df_detailed.copy()
df3['split'] = df3['base_model'].apply(lambda x: 'random' if 'random' in x else 'chrono')

metrics = ['pr_auc', 'roc_auc']
results = []

#%%
for metric in ['pr_auc', 'roc_auc']:
    chrono = df3[df3['split'] == 'chrono'][metric]
    random = df3[df3['split'] == 'random'][metric]

    stat_chrono, p_chrono = shapiro(chrono)
    stat_random, p_random = shapiro(random)

    print(f"{metric.upper()} - Shapiro Chrono: W={stat_chrono:.3f}, p={p_chrono:.4f}")
    print(f"{metric.upper()} - Shapiro Random: W={stat_random:.3f}, p={p_random:.4f}")


#%%
metrics = ['pr_auc', 'roc_auc']

for metric in metrics:
    for split in ['chrono', 'random']:
        values = df_detailed[df_detailed['base_model'].str.contains(split)][metric]

        plt.figure(figsize=(12, 5))

        # Histogram + KDE
        plt.subplot(1, 2, 1)
        sns.histplot(values, kde=True, bins=10, color='skyblue', edgecolor='black')
        plt.title(f"{metric.upper()} - {split} split\nHistogram with KDE")
        plt.xlabel(metric.upper())
        plt.ylabel("Frequency")
        plt.grid(True)

        # QQ-plot
        plt.subplot(1, 2, 2)
        stats.probplot(values, dist="norm", plot=plt)
        plt.title(f"{metric.upper()} - {split} split\nQQ-plot (Normality Check)")

        plt.tight_layout()
        plt.savefig(f"plots/Experiment_2/{metric}_{split}_distribution.png", dpi=300)
        plt.show()

#%%
metrics = ['pr_auc', 'roc_auc']
results = []

for metric in metrics:
    chrono = df3[df3['split'] == 'chrono'][metric]
    random = df3[df3['split'] == 'random'][metric]

    # Mann-Whitney U test
    stat, p = mannwhitneyu(chrono, random, alternative='two-sided')

    # Statistiques descriptives : median et IQR
    chrono_median = chrono.median()
    chrono_q1 = chrono.quantile(0.25)
    chrono_q3 = chrono.quantile(0.75)

    random_median = random.median()
    random_q1 = random.quantile(0.25)
    random_q3 = random.quantile(0.75)

    results.append({
        'Metric': metric.upper(),
        'Chrono Median [IQR]': f"{chrono_median:.4f} [{chrono_q1:.4f}–{chrono_q3:.4f}]",
        'Random Median [IQR]': f"{random_median:.4f} [{random_q1:.4f}–{random_q3:.4f}]",
        'Mann-Whitney U': round(stat, 1),
        'p-value': f"{p:.4f}",
        'Significant': p < 0.05
    })

df_split_comparison = pd.DataFrame(results)
print(df_split_comparison)
#%%
# Split type
df_detailed['split'] = df_detailed['base_model'].apply(lambda x: 'random' if 'random' in x else 'chrono')

# Metric
metric = 'roc_auc' # or roc_auc
ylabel = metric.upper()

# Summary
summary_split = (
    df_detailed.groupby('split')[metric]
    .agg(['mean', 'std'])
    .reset_index()
    .sort_values('split')  # Forcing 'chrono' before 'random'
)
print(summary_split)
#%%

# Plot
plt.figure(figsize=(6, 4))
barplot = sns.barplot(
    data=summary_split,
    x='split', y='mean',
    yerr=summary_split['std'],
    palette='pastel',
    errorbar=None 
)

for i, row in summary_split.iterrows():
    plt.text(
        i,
        row['mean'] + 0.001,  # Just above the bar
        f"{row['mean']:.3f} ± {row['std']:.3f}",
        ha='center',
        va='bottom',
        fontsize=9,
        weight='bold'
    )

plt.title(f"{ylabel} by Split Strategy", fontsize=13)
plt.ylabel(ylabel)
plt.xlabel("Split Type")
plt.ylim(summary_split['mean'].min() - 0.005, summary_split['mean'].max() + 0.01)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
#%%
##  Etape 5
df_detailed['imputer'] = df_detailed['base_model'].apply(
    lambda x: 'knn' if 'knn' in x else 'mean'
)
# %%
metrics = ['pr_auc', 'roc_auc']
for metric in metrics:
    for imp in ['mean', 'knn']:
        values = df_detailed[df_detailed['imputer'] == imp][metric]
        stat, p = shapiro(values)
        print(f"{metric.upper()} - {imp} imputer: W={stat:.3f}, p={p:.4f}")
# %%
metric = "pr_auc" # or roc_auc
for imp in ['mean', 'knn']:
    subset = df_detailed[df_detailed['imputer'] == imp][metric]
    
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(subset, kde=True, bins=10, color='skyblue')
    plt.title(f"{metric.upper()} - {imp} imputer\nHistogram with KDE")
    
    # QQ-plot
    plt.subplot(1, 2, 2)
    stats.probplot(subset, dist="norm", plot=plt)
    plt.title(f"{metric.upper()} - {imp} imputer\nQQ-plot (Normality Check)")

    plt.tight_layout()
    plt.show()
# %%
df4 = df_detailed.copy()
df4['imputer'] = df4['base_model'].apply(lambda x: 'knn' if 'knn' in x else 'mean')

metrics = ['pr_auc', 'roc_auc']
results = []

for metric in metrics:
    mean_vals = df4[df4['imputer'] == 'mean'][metric]
    knn_vals = df4[df4['imputer'] == 'knn'][metric]

    # Shapiro déjà fait
    u_stat, p_val = mannwhitneyu(mean_vals, knn_vals, alternative='two-sided')

    results.append({
        'Metric': metric.upper(),
        'Mean Imputer Mean ± Std': f"{mean_vals.mean():.4f} ± {mean_vals.std():.4f}",
        'KNN Imputer Mean ± Std': f"{knn_vals.mean():.4f} ± {knn_vals.std():.4f}",
        'Mann-Whitney U': f"{u_stat:.1f}",
        'p-value': f"{p_val:.4f}",
        'Significant': p_val < 0.05
    })

df_comparison = pd.DataFrame(results)
print(df_comparison)


#%%
df_detailed['imputer'] = df_detailed['base_model'].apply(lambda x: 'knn' if 'knn' in x else 'mean')

for metric in ['pr_auc', 'roc_auc']:
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df_detailed, x='imputer', y=metric, ci='sd', palette='pastel')

    
    for i, imputer_type in enumerate(df_detailed['imputer'].unique()):
        subset = df_detailed[df_detailed['imputer'] == imputer_type]
        mean = subset[metric].mean()
        std = subset[metric].std()
        bar = ax.patches[i]
        ax.annotate(f'{mean:.4f} ± {std:.4f}',
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    mean_overall = df_detailed[metric].mean()
    std_overall = df_detailed[metric].std()
    plt.ylim(mean_overall - 2 * std_overall, mean_overall + 2 * std_overall)

    plt.title(f"{metric.upper()} by Imputation Strategy")
    plt.ylabel(metric.upper())
    plt.xlabel("Imputer Type")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"barplot_{metric}_by_imputer_zoomed.png", dpi=300)
    plt.show()

#%%
df_detailed['imputer'] = df_detailed['base_model'].apply(lambda x: 'knn' if 'knn' in x else 'mean')

for metric in ['pr_auc', 'roc_auc']:
    plt.figure(figsize=(6, 4))
    # On utilise ci=None pour désactiver les erreurs standards car on va afficher l'IQR manuellement
    ax = sns.barplot(data=df_detailed, x='imputer', y=metric, ci=None, palette='pastel', estimator=np.median)

    # Annotation
    for i, imputer_type in enumerate(df_detailed['imputer'].unique()):
        subset = df_detailed[df_detailed['imputer'] == imputer_type][metric]
        median = subset.median()
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        iqr = q3 - q1
        bar = ax.patches[i]
        ax.annotate(f'{median:.4f} ± {iqr/2:.4f}',
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Ajustement de l'axe y avec la médiane et IQR
    median_overall = df_detailed[metric].median()
    iqr_overall = df_detailed[metric].quantile(0.75) - df_detailed[metric].quantile(0.25)
    plt.ylim(median_overall - 2 * iqr_overall, median_overall + 2 * iqr_overall)

    plt.title(f"{metric.upper()} by Imputation Strategy (Median ± IQR/2)")
    plt.ylabel(metric.upper())
    plt.xlabel("Imputer Type")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"barplot_{metric}_by_imputer_median.png", dpi=300)
    plt.show()



### Etape 6 Time and performance
#%%
df_model_summary = grouped.copy()

rho, pval = spearmanr(df_model_summary['f1_mean'], df_model_summary['fit_time_sec_mean'])

print(f"Spearman correlation (F1 vs Time): ρ = {rho:.3f}, p = {pval:.4f}")

#%%

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_model_summary,
    x="fit_time_sec_mean",
    y="f1_mean",
    hue="model",
    s=150,
    palette="Set2",
    edgecolor="black"
)

sns.regplot(
    data=df_model_summary,
    x="fit_time_sec_mean",
    y="f1_mean",
    scatter=False,
    ci=None,
    line_kws={"color": "gray", "linestyle": "--"}
)

for _, row in df_model_summary.iterrows():
    plt.text(row["fit_time_sec_mean"] + 30,
             row["f1_mean"] + 0.003,
             row["model"],
             fontsize=9,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5))

plt.title(f"Training Time vs F1 Score\n(Spearman ρ = {rho:.2f}, p = {pval:.4f})",
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Training Time (seconds)")
plt.ylabel("F1 Score")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(plots_exp2_dir + "/scatter_fit_time_vs_f1_enhanced.png")
plt.show()

#%% 
# PR-AUC vs Fit Time
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_model_summary,
    x="fit_time_sec_mean",
    y="pr_auc_mean",
    hue="model",
    s=150,
    palette="Set2",
    edgecolor="black"
)

# Ligne de tendance
sns.regplot(
    data=df_model_summary,
    x="fit_time_sec_mean",
    y="pr_auc_mean",
    scatter=False,
    ci=None,
    line_kws={"color": "gray", "linestyle": "--"}
)

# Annotations
for _, row in df_model_summary.iterrows():
    plt.text(row["fit_time_sec_mean"] + 30,
             row["pr_auc_mean"] + 0.002,
             row["model"],
             fontsize=9,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5))

# Calcul de Spearman
rho_pr, pval_pr = spearmanr(df_model_summary["pr_auc_mean"], df_model_summary["fit_time_sec_mean"])

plt.title(f"Training Time vs PR-AUC\n(Spearman ρ = {rho_pr:.2f}, p = {pval_pr:.4f})",
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Training Time (seconds)")
plt.ylabel("PR-AUC Score")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(plots_exp2_dir + "/scatter_fit_time_vs_pr_auc_enhanced.png")
plt.show()


#%%
#x_train_path = r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\DataExp2\RawDataExp2\X_train_rand.pkl"
X_train_array = joblib.load(r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\DataExp2\RawDataExp2\X_train_simple.pkl")
X_train = pd.DataFrame(X_train_array, columns=all_features)

#%%
x_test_path = r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\DataExp2\RawDataExp2\X_test_simple.pkl"
X_test_array = joblib.load(x_test_path)
#%%
X_test = pd.DataFrame(X_test_array, columns=all_features)
#%%
print(X_train.shape)
#%%
X_test.shape
#%%
prep_dir = r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\PrepData"
data_exp2 = pd.read_pickle(os.path.join(prep_dir, "data_bft.pkl"))

#%%
X_test_chrono_full = data_exp2.loc[X_test.index].copy()

#%%
X_test_chrono_full['y_proba'] = y_proba  
X_test_chrono_full['y_pred'] = y_pred  

#%%
X_test_chrono_full['y_proba'].value_counts()

#%%
assert all(X_test_chrono_full.index == X_test.index)

#%%
X_test_chrono_full['y_proba'].describe()
#%%
# === 3. Rebuild X_test DataFrame ===
X_test = pd.DataFrame(X_test_array, columns=feature_names)

#%%
X_test.drop(columns=['last_shot_time'], inplace=True, errors='ignore')

#%%
y_test = joblib.load(r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\DataExp2\RawDataExp2\y_test_simple.pkl")

#%%

X_test_array = joblib.load(r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\DataExp2\RawDataExp2\X_test_simple.pkl")
features = joblib.load(r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\DataExp2\RawDataExp2\simple_features.pkl")
X_test = pd.DataFrame(X_test_array, columns=features)

#%%
X_test.drop(columns=['last_shot_time'], inplace=True, errors='ignore')

#%%
df_diag_lgbm = X_test.copy()
df_diag_lgbm["y_proba"] = y_proba
df_diag_lgbm["y_true"] = y_true

#%%
grouped_cm = df_results.groupby('model')[['confusion_tp', 'confusion_fp', 'confusion_fn', 'confusion_tn']].mean().round(1)

# Standardisation by class (True Positives + False Negatives etc.)
grouped_cm['TPR'] = grouped_cm['confusion_tp'] / (grouped_cm['confusion_tp'] + grouped_cm['confusion_fn'])
grouped_cm['TNR'] = grouped_cm['confusion_tn'] / (grouped_cm['confusion_tn'] + grouped_cm['confusion_fp'])

print(grouped_cm[['TPR', 'TNR']])


#%%
perf_std = df_results.groupby('model')[['f1', 'pr_auc', 'roc_auc']].std().round(4)
print(perf_std)

#%%
y_pred = (y_proba >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No contrib", "Contrib"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Blues", values_format=".2f")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png", dpi=300)
plt.show()

#%%
plt.figure(figsize=(9, 5))

sns.histplot(df_diag_lgbm[df_diag_lgbm.y_true == 1]["y_proba"],
             label="Positive", kde=True, stat="density", bins=50, color="#1f77b4", alpha=0.5)

sns.histplot(df_diag_lgbm[df_diag_lgbm.y_true == 0]["y_proba"],
             label="Negative", kde=True, stat="density", bins=50, color="#ff7f0e", alpha=0.5)

ks_stat, p_val = ks_2samp(
    df_diag_lgbm[df_diag_lgbm.y_true == 1]["y_proba"],
    df_diag_lgbm[df_diag_lgbm.y_true == 0]["y_proba"]
)

plt.text(0.6, plt.ylim()[1]*0.8, f"KS = {ks_stat:.3f}", fontsize=12,
         bbox=dict(facecolor='white', edgecolor='gray'))

plt.title("Predicted Probability Distributions by True Class", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Probability", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("hist_y_proba_by_class_improved.png", dpi=300)
plt.show()

#%%
plt.figure(figsize=(12, 5))
sns.boxplot(data=X_test_chrono_full, x="eventId", y="y_proba")
plt.xticks(rotation=90)
plt.title("Distribution of predicted probabilities by event type")
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(12, 5))
sns.boxplot(data=X_test_chrono_full, x="subEventId", y="y_proba")
plt.xticks(rotation=45)
plt.title("Distribution of predicted probabilities by high-level subevent type")
plt.tight_layout()
plt.show()

#%%
from sklearn.impute import SimpleImputer
X_encoded = pd.get_dummies(X_test, columns=['role_code', 'team_game_state'], drop_first=True)
X_imputed = SimpleImputer(strategy="mean").fit_transform(X_encoded)
X_scaled = StandardScaler().fit_transform(X_imputed)


#%%
print(X_encoded.isnull().sum().sort_values(ascending=False).head(10))
#%%

# PCA
X_pca = PCA(n_components=2, random_state=89).fit_transform(X_scaled)

#%%
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=X_test_chrono_full["y_proba"], cmap='viridis', alpha=0.5)
plt.colorbar(label='Predicted probability')
plt.title("PCA projection colored by predicted probability")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()

#%%
# DBSCAN clustering
clusters = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_pca)

X_test_chrono_full['cluster'] = clusters
X_test_chrono_full['pca_1'] = X_pca[:, 0]
X_test_chrono_full['pca_2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=X_test_chrono_full, x='pca_1', y='pca_2', hue='cluster', palette='tab10')
plt.title("DBSCAN Clustering on PCA-reduced features")
plt.tight_layout()
plt.show()

#%%
cluster_0 = X_test_chrono_full[X_test_chrono_full['cluster'] == 0]
others = X_test_chrono_full[X_test_chrono_full['cluster'] != 0]

#%%
numeric_cols = X_test_chrono_full.select_dtypes(include=['number']).columns.drop(['cluster', 'pca_1', 'pca_2'])

results = []

for feature in numeric_cols:
    data0 = cluster_0[feature]
    data1 = others[feature]

    if pd.api.types.is_numeric_dtype(data0) and pd.api.types.is_numeric_dtype(data1):
        try:
            median_0 = data0.median()
            q1_0 = data0.quantile(0.25)
            q3_0 = data0.quantile(0.75)

            median_1 = data1.median()
            q1_1 = data1.quantile(0.25)
            q3_1 = data1.quantile(0.75)

            u_stat, p_val = mannwhitneyu(data0, data1, alternative='two-sided')

            results.append({
                'Feature': feature,
                'Cluster 0 Median [IQR]': f"{median_0:.4f} [{q1_0:.4f}–{q3_0:.4f}]",
                'Others Median [IQR]': f"{median_1:.4f} [{q1_1:.4f}–{q3_1:.4f}]",
                'Mann–Whitney U': round(u_stat, 2),
                'p-value': round(p_val, 4),
                'Significant': p_val < 0.05
            })
        except Exception as e:
            print(f"Skipped feature {feature} due to error: {e}")

df_cluster_comparison = pd.DataFrame(results)

#%%
pd.set_option("display.max_rows", None)  
print(df_cluster_comparison)

#%%
significant_features = df_cluster_comparison[df_cluster_comparison["Significant"] == True]
print(significant_features[['Feature', 'Cluster 0 Median [IQR]', 'Others Median [IQR]', 'p-value']])

#%%
for feature in significant_features['Feature']:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(cluster_0[feature], label='Cluster 0', fill=True)
    sns.kdeplot(others[feature], label='Others', fill=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.legend()
    plt.tight_layout()
    plt.show()


#%%
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
labels = clusterer.fit_predict(X_pca)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
plt.title("Clustering with HDBSCAN")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()

#%%
X_test_chrono_full['hdbscan_cluster'] = labels

#%%
X_test_chrono_full['hdbscan_cluster'].value_counts()

#%%
# Percentage of events per cluster
event_dist = X_test_chrono_full.groupby('hdbscan_cluster')['eventName'].value_counts(normalize=True).unstack().fillna(0)
event_dist.head()  

#%%
plt.figure(figsize=(12, 6))
sns.heatmap(event_dist.T, cmap='Blues', annot=True, fmt=".2f")
plt.title("Breakdown of event types by HDBSCAN cluster")
plt.xlabel("Cluster")
plt.ylabel("Event Type")
plt.tight_layout()
plt.show()

#%%
def summarize_cluster(df, cluster_id):
    sub_df = df[df['hdbscan_cluster'] == cluster_id]
    return {
        'cluster': cluster_id,
        'n': len(sub_df),
        'mean_proba': sub_df['y_proba'].mean(),
        'top_event': sub_df['eventName'].value_counts().idxmax()
    }

summaries_cluster = [summarize_cluster(X_test_chrono_full, c) for c in sorted(X_test_chrono_full['hdbscan_cluster'].unique())]
pd.DataFrame(summaries_cluster)

#%%
sns.scatterplot(data=X_test_chrono_full[X_test_chrono_full['eventName'] == 'Shot'],
                x='coordinates_x', y='coordinates_y', hue='hdbscan_cluster', palette='tab10')
plt.title("Position of shots per HDBSCAN cluster")
plt.gca().invert_yaxis()
plt.show()

#%%
cluster_3 = X_test_chrono_full[X_test_chrono_full['hdbscan_cluster'] == 3]
cluster_3[['coordinates_x', 'coordinates_y', 'y_proba']].describe()

#%%
X_test_chrono_full[X_test_chrono_full['hdbscan_cluster'] == 3]['subEventName'].value_counts()

#%%
X_test_chrono_full[X_test_chrono_full['hdbscan_cluster'] == 1][['coordinates_x', 'coordinates_y', 'y_proba']].describe()


#%%
features_to_inspect = ['coordinates_x', 'coordinates_y', 'y_proba', 'is_shot', 'is_pass', 'is_quality_shot']

X_test_chrono_full.groupby('hdbscan_cluster')[features_to_inspect].describe().transpose()

#%%
def summarize_cluster(df, cid):
    sub_df = df[df['hdbscan_cluster'] == cid]
    return {
        'Cluster': cid,
        'n': len(sub_df),
        'mean_proba': sub_df['y_proba'].mean(),
        'main_event': sub_df['eventName'].value_counts().idxmax(),
        'x_mean': sub_df['coordinates_x'].mean(),
        'y_mean': sub_df['coordinates_y'].mean()
    }

summary_table_clusters = pd.DataFrame([summarize_cluster(X_test_chrono_full, cid) 
                              for cid in sorted(X_test_chrono_full['hdbscan_cluster'].unique())])
print(summary_table_clusters)

#%%
non_numeric_cols = X_test.select_dtypes(include='object').columns
print(non_numeric_cols)
#%%
freq = data_exp2["subEventId"].value_counts(normalize=True).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=freq.index, y=freq.values)
plt.xticks(rotation=90)
plt.title("Relative Frequency of subEventId types")
plt.xlabel("subEventId")
plt.ylabel("Proportion")
plt.tight_layout()
plt.show()


#%%
freq = data_exp2["eventId"].value_counts(normalize=True).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=freq.index, y=freq.values)
plt.xticks(rotation=90)
plt.title("Relative Frequency of eventId types")
plt.xlabel("eventId")
plt.ylabel("Proportion")
plt.tight_layout()
plt.show()

#%%
print("Class distribution (chrono split):")
print(pd.Series(y_test).value_counts(normalize=True))

#%%
#x_train_path = r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\DataExp2\RawDataExp2\X_train_rand.pkl"
X_train_array = joblib.load(r"C:\Users\loicm\Documents\2024-2025\MasterThesis\Data\DataExp2\RawDataExp2\X_train_simple.pkl")
X_train = pd.DataFrame(X_train_array, columns=features)

#%%
X_train.drop(columns=['last_shot_time'], inplace=True, errors='ignore')
#%%

missing_per_column = X_train.isnull().sum()
missing_percentage_per_column = 100 * missing_per_column / len(X_train)
missing_summary = pd.DataFrame({
    "Missing Values": missing_per_column,
    "Missing (%)": missing_percentage_per_column
})
missing_summary = missing_summary[missing_summary["Missing Values"] > 0].sort_values("Missing (%)", ascending=False)

#%%
print(missing_summary)


#%%
total_cells = X_train.size
total_missing = X_train.isnull().sum().sum()
global_missing_rate = 100 * total_missing / total_cells

print(f"🔍 Global missing rate: {global_missing_rate:.2f}% ({total_missing} / {total_cells} cells)")

#%%

plt.figure(figsize=(12, 6))
sns.heatmap(X_train.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Heatmap of Missing Values in X_train")
plt.tight_layout()
plt.show()

#%%
data_exp2 = data_exp2.drop(columns=['last_shot_time','steps_to_shot'], errors='ignore')
#%%

missing_counts = data_exp2.isnull().sum()
missing_pct = data_exp2.isnull().mean() * 100

missing_summary = pd.DataFrame({
    "Missing Values": missing_counts,
    "Missing (%)": missing_pct.round(4)
}).sort_values("Missing Values", ascending=False)

print(missing_summary[missing_summary["Missing Values"] > 0])


#%%
total_cells = data_exp2.shape[0] * data_exp2.shape[1]
total_missing = data_exp2.isnull().sum().sum()
global_missing_rate = total_missing / total_cells * 100

print(f"🔍 Global missing rate: {global_missing_rate:.6f}% ({total_missing} / {total_cells} cells)")

#%%
plt.figure(figsize=(18, 6))
sns.heatmap(data_exp2.isnull(), cbar=False, cmap="viridis")
plt.title("Heatmap of Missing Values in data_exp2")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.tight_layout()
plt.show()


