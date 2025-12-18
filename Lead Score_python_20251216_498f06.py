# ML Project Rental.py
# Corrected version - Fixed multiple issues

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
import xgboost as xgb
import smtplib
from email.message import EmailMessage
import requests
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, sin, cos, asin, sqrt

# ============================================
# 1. File Inspection
# ============================================

print("Current working directory:")
print(os.getcwd())

# Load the data
df = pd.read_excel('5000_rental_crm_leads.xlsx')

# Display first 20 rows
print("\nFirst 20 rows:")
print(df.head(20))

# Display last 20 rows
print("\nLast 20 rows:")
print(df.tail(20))

# Dataset information
print("\nDataset info:")
df.info()

# Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Dataset shape
print(f"\nDataset shape: {df.shape}")

# Data types
print("\nData types:")
print(df.dtypes)

# ============================================
# 2. Missing Values
# ============================================

print("\nMissing values:")
print(df.isnull().sum())

# Drop rows with missing values
df_clean = df.dropna()
print(f"\nOriginal data shape: {df.shape}")
print(f"After dropping missing values: {df_clean.shape}")

# Fill missing values with 0
df_fill = df.fillna(0)
print(f"After filling with 0: {df_fill.shape}")

# Function to fill missing values with -1
def fill_missing_minus_one(df):
    df_copy = df.copy()
    df_copy.fillna(-1, inplace=True)
    return df_copy

# ============================================
# 3. Remove Duplicates
# ============================================

print(f"\nOriginal data shape: {df.shape}")
df_no_duplicates = df.drop_duplicates()
print(f"After removing duplicates: {df_no_duplicates.shape}")

# ============================================
# 4. Feature Engineering
# ============================================

# Reload data for feature engineering
df = pd.read_excel('5000_rental_crm_leads.xlsx')

# 4.1 Remove rows with missing values
df_clean = df.dropna()
print(f"After dropping missing values: {df_clean.shape}")

# 4.2 Fill missing values with constant
df_fill = df.fillna({'name': 'Unknown', 'budget_min': 0, 'budget_max': 0})
print(f"After filling with constants: {df_fill.shape}")

# 4.3 Remove all duplicate rows
df_no_duplicates = df.drop_duplicates()
print(f"After removing duplicates: {df_no_duplicates.shape}")

# FIXED: Clean preferred_area column (was incorrectly using user_type)
if 'preferred_area' in df.columns:
    df['preferred_area'] = df['preferred_area'].replace({
        'BHK': 'bhk',
        'Bhk': 'bhk',
    })
    print("Before cleaning:", df['preferred_area'].unique()[:10])  # Show first 10
    print("After cleaning:", df['preferred_area'].unique()[:10])

# ============================================
# 5. Lead Scoring Functions
# ============================================

def map_probability_to_category(prob_score):
    """Map probability (0-100) to category label."""
    if prob_score >= 70:
        return "Hot"
    if prob_score >= 40:
        return "Warm"
    return "Cold"

def score_leads_from_model(df, model, feature_cols):
    """
    Score leads using a trained model (XGBoost sklearn/booster).
    - df: DataFrame of raw leads
    - model: trained model object
    - feature_cols: columns used for prediction
    Returns df with new columns: score (0-100), category
    """
    X = df[feature_cols].fillna(0)
    # handle both sklearn wrapper and xgboost.Booster
    try:
        probs = model.predict_proba(X)[:, 1]  # sklearn style
    except Exception:
        try:
            dmat = xgb.DMatrix(X)
            probs = model.predict(dmat)  # xgboost.Booster
        except Exception:
            raise RuntimeError("Model predict failed. Check model type.")
    
    df = df.copy()
    df['score'] = (probs * 100).round(2)
    df['category'] = df['score'].apply(map_probability_to_category)
    df = df.sort_values('score', ascending=False)
    return df

# ============================================
# 6. Lead Prioritization Functions
# ============================================

def prioritize_leads(df, top_n_per_agent=10):
    """
    Given scored df with 'score' and 'agent_id' optional, return top N prioritized leads.
    If no agent assignment present, it returns global top_n.
    """
    if 'agent_id' in df.columns:
        prioritized = df.groupby('agent_id').apply(
            lambda g: g.sort_values('score', ascending=False).head(top_n_per_agent)
        ).reset_index(drop=True)
    else:
        prioritized = df.sort_values('score', ascending=False).head(top_n_per_agent)
    return prioritized

def select_for_immediate_followup(df, min_score=60, max_age_days=7):
    """Select leads requiring immediate follow-up based on score and age."""
    df = df.copy()
    if 'created_date' in df.columns:
        df['lead_age_days'] = (pd.Timestamp.now() - pd.to_datetime(df['created_date'], errors='coerce')).dt.days
        df['lead_age_days'] = df['lead_age_days'].fillna(999)
        return df[(df['score'] >= min_score) & (df['lead_age_days'] <= max_age_days)].sort_values('score', ascending=False)
    else:
        # If no created_date, just filter by score
        return df[df['score'] >= min_score].sort_values('score', ascending=False)

# ============================================
# 7. Automated Alerts Functions
# ============================================

def send_email_notification(to_email, subject, body, smtp_cfg):
    """Simple SMTP mailer. Replace with SendGrid/Twilio Send API in prod."""
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = smtp_cfg['from']
        msg['To'] = to_email
        msg.set_content(body)
        
        with smtplib.SMTP(smtp_cfg['host'], smtp_cfg.get('port', 25)) as s:
            if smtp_cfg.get('starttls', False):
                s.starttls()
            if smtp_cfg.get('username'):
                s.login(smtp_cfg['username'], smtp_cfg['password'])
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False

def alert_high_intent_leads(df, smtp_cfg, agent_lookup):
    """
    For all leads category == 'Hot', send alert to assigned agent(s).
    - agent_lookup: function or dict mapping agent_id -> {'email':..., 'phone':...}
    """
    hot = df[df['category'] == 'Hot'].copy()
    for _, lead in hot.iterrows():
        agent_info = agent_lookup.get(lead.get('agent_id')) if isinstance(agent_lookup, dict) else agent_lookup(lead)
        if not agent_info:
            continue
        
        subject = f"🔥 Hot Lead: {lead.get('lead_id')} ({lead.get('score')}%)"
        body = f"""
Lead ID: {lead.get('lead_id')}
Name: {lead.get('name')}
Score: {lead.get('score')}%
Last active: {lead.get('last_active')}
Source: {lead.get('source')}
Please follow up within the SLA.
"""
        # Email (placeholder)
        send_email_notification(agent_info['email'], subject, body, smtp_cfg)
        # TODO: phone / SMS via Twilio or WhatsApp API

# ============================================
# 8. Lead Assignment Functions
# ============================================

def assign_leads_round_robin(df, agents):
    """
    Assign leads to agents in round-robin order (simple balancing).
    - agents: list of agent dicts or agent ids
    """
    df = df.copy()
    if len(agents) == 0:
        return df
    
    agent_ids = [agent if isinstance(agent, (int, str)) else agent['id'] for agent in agents]
    assigned = []
    
    for i, _ in enumerate(df.sort_values('score', ascending=False).iterrows()):
        assigned.append(agent_ids[i % len(agent_ids)])
    
    df['agent_id'] = assigned
    return df

def assign_by_capacity_and_skill(df, agents_df):
    """
    agents_df columns: agent_id, capacity_remaining, skill_score (higher better)
    Strategy: consider score * skill_score and capacity.
    """
    df = df.copy()
    df['assigned_agent'] = None
    agents_sorted = agents_df.sort_values(['capacity_remaining', 'skill_score'], ascending=[False, False]).copy()
    
    for idx, lead in df.sort_values('score', ascending=False).iterrows():
        # pick first agent with capacity > 0 and highest skill relevancy
        available_agents = agents_sorted[agents_sorted['capacity_remaining'] > 0]
        if len(available_agents) == 0:
            break
        
        candidate = available_agents.iloc[0]
        df.at[idx, 'assigned_agent'] = candidate['agent_id']
        # Update capacity
        agents_sorted.loc[agents_sorted['agent_id'] == candidate['agent_id'], 'capacity_remaining'] -= 1
    
    return df

# ============================================
# 9. KPI Tracking Functions
# ============================================

def compute_kpis(leads_df, interactions_df, revenue_df):
    """
    - leads_df must include lead_id, assigned_agent, created_date, converted_date (if any)
    - interactions_df: events with lead_id, event_type, timestamp
    - revenue_df: lead_id, revenue_amount
    Returns a KPIs dict and KPI DataFrames.
    """
    # Time to convert (days)
    conv = leads_df.dropna(subset=['converted_date']).copy()
    if len(conv) > 0:
        conv['time_to_convert_days'] = (pd.to_datetime(conv['converted_date']) - 
                                        pd.to_datetime(conv['created_date'])).dt.total_seconds() / 86400.0

        # Per-agent KPIs
        agent_stats = conv.groupby('assigned_agent').agg(
            conversions=('lead_id', 'count'),
            avg_time_to_convert=('time_to_convert_days', 'mean')
        ).reset_index()

        revenue_per_lead = revenue_df.groupby('lead_id')['revenue_amount'].sum().reset_index()
        conv = conv.merge(revenue_per_lead, on='lead_id', how='left').fillna({'revenue_amount': 0})

        revenue_by_agent = conv.groupby('assigned_agent')['revenue_amount'].sum().reset_index().rename(
            columns={'revenue_amount': 'revenue'})

        agent_kpis = agent_stats.merge(revenue_by_agent, on='assigned_agent', how='left').fillna(0)
        
        overall = {
            'total_leads': len(leads_df),
            'total_conversions': len(conv),
            'avg_time_to_convert_days': conv['time_to_convert_days'].mean(),
            'total_revenue': conv['revenue_amount'].sum()
        }
    else:
        overall = {
            'total_leads': len(leads_df),
            'total_conversions': 0,
            'avg_time_to_convert_days': None,
            'total_revenue': 0
        }
        agent_kpis = pd.DataFrame()
    
    return overall, agent_kpis

# ============================================
# 10. Batch Scoring and CRM Integration
# ============================================

def batch_score_and_export(df, model, feature_cols, crm_update_endpoint, api_key):
    """
    Score leads in batch and push results back to CRM via API.
    - crm_update_endpoint: URL to update lead fields in CRM
    """
    scored = score_leads_from_model(df, model, feature_cols)
    
    # Create payload (CRM-specific)
    payloads = []
    for _, row in scored.iterrows():
        payloads.append({
            "lead_id": int(row['lead_id']) if pd.notna(row['lead_id']) else None,
            "score": float(row['score']),
            "category": row['category']
        })
    
    # Push updates (example)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for p in payloads:
        try:
            # Replace with CRM bulk API when available
            requests.post(crm_update_endpoint, json=p, headers=headers, timeout=5)
        except Exception as e:
            print(f"Failed to update lead {p['lead_id']}: {e}")
    
    return scored

# ============================================
# 11. Budget Match Functions
# ============================================

def budget_match_score(lead_budget, property_rent):
    """
    Score in [0,1]. 1 = perfect match (equal), 0 = huge mismatch.
    Uses relative difference: score = max(0, 1 - abs(diff) / max(lead_budget, property_rent))
    Works when both are positive numbers.
    """
    if pd.isna(lead_budget) or pd.isna(property_rent):
        return np.nan
    if lead_budget <= 0 or property_rent <= 0:
        return np.nan
    
    diff = abs(lead_budget - property_rent)
    denom = max(lead_budget, property_rent)
    return max(0.0, 1.0 - diff / denom)

def budget_match_best_property(lead_budget, properties_df, rent_col='rent'):
    """
    For a lead, compute best budget_match across multiple candidate properties and return the best score.
    properties_df: DataFrame of properties (each row has rent_col)
    """
    if len(properties_df) == 0 or pd.isna(lead_budget):
        return np.nan
    
    rents = properties_df[rent_col].dropna().astype(float)
    if rents.empty:
        return np.nan
    
    scores = rents.apply(lambda r: budget_match_score(lead_budget, r))
    return float(scores.max())

def compute_budget_match(df, lead_budget_col="budget", property_min_col="prop_rent_min", property_max_col="prop_rent_max"):
    """
    Returns a budget_match score in [0,1]:
      - 1.0 if lead budget falls inside property's [min,max]
      - declines linearly when outside (capped at 0)
    """
    lb = pd.to_numeric(df[lead_budget_col], errors='coerce')
    pmin = pd.to_numeric(df[property_min_col], errors='coerce')
    pmax = pd.to_numeric(df[property_max_col], errors='coerce')

    # when budget inside range => 1
    inside = (lb >= pmin) & (lb <= pmax)
    budget_match = np.where(inside, 1.0, 0.0)

    # For outside values, compute distance ratio and map to (0,1)
    scale = (pmax - pmin).abs().replace(0, 5000)  # fallback scale
    below = lb < pmin
    above = lb > pmax

    budget_match = np.where(
        below,
        np.maximum(0, 1 - (pmin - lb) / (scale + 1e-9)),
        budget_match
    )
    budget_match = np.where(
        above,
        np.maximum(0, 1 - (lb - pmax) / (scale + 1e-9)),
        budget_match
    )

    return pd.Series(budget_match, index=df.index, name="budget_match")

# ============================================
# 12. Area Match Functions
# ============================================

def simple_area_match(lead_area, property_area):
    """
    Returns 1 if exact match, 0.5 if same locality token exists, else 0.
    Accepts strings or list-like.
    """
    if pd.isna(lead_area) or pd.isna(property_area):
        return 0.0
    
    la = str(lead_area).lower()
    pa = str(property_area).lower()
    
    if la == pa:
        return 1.0
    
    # token overlap
    la_tokens = set(la.replace(',', ' ').split())
    pa_tokens = set(pa.replace(',', ' ').split())
    overlap = la_tokens.intersection(pa_tokens)
    
    if len(overlap) > 0:
        return min(0.75, 0.5 + 0.1 * len(overlap))  # modest boost for partial matches
    
    return 0.0

def tfidf_area_match(series_lead_areas, series_property_areas):
    """Compute TF-IDF based area matching scores."""
    corpus = pd.concat([series_lead_areas.fillna(""), series_property_areas.fillna("")]).astype(str)
    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = tf.fit_transform(corpus)
    
    n = len(series_lead_areas)
    lead_mat = tfidf[:n]
    prop_mat = tfidf[n:]
    
    sims = np.array([cosine_similarity(lead_mat[i], prop_mat[i])[0, 0] for i in range(n)])
    return pd.Series(sims, index=series_lead_areas.index, name="area_match_tfidf")

# ============================================
# 13. Engagement Score Functions
# ============================================

def compute_engagement_score(df,
                             views_col="views_count",
                             avg_time_col="avg_view_time_sec",
                             saved_col="saved_properties",
                             repeat_col="repeated_visits"):
    """Compute engagement score from behavioral metrics."""
    # Normalize each metric to 0-1 using log-scaling to reduce skew
    def norm_series(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        max_val = s.max()
        if max_val > 0:
            return np.log1p(s) / np.log1p(max_val)
        return s * 0.0

    v = norm_series(df[views_col])
    t = norm_series(df[avg_time_col])
    s = norm_series(df[saved_col])
    r = norm_series(df[repeat_col])

    # weights (tune for your business)
    wv, wt, ws, wr = 0.4, 0.25, 0.2, 0.15
    engagement = wv * v + wt * t + ws * s + wr * r
    
    # Clip 0-1
    return pd.Series(np.clip(engagement, 0, 1), index=df.index, name="engagement_score")

def compute_interaction_features(df,
                                 first_resp_col="first_response_time_hours",
                                 followups_col="follow_up_count",
                                 lead_resp_col="lead_response_time_hours",
                                 whatsapp_col="whatsapp_clicks",
                                 call_col="call_clicks",
                                 chat_col="chat_messages"):
    """Compute interaction-based features."""
    # Ensure numeric and fillna
    res = pd.DataFrame(index=df.index)
    
    # First response time
    if first_resp_col in df.columns:
        res["first_response_time_hours"] = pd.to_numeric(df[first_resp_col], errors="coerce")
        median_val = res["first_response_time_hours"].median()
        res["first_response_time_hours"] = res["first_response_time_hours"].fillna(median_val if pd.notna(median_val) else 24)
    else:
        res["first_response_time_hours"] = 24

    # Follow-up count
    res["follow_up_count"] = pd.to_numeric(df.get(followups_col, 0), errors="coerce").fillna(0)
    
    # Lead response time
    if lead_resp_col in df.columns:
        res["lead_response_time_hours"] = pd.to_numeric(df[lead_resp_col], errors="coerce")
        median_val = res["lead_response_time_hours"].median()
        res["lead_response_time_hours"] = res["lead_response_time_hours"].fillna(median_val if pd.notna(median_val) else 24)
    else:
        res["lead_response_time_hours"] = 24

    # Interaction counts
    res["whatsapp_clicks"] = pd.to_numeric(df.get(whatsapp_col, 0), errors="coerce").fillna(0)
    res["call_clicks"] = pd.to_numeric(df.get(call_col, 0), errors="coerce").fillna(0)
    res["chat_messages"] = pd.to_numeric(df.get(chat_col, 0), errors="coerce").fillna(0)

    # Derived features
    res["agent_response_fast"] = (res["first_response_time_hours"] <= 1).astype(int)
    res["interaction_count"] = res[["follow_up_count", "whatsapp_clicks", "call_clicks", "chat_messages"]].sum(axis=1)
    
    return res

# ============================================
# 14. RandomForest Model Training
# ============================================

def train_randomforest_model():
    """
    RandomForest lead scoring pipeline (supervised + fallback semi-supervised)
    """
    # Load data
    path = "5000_rental_crm_leads.xlsx"
    df = pd.read_excel(path)

    print("Rows, cols:", df.shape)
    print("Columns:", df.columns.tolist())

    # Feature engineering
    if "budget_min" in df.columns and "budget_max" in df.columns:
        df["budget_mid"] = df[["budget_min", "budget_max"]].mean(axis=1)
    elif "budget" in df.columns:
        df["budget_mid"] = pd.to_numeric(df["budget"], errors='coerce')
    else:
        df["budget_mid"] = np.nan

    # Budget match feature
    if df["budget_mid"].notna().any():
        min_b, max_b = df["budget_mid"].min(), df["budget_mid"].max()
        if min_b == max_b or pd.isna(min_b) or pd.isna(max_b):
            df["budget_match"] = 1.0
        else:
            df["budget_match"] = (df["budget_mid"] - min_b) / (max_b - min_b)
    else:
        df["budget_match"] = 0.5

    # Area match feature
    if "preferred_area" in df.columns:
        area_freq = df["preferred_area"].fillna("unknown").value_counts(normalize=True)
        df["area_match"] = df["preferred_area"].fillna("unknown").map(area_freq).fillna(0.5)
    else:
        df["area_match"] = 0.5

    # Behavior scores
    beh_cols = ["views_count", "avg_view_time_sec", "saved_properties", "repeated_visits"]
    for c in beh_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Normalize behavior columns
    for c in beh_cols:
        mx = df[c].max()
        if mx > 0:
            df[c + "_norm"] = df[c] / mx
        else:
            df[c + "_norm"] = 0.0

    # Engagement score
    df["engagement_score"] = (
        0.4 * df["views_count_norm"] +
        0.2 * df["avg_view_time_sec_norm"] +
        0.25 * df["saved_properties_norm"] +
        0.15 * df["repeated_visits_norm"]
    )

    # Interaction features
    inter_cols = ["whatsapp_clicks", "call_clicks", "chat_messages"]
    for c in inter_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["total_interactions"] = df[inter_cols].sum(axis=1)

    # Recency features
    if "last_active_time" in df.columns:
        df["last_active_time"] = pd.to_datetime(df["last_active_time"], errors="coerce")
        now = pd.Timestamp.now()
        df["days_since_active"] = (now - df["last_active_time"]).dt.days.fillna(999)
        df["recency_score"] = 1 / (1 + df["days_since_active"])
    else:
        df["recency_score"] = 0.0

    # Prepare features
    feature_cols = [
        "budget_match", "area_match", "engagement_score",
        "total_interactions", "recency_score",
    ]
    
    if "source" in df.columns:
        feature_cols.append("source")
    if "bhk" in df.columns:
        feature_cols.append("bhk")

    X = df[feature_cols].copy()

    # Prepare target
    y = None
    if "converted" in df.columns:
        y = pd.to_numeric(df["converted"], errors="coerce")

    # Handle missing labels
    if y is None:
        print("No 'converted' column found. Using unsupervised KMeans to create pseudo-labels.")
        numeric_for_kmeans = X.select_dtypes(include=[np.number]).fillna(0)
        kmeans = KMeans(n_clusters=2, random_state=42)
        pseudo_labels = kmeans.fit_predict(numeric_for_kmeans)
        y = pd.Series(pseudo_labels, index=X.index)
        use_pseudo = True
    else:
        use_pseudo = False
        n_missing = y.isna().sum()
        print(f"'converted' column found with {n_missing} missing values out of {len(y)} rows.")
        
        if n_missing > 0:
            print("Dropping rows with NaN labels for supervised training.")
        mask = y.notna()
        X = X[mask].reset_index(drop=True)
        y = y[mask].astype(int).reset_index(drop=True)

    # Check if we have enough data
    if len(X) < 10:
        raise ValueError("Not enough data to train model after cleaning")

    # Build preprocessing pipeline
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if num_cols:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_transformer, num_cols))

    if cat_cols:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # RandomForest model
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("rf", rf)
    ])

    # Train/test split
    stratify_y = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify_y
    )

    print(f"Training rows: {X_train.shape}, Test rows: {X_test.shape}")

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

    # Evaluation
    print("\n--- Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            print("ROC AUC:", roc_auc_score(y_test, y_proba))
        except Exception as e:
            print("ROC AUC couldn't be computed:", e)
    
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Cross-validation
    if len(np.unique(y)) == 2 and len(X) >= 20:
        try:
            cv_scores = cross_val_score(pipeline, X, y, cv=min(5, len(X) // 4), scoring="roc_auc")
            try:
    print(f"CV ROC AUC mean: {cv_scores.mean():.4f}")
except Exception as e:
    print(f"Error calculating CV ROC AUC mean: {e}")
