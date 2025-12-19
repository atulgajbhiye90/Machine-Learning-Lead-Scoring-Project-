import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Lead Scoring Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - PROFESSIONAL DESIGN
# ============================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: white;
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }
    
    /* Custom Metric Card */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    
    .metric-card-hot {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-color: #ef4444;
    }
    
    .metric-card-warm {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-color: #f59e0b;
    }
    
    .metric-card-cold {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-color: #3b82f6;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: #f8fafc;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #64748b;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: white;
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        font-weight: 600;
        color: #1e293b;
        padding: 1rem;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Info/Success/Warning Boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
        font-weight: 500;
    }
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Table */
    .dataframe {
        border: none !important;
        border-radius: 8px;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    
    .dataframe tbody tr:hover {
        background-color: #f8fafc !important;
        transform: scale(1.01);
        transition: all 0.2s ease;
    }
    
    /* Section Headers */
    h3 {
        color: #1e293b;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 2px solid #f1f5f9;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        border-color: #667eea;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Welcome section */
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 2px solid #f1f5f9;
    }
    
    .feature-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-hot {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #dc2626;
        border: 2px solid #ef4444;
    }
    
    .badge-warm {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #d97706;
        border: 2px solid #f59e0b;
    }
    
    .badge-cold {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #2563eb;
        border: 2px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def map_probability_to_category(prob_score):
    """Map probability (0-100) to category label."""
    if prob_score >= 70:
        return "Hot"
    elif prob_score >= 40:
        return "Warm"
    else:
        return "Cold"

@st.cache_data
def load_data(file_path):
    """Load data from Excel file"""
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_resource
def train_model(df):
    """Train RandomForest model with progress tracking"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Feature engineering
    status_text.markdown("🔧 **Step 1/5:** Feature Engineering...")
    progress_bar.progress(20)
    
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
    status_text.markdown("📊 **Step 2/5:** Preparing Features...")
    progress_bar.progress(40)
    
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
    if y is None or y.isna().all():
        status_text.markdown("🤖 **Using unsupervised learning:** Creating pseudo-labels with KMeans...")
        numeric_for_kmeans = X.select_dtypes(include=[np.number]).fillna(0)
        kmeans = KMeans(n_clusters=2, random_state=42)
        pseudo_labels = kmeans.fit_predict(numeric_for_kmeans)
        y = pd.Series(pseudo_labels, index=X.index)
    else:
        mask = y.notna()
        X = X[mask].reset_index(drop=True)
        y = y[mask].astype(int).reset_index(drop=True)

    if len(X) < 10:
        raise ValueError("Not enough data to train model after cleaning")

    # Build preprocessing pipeline
    status_text.markdown("🔨 **Step 3/5:** Building ML Pipeline...")
    progress_bar.progress(60)
    
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
    status_text.markdown("🎯 **Step 4/5:** Training Model...")
    progress_bar.progress(80)
    
    stratify_y = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify_y
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = None
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            pass

    # Score all leads
    status_text.markdown("✨ **Step 5/5:** Scoring All Leads...")
    progress_bar.progress(100)
    
    df_scored = df.copy()
    lead_probability = pipeline.predict_proba(X)[:, 1]
    df_scored.loc[X.index, "lead_score"] = (lead_probability * 100).round(0).astype(int)
    df_scored["lead_score"] = df_scored["lead_score"].fillna(0).astype(int)
    df_scored["lead_category"] = df_scored["lead_score"].apply(map_probability_to_category)
    
    status_text.markdown("✅ **Model Training Complete!**")
    progress_bar.progress(100)
    
    return pipeline, df_scored, feature_cols, accuracy, roc_auc

def create_gauge_chart(value, title, color):
    """Create a professional gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20, 'color': '#1e293b', 'family': 'Inter'}},
        number={'font': {'size': 40, 'color': color, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#cbd5e1"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 40], 'color': '#dbeafe'},
                {'range': [40, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    
    return fig

# ============================================
# MAIN APP
# ============================================

def main():
    # Animated Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 class="main-header">🚀 AI-Powered Lead Scoring Platform</h1>
            <p class="sub-header">Transform your CRM leads into actionable insights with machine learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Center")
        st.markdown("")
        
        # File upload option
        upload_option = st.radio(
            "📂 Select Data Source:",
            ["Use Default Dataset", "Upload Custom File"],
            help="Choose between using the default dataset or uploading your own"
        )
        
        if upload_option == "Upload Custom File":
            uploaded_file = st.file_uploader(
                "Upload Excel File",
                type=['xlsx', 'xls'],
                help="Upload your CRM leads Excel file"
            )
            data_path = uploaded_file
        else:
            data_path = "5000_rental_crm_leads.xlsx"
        
        st.markdown("---")
        
        # Train button with icon
        train_button = st.button(
            "🚀 Train Model & Score Leads",
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Sidebar info boxes
        st.markdown("""
            <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                <h4 style='color: white; margin: 0;'>💡 Pro Tip</h4>
                <p style='color: #cbd5e1; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>
                    Training typically takes 30-60 seconds. Larger datasets may take longer.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if 'scored_df' in st.session_state:
            df = st.session_state['scored_df']
            st.markdown("""
                <div style='background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 10px;'>
                    <h4 style='color: white; margin: 0;'>✅ Model Status</h4>
                    <p style='color: #cbd5e1; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>
                        Model is trained and ready!<br>
                        <strong>{:,}</strong> leads scored
                    </p>
                </div>
            """.format(len(df)), unsafe_allow_html=True)
    
    # Main content
    if train_button and data_path:
        # Load data
        with st.spinner("🔄 Loading data..."):
            df = load_data(data_path)
        
        if df is not None:
            # Display data info
            with st.expander("📊 Dataset Preview & Information", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📝 Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("📋 Columns", len(df.columns))
                with col3:
                    st.metric("❓ Missing Values", df.isnull().sum().sum())
                with col4:
                    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                    st.metric("💾 Memory", f"{memory_usage:.2f} MB")
                
                st.dataframe(df.head(10), use_container_width=True)
            
            # Train model
            try:
                model, scored_df, features, accuracy, roc_auc = train_model(df)
                
                # Store in session state
                st.session_state['model'] = model
                st.session_state['scored_df'] = scored_df
                st.session_state['features'] = features
                st.session_state['accuracy'] = accuracy
                st.session_state['roc_auc'] = roc_auc
                
                st.success("✅ Model trained successfully! Explore the results below.")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error during training: {e}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
    
    # Display results if model is trained
    if 'scored_df' in st.session_state:
        df = st.session_state['scored_df']
        accuracy = st.session_state.get('accuracy', 0)
        roc_auc = st.session_state.get('roc_auc', None)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard",
            "🔥 Priority Leads",
            "📈 Analytics",
            "📋 All Leads",
            "💾 Export"
        ])
        
        with tab1:
            st.markdown("### 📊 Performance Dashboard")
            st.markdown("")
            
            # Top metrics row with custom styling
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📊 Total Leads", f"{len(df):,}", help="Total number of leads in dataset")
            with col2:
                hot_leads = len(df[df['lead_category'] == 'Hot'])
                st.markdown(f"""
                    <div class="metric-card metric-card-hot">
                        <div style="font-size: 0.9rem; color: #991b1b; font-weight: 600; text-transform: uppercase;">🔥 Hot Leads</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #dc2626; margin: 0.5rem 0;">{hot_leads}</div>
                        <div style="font-size: 0.85rem; color: #991b1b;">{hot_leads/len(df)*100:.1f}% of total</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                warm_leads = len(df[df['lead_category'] == 'Warm'])
                st.markdown(f"""
                    <div class="metric-card metric-card-warm">
                        <div style="font-size: 0.9rem; color: #92400e; font-weight: 600; text-transform: uppercase;">🌡️ Warm Leads</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #d97706; margin: 0.5rem 0;">{warm_leads}</div>
                        <div style="font-size: 0.85rem; color: #92400e;">{warm_leads/len(df)*100:.1f}% of total</div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                cold_leads = len(df[df['lead_category'] == 'Cold'])
                st.markdown(f"""
                    <div class="metric-card metric-card-cold">
                        <div style="font-size: 0.9rem; color: #1e40af; font-weight: 600; text-transform: uppercase;">❄️ Cold Leads</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #2563eb; margin: 0.5rem 0;">{cold_leads}</div>
                        <div style="font-size: 0.85rem; color: #1e40af;">{cold_leads/len(df)*100:.1f}% of total</div>
                    </div>
                """, unsafe_allow_html=True)
            with col5:
                avg_score = df['lead_score'].mean()
                st.metric("⭐ Average Score", f"{avg_score:.1f}", help="Mean lead score across all leads")
            
            st.markdown("")
            st.markdown("---")
            
            # Model Performance Section
            st.markdown("### 🎯 Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gauge_accuracy = create_gauge_chart(accuracy*100, "Accuracy Score", "#667eea")
                st.plotly_chart(gauge_accuracy, use_container_width=True)
            
            with col2:
                if roc_auc:
                    gauge_roc = create_gauge_chart(roc_auc*100, "ROC AUC Score", "#764ba2")
                    st.plotly_chart(gauge_roc, use_container_width=True)
                else:
                    st.info("ROC AUC metric not available for this model")
            
            with col3:
                conversion_rate = (df['lead_score'] > 70).sum() / len(df) * 100
                gauge_conversion = create_gauge_chart(conversion_rate, "High-Quality Leads %", "#10b981")
                st.plotly_chart(gauge_conversion, use_container_width=True)
            
            st.markdown("---")
            
            # Visualizations
            st.markdown("### 📊 Lead Distribution Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced Pie Chart
                category_counts = df['lead_category'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    hole=0.4,
                    marker=dict(colors=['#ef4444', '#f59e0b', '#3b82f6']),
                    textinfo='label+percent',
                    textfont=dict(size=14, family='Inter', color='white'),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )])
                fig_pie.update_layout(
                    title={
                        'text': "Lead Category Distribution",
                        'font': {'size': 18, 'family': 'Inter', 'color': '#1e293b'},
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    height=400,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Enhanced Histogram
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=df['lead_score'],
                    nbinsx=20,
                    marker=dict(
                        color=df['lead_score'],
                        colorscale='Viridis',
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
                ))
                fig_hist.update_layout(
                    title={
                        'text': "Lead Score Distribution",
                        'font': {'size': 18, 'family': 'Inter', 'color': '#1e293b'},
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    xaxis_title="Lead Score",
                    yaxis_title="Number of Leads",
                    height=400,
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter'},
                    xaxis=dict(gridcolor='#f1f5f9'),
                    yaxis=dict(gridcolor='#f1f5f9')
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            st.markdown("### 🔥 Top Priority Leads")
            st.markdown("")
            
            # Enhanced Filter controls
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                category_filter = st.multiselect(
                    "🎯 Filter by Category:",
                    options=['Hot', 'Warm', 'Cold'],
                    default=['Hot'],
                    help="Select one or more categories to filter"
                )
            with col2:
                min_score = st.slider("📊 Minimum Score:", 0, 100, 70, help="Filter leads with score above this value")
            with col3:
                show_count = st.number_input("📋 Show Top:", 10, 100, 20, step=10)
            
            # Filter data
            filtered_df = df[
                (df['lead_category'].isin(category_filter)) & 
                (df['lead_score'] >= min_score)
            ]
            
            # Display columns selection
            display_cols = ['lead_id', 'name', 'lead_score', 'lead_category']
            optional_cols = ['source', 'budget_mid', 'preferred_area', 'total_interactions']
            available_optional = [col for col in optional_cols if col in filtered_df.columns]
            
            if available_optional:
                selected_cols = st.multiselect(
                    "➕ Additional Columns:",
                    options=available_optional,
                    default=available_optional[:2] if len(available_optional) >= 2 else available_optional
                )
                display_cols.extend(selected_cols)
            
            st.markdown("")
            
            # Display top leads
            top_leads = filtered_df.nlargest(show_count, 'lead_score')[display_cols]
            
            # Enhanced styling
            def highlight_category(row):
                if row['lead_category'] == 'Hot':
                    return ['background-color: #fee2e2'] * len(row)
                elif row['lead_category'] == 'Warm':
                    return ['background-color: #fef3c7'] * len(row)
                else:
                    return ['background-color: #dbeafe'] * len(row)
            
            styled_df = top_leads.style.apply(highlight_category, axis=1)
            
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Filtered Leads", len(filtered_df))
            with col2:
                st.metric("📊 Avg Score (Filtered)", f"{filtered_df['lead_score'].mean():.1f}")
            with col3:
                st.metric("📈 Max Score", filtered_df['lead_score'].max())
        
        with tab3:
            st.markdown("### 📈 Advanced Analytics & Insights")
            st.markdown("")
            
            # Source analysis if available
            if 'source' in df.columns:
                st.markdown("#### 🎯 Lead Performance by Source")
                source_stats = df.groupby('source').agg({
                    'lead_score': ['mean', 'count'],
                    'lead_category': lambda x: (x == 'Hot').sum()
                }).round(2)
                source_stats.columns = ['Avg Score', 'Count', 'Hot Leads']
                source_stats = source_stats.sort_values('Avg Score', ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(source_stats, use_container_width=True)
                with col2:
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=source_stats.index,
                        y=source_stats['Avg Score'],
                        marker=dict(
                            color=source_stats['Avg Score'],
                            colorscale='Viridis',
                            line=dict(color='white', width=2)
                        ),
                        text=source_stats['Avg Score'].round(1),
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Avg Score: %{y:.1f}<extra></extra>'
                    ))
                    fig_bar.update_layout(
                        title="Average Lead Score by Source",
                        xaxis_title="Source",
                        yaxis_title="Average Score",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'family': 'Inter'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.markdown("")
            
            # Budget analysis if available
            if 'budget_mid' in df.columns:
                st.markdown("#### 💰 Score vs Budget Analysis")
                fig_scatter = px.scatter(
                    df.dropna(subset=['budget_mid']),
                    x='budget_mid',
                    y='lead_score',
                    color='lead_category',
                    size='total_interactions' if 'total_interactions' in df.columns else None,
                    title='Lead Score vs Budget',
                    labels={'budget_mid': 'Budget', 'lead_score': 'Lead Score'},
                    color_discrete_map={'Hot': '#ef4444', 'Warm': '#f59e0b', 'Cold': '#3b82f6'},
                    hover_data=['name'] if 'name' in df.columns else None
                )
                fig_scatter.update_layout(
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.markdown("")
            
            # Engagement analysis
            if 'engagement_score' in df.columns:
                st.markdown("#### 📱 Engagement Score Distribution by Category")
                fig_box = go.Figure()
                for category, color in [('Hot', '#ef4444'), ('Warm', '#f59e0b'), ('Cold', '#3b82f6')]:
                    data = df[df['lead_category'] == category]['engagement_score']
                    fig_box.add_trace(go.Box(
                        y=data,
                        name=category,
                        marker_color=color,
                        boxmean='sd'
                    ))
                fig_box.update_layout(
                    title="Engagement Score Distribution",
                    yaxis_title="Engagement Score",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        with tab4:
            st.markdown("### 📋 Complete Lead Database")
            st.markdown("")
            
            # Enhanced search and filter
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                search_term = st.text_input("🔍 Search:", placeholder="Name or ID...")
            with col2:
                score_range = st.slider("📊 Score Range:", 0, 100, (0, 100))
            with col3:
                sort_by = st.selectbox("📈 Sort by:", ['lead_score', 'lead_id', 'name'])
            with col4:
                sort_order = st.radio("⬆️⬇️", ['Descending', 'Ascending'])
            
            # Apply filters
            filtered = df.copy()
            if search_term:
                filtered = filtered[
                    filtered['name'].str.contains(search_term, case=False, na=False) |
                    filtered['lead_id'].astype(str).str.contains(search_term, case=False)
                ]
            filtered = filtered[
                (filtered['lead_score'] >= score_range[0]) &
                (filtered['lead_score'] <= score_range[1])
            ]
            filtered = filtered.sort_values(sort_by, ascending=(sort_order == 'Ascending'))
            
            # Display info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"📊 Showing **{len(filtered):,}** of **{len(df):,}** leads")
            with col2:
                if len(filtered) > 0:
                    st.success(f"⭐ Avg Score: **{filtered['lead_score'].mean():.1f}**")
            with col3:
                if len(filtered) > 0:
                    hot_pct = (filtered['lead_category'] == 'Hot').sum() / len(filtered) * 100
                    st.warning(f"🔥 Hot Leads: **{hot_pct:.1f}%**")
            
            st.markdown("")
            st.dataframe(filtered, use_container_width=True, height=600)
        
        with tab5:
            st.markdown("### 💾 Export Your Scored Leads")
            st.markdown("")
            
            st.markdown("""
                <div class="card">
                    <h4>📥 Download Options</h4>
                    <p>Export your scored leads in your preferred format</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV Export
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name='scored_leads.csv',
                    mime='text/csv',
                    use_container_width=True,
                    help="Download as comma-separated values"
                )
            
            with col2:
                # Excel Export
                @st.cache_data
                def convert_df_to_excel(dataframe):
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        dataframe.to_excel(writer, index=False, sheet_name='Scored Leads')
                    return output.getvalue()
                
                excel_data = convert_df_to_excel(df)
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name='scored_leads.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True,
                    help="Download as Excel spreadsheet"
                )
            
            with col3:
                # Hot leads only
                hot_leads_df = df[df['lead_category'] == 'Hot']
                hot_csv = hot_leads_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="🔥 Hot Leads Only",
                    data=hot_csv,
                    file_name='hot_leads_only.csv',
                    mime='text/csv',
                    use_container_width=True,
                    help="Download only hot priority leads"
                )
            
            st.markdown("---")
            
            # Export summary
            st.markdown("#### 📊 Export Summary")
            
            summary_df = pd.DataFrame({
                'Metric': [
                    '📊 Total Leads',
                    '🔥 Hot Leads',
                    '🌡️ Warm Leads',
                    '❄️ Cold Leads',
                    '⭐ Average Score',
                    '📈 Highest Score',
                    '📉 Lowest Score'
                ],
                'Value': [
                    f"{len(df):,}",
                    f"{len(df[df['lead_category'] == 'Hot']):,}",
                    f"{len(df[df['lead_category'] == 'Warm']):,}",
                    f"{len(df[df['lead_category'] == 'Cold']):,}",
                    f"{df['lead_score'].mean():.2f}",
                    f"{df['lead_score'].max()}",
                    f"{df['lead_score'].min()}"
                ]
            })
            
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    else:
        # Enhanced Welcome screen
        st.markdown("""
            <div class="welcome-card">
                <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">🚀 Welcome to AI Lead Scoring</h1>
                <p style="font-size: 1.2rem; opacity: 0.9;">
                    Transform your rental CRM leads into actionable insights with cutting-edge machine learning
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="feature-item">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-title">Intelligent Scoring</div>
                    <div class="feature-desc">
                        Advanced Random Forest algorithm automatically evaluates and scores each lead based on multiple factors
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="feature-item">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Visual Analytics</div>
                    <div class="feature-desc">
                        Interactive dashboards with real-time insights, charts, and performance metrics
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="feature-item">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-title">Priority Management</div>
                    <div class="feature-desc">
                        Automatically identify and prioritize hot leads for immediate follow-up
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # How to use section
        st.markdown("### 🚀 Getting Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="card">
                    <h4>📋 Quick Start Guide</h4>
                    <ol style="line-height: 2;">
                        <li>Choose your data source in the sidebar</li>
                        <li>Click "Train Model & Score Leads"</li>
                        <li>Wait for the training process (30-60 seconds)</li>
                        <li>Explore results across different tabs</li>
                        <li>Export scored leads for your CRM</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="card">
                    <h4>📊 Required Data Format</h4>
                    <p><strong>Essential columns:</strong></p>
                    <ul style="line-height: 2;">
                        <li>✓ lead_id, name (identifiers)</li>
                        <li>✓ budget_min, budget_max (numerical)</li>
                        <li>✓ preferred_area (location)</li>
                        <li>✓ views_count, saved_properties</li>
                        <li>✓ converted (optional, for training)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
