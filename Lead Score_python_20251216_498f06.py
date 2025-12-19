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
    page_title="ML Lead Scoring System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
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
    status_text.text("Step 1/5: Feature Engineering...")
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
    status_text.text("Step 2/5: Preparing Features...")
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
        status_text.text("No 'converted' column found. Using KMeans for pseudo-labels...")
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
    status_text.text("Step 3/5: Building Pipeline...")
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
    status_text.text("Step 4/5: Training Model...")
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
    status_text.text("Step 5/5: Scoring Leads...")
    progress_bar.progress(100)
    
    df_scored = df.copy()
    lead_probability = pipeline.predict_proba(X)[:, 1]
    df_scored.loc[X.index, "lead_score"] = (lead_probability * 100).round(0).astype(int)
    df_scored["lead_score"] = df_scored["lead_score"].fillna(0).astype(int)
    df_scored["lead_category"] = df_scored["lead_score"].apply(map_probability_to_category)
    
    status_text.text("✅ Model Training Complete!")
    progress_bar.progress(100)
    
    return pipeline, df_scored, feature_cols, accuracy, roc_auc

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 ML Lead Scoring System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload option
        upload_option = st.radio(
            "Data Source:",
            ["Use Default Dataset", "Upload Custom File"]
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
        
        # Train button
        train_button = st.button("🚀 Train Model & Score Leads", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.info("💡 **Tip:** Training may take 30-60 seconds depending on data size.")
    
    # Main content
    if train_button and data_path:
        # Load data
        with st.spinner("Loading data..."):
            df = load_data(data_path)
        
        if df is not None:
            # Display data info
            with st.expander("📊 Dataset Information", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
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
                
                st.success("✅ Model trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during training: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # Display results if model is trained
    if 'scored_df' in st.session_state:
        df = st.session_state['scored_df']
        accuracy = st.session_state.get('accuracy', 0)
        roc_auc = st.session_state.get('roc_auc', None)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🔥 Top Leads", 
            "📈 Analytics",
            "📋 All Leads",
            "💾 Export"
        ])
        
        with tab1:
            st.subheader("Key Metrics")
            
            # Metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Leads", f"{len(df):,}")
            with col2:
                hot_leads = len(df[df['lead_category'] == 'Hot'])
                st.metric("🔥 Hot Leads", hot_leads, delta=f"{hot_leads/len(df)*100:.1f}%")
            with col3:
                warm_leads = len(df[df['lead_category'] == 'Warm'])
                st.metric("🌡️ Warm Leads", warm_leads, delta=f"{warm_leads/len(df)*100:.1f}%")
            with col4:
                cold_leads = len(df[df['lead_category'] == 'Cold'])
                st.metric("❄️ Cold Leads", cold_leads, delta=f"{cold_leads/len(df)*100:.1f}%")
            with col5:
                avg_score = df['lead_score'].mean()
                st.metric("Avg Score", f"{avg_score:.1f}")
            
            st.markdown("---")
            
            # Model performance
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy*100:.2f}%")
            with col2:
                if roc_auc:
                    st.metric("ROC AUC", f"{roc_auc:.3f}")
                else:
                    st.metric("ROC AUC", "N/A")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Category distribution
                category_counts = df['lead_category'].value_counts()
                fig_pie = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Lead Distribution by Category",
                    color=category_counts.index,
                    color_discrete_map={'Hot': '#ff4444', 'Warm': '#ffaa00', 'Cold': '#4444ff'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Score distribution
                fig_hist = px.histogram(
                    df,
                    x='lead_score',
                    nbins=20,
                    title="Lead Score Distribution",
                    labels={'lead_score': 'Lead Score', 'count': 'Number of Leads'}
                )
                fig_hist.update_traces(marker_color='#1f77b4')
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            st.subheader("🔥 Top 20 Priority Leads")
            
            # Filter controls
            col1, col2 = st.columns([3, 1])
            with col1:
                category_filter = st.multiselect(
                    "Filter by Category:",
                    options=['Hot', 'Warm', 'Cold'],
                    default=['Hot']
                )
            with col2:
                min_score = st.number_input("Min Score:", 0, 100, 70)
            
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
                    "Additional Columns:",
                    options=available_optional,
                    default=available_optional[:2] if len(available_optional) >= 2 else available_optional
                )
                display_cols.extend(selected_cols)
            
            # Display top leads
            top_leads = filtered_df.nlargest(20, 'lead_score')[display_cols]
            
            # Color code the dataframe
            def color_category(val):
                if val == 'Hot':
                    return 'background-color: #ffcccc'
                elif val == 'Warm':
                    return 'background-color: #fff4cc'
                else:
                    return 'background-color: #ccddff'
            
            styled_df = top_leads.style.applymap(
                color_category,
                subset=['lead_category'] if 'lead_category' in top_leads.columns else []
            )
            
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            st.metric("Filtered Leads", len(filtered_df))
        
        with tab3:
            st.subheader("📈 Advanced Analytics")
            
            # Source analysis if available
            if 'source' in df.columns:
                st.markdown("#### Lead Performance by Source")
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
                    fig = px.bar(
                        source_stats.reset_index(),
                        x='source',
                        y='Avg Score',
                        title='Average Lead Score by Source',
                        color='Avg Score',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Budget analysis if available
            if 'budget_mid' in df.columns:
                st.markdown("#### Score vs Budget Analysis")
                fig_scatter = px.scatter(
                    df.dropna(subset=['budget_mid']),
                    x='budget_mid',
                    y='lead_score',
                    color='lead_category',
                    title='Lead Score vs Budget',
                    labels={'budget_mid': 'Budget', 'lead_score': 'Lead Score'},
                    color_discrete_map={'Hot': '#ff4444', 'Warm': '#ffaa00', 'Cold': '#4444ff'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Engagement analysis
            if 'engagement_score' in df.columns:
                st.markdown("#### Engagement Score Distribution")
                fig_box = px.box(
                    df,
                    x='lead_category',
                    y='engagement_score',
                    title='Engagement Score by Category',
                    color='lead_category',
                    color_discrete_map={'Hot': '#ff4444', 'Warm': '#ffaa00', 'Cold': '#4444ff'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        with tab4:
            st.subheader("📋 All Scored Leads")
            
            # Search and filter
            col1, col2, col3 = st.columns(3)
            with col1:
                search_term = st.text_input("🔍 Search by Name/ID:", "")
            with col2:
                score_range = st.slider("Score Range:", 0, 100, (0, 100))
            with col3:
                sort_by = st.selectbox("Sort by:", ['lead_score', 'lead_id', 'name'])
            
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
            filtered = filtered.sort_values(sort_by, ascending=False)
            
            st.info(f"Showing {len(filtered)} of {len(df)} leads")
            st.dataframe(filtered, use_container_width=True, height=600)
        
        with tab5:
            st.subheader("💾 Export Scored Leads")
            
            st.markdown("#### Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name='scored_leads.csv',
                    mime='text/csv',
                    use_container_width=True
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
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name='scored_leads.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Export summary
            st.markdown("#### Export Summary")
            summary_data = {
                'Metric': ['Total Leads', 'Hot Leads', 'Warm Leads', 'Cold Leads', 'Average Score'],
                'Value': [
                    len(df),
                    len(df[df['lead_category'] == 'Hot']),
                    len(df[df['lead_category'] == 'Warm']),
                    len(df[df['lead_category'] == 'Cold']),
                    f"{df['lead_score'].mean():.2f}"
                ]
            }
            st.table(pd.DataFrame(summary_data))
    
    else:
        # Welcome screen
        st.info("👈 Click 'Train Model & Score Leads' in the sidebar to get started!")
        
        st.markdown("""
        ### 🎯 Welcome to ML Lead Scoring System
        
        This application uses machine learning to automatically score and prioritize your rental CRM leads.
        
        #### ✨ Features:
        - 🤖 **Automated Lead Scoring** using Random Forest ML model
        - 📊 **Interactive Analytics** with visual insights
        - 🔥 **Priority Leads** identification (Hot/Warm/Cold)
        - 📈 **Performance Metrics** and model evaluation
        - 💾 **Easy Export** to CSV or Excel
        
        #### 🚀 How to Use:
        1. Use the default dataset or upload your own Excel file
        2. Click "Train Model & Score Leads" in the sidebar
        3. Explore results in different tabs
        4. Export scored leads for your CRM
        
        #### 📋 Required Data Format:
        Your Excel file should contain columns like:
        - `lead_id`, `name` (identifiers)
        - `budget_min`, `budget_max` or `budget` (numerical)
        - `preferred_area` (text)
        - `views_count`, `saved_properties` (engagement metrics)
        - `converted` (optional: 1/0 for supervised learning)
        """)

if __name__ == "__main__":
    main()
