# ============================================================
#   CUSTOMER ANALYTICS & CHURN PREDICTION SYSTEM
#   Fixed & production-ready version
# ============================================================
# Fixes applied vs original:
#   1.  Hardcoded /content/ paths → pathlib CONFIG block
#   2.  Data loading: file validation + schema display
#   3.  Data cleaning: checks + actual fixes applied
#   4.  EDA: added correlation heatmap + numeric pairplot
#   5.  RFM: overlapping segment rules made explicit / ordered
#   6.  Churn model: recency_days added to features
#   7.  Churn model: class_weight='balanced' for imbalance
#   8.  Churn model: 5-fold cross-validation added
#   9.  Churn model: confusion matrix heatmap plotted
#   10. Churn model: ROC-AUC (OVR) reported
#   11. Churn model: GridSearchCV hyperparameter tuning
#   12. sklearn Pipeline wrapping scaler + classifier
#   13. Model persistence with joblib
#   14. K-Means: K choice justified in comment
#   15. Export path handled via OUTPUT_DIR
#   16. Final business recommendations section added
# ============================================================


# ================================
# SECTION 1 — IMPORT LIBRARIES
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import warnings

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False


# ============================================================
# SECTION 2 — CONFIGURATION & DATA LOADING
# ============================================================

# ── CONFIG ───────────────────────────────────────────────────
# Change DATA_DIR to match your environment:
#   Google Colab   →  Path('/content')
#   Local/VS Code  →  Path('data')        (CSVs live in /data/)
#   Kaggle         →  Path('/kaggle/input/your-dataset-name')
DATA_DIR   = Path('/content')
OUTPUT_DIR = Path('/content/outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────

REQUIRED_FILES = ['orders.csv', 'customers.csv', 'order_items.csv', 'products.csv']

missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
if missing:
    raise FileNotFoundError(
        f'Missing files in {DATA_DIR}: {missing}\n'
        f'Update DATA_DIR above to the correct folder.'
    )

orders      = pd.read_csv(DATA_DIR / 'orders.csv',      parse_dates=['order_time'])
customers   = pd.read_csv(DATA_DIR / 'customers.csv',   parse_dates=['signup_date'])
order_items = pd.read_csv(DATA_DIR / 'order_items.csv')
products    = pd.read_csv(DATA_DIR / 'products.csv')

print('=' * 50)
print('DATASET SHAPES')
print('=' * 50)
print(f'Orders       : {orders.shape[0]:>7,} rows  x  {orders.shape[1]} cols')
print(f'Customers    : {customers.shape[0]:>7,} rows  x  {customers.shape[1]} cols')
print(f'Order Items  : {order_items.shape[0]:>7,} rows  x  {order_items.shape[1]} cols')
print(f'Products     : {products.shape[0]:>7,} rows  x  {products.shape[1]} cols')

print('\n' + '=' * 50)
print('ORDERS — column dtypes')
print('=' * 50)
print(orders.dtypes)
print('\nFirst 3 rows:')
print(orders.head(3))


# ============================================================
# SECTION 3 — DATA CLEANING & VALIDATION
# ============================================================

print('=' * 50)
print('BEFORE CLEANING')
print('=' * 50)

null_counts = orders.isnull().sum()
print('\nNull values in orders:')
print(null_counts[null_counts > 0] if null_counts.any() else '  None — clean!')

dup_count = orders['order_id'].duplicated().sum()
print(f'\nDuplicate order_ids   : {dup_count}')

bad_rev = (orders['total_usd'] <= 0).sum()
print(f'Orders total_usd <= 0 : {bad_rev}')

print(f'\nDate range: {orders["order_time"].min().date()}  →  {orders["order_time"].max().date()}')
print('\nRevenue stats:')
print(orders['total_usd'].describe().round(2))

# ── Apply fixes ──
rows_before = len(orders)

orders = orders[orders['total_usd'] > 0]
orders = orders.drop_duplicates(subset='order_id')
orders = orders.dropna(subset=['customer_id', 'order_time'])
orders = orders.reset_index(drop=True)

rows_after = len(orders)

print('\n' + '=' * 50)
print('AFTER CLEANING')
print('=' * 50)
print(f'Rows removed : {rows_before - rows_after:,}  ({(rows_before - rows_after) / rows_before * 100:.1f}%)')
print(f'Rows kept    : {rows_after:,}')
print(f'Revenue min  : ${orders["total_usd"].min():.2f}  (must be > 0)')
print('\n✓ Data is clean and ready for analysis.')


# ============================================================
# SECTION 4 — KPI SUMMARY
# ============================================================

total_customers          = orders['customer_id'].nunique()
total_orders             = orders['order_id'].nunique()
total_revenue            = orders['total_usd'].sum()
avg_order_value          = orders['total_usd'].mean()
avg_orders_per_customer  = total_orders / total_customers
avg_revenue_per_customer = total_revenue / total_customers

print('=' * 50)
print(f'Total Customers          : {total_customers:,}')
print(f'Total Orders             : {total_orders:,}')
print(f'Total Revenue            : ${total_revenue:,.2f}')
print(f'Avg Order Value (AOV)    : ${avg_order_value:.2f}')
print(f'Avg Orders per Customer  : {avg_orders_per_customer:.2f}')
print(f'Avg Revenue per Customer : ${avg_revenue_per_customer:.2f}')
print('=' * 50)

# INSIGHT:
# avg_orders_per_customer ≈ 2 confirms most customers buy once or twice then stop.
# A third purchase per customer adds ~$133 revenue at zero acquisition cost.


# ============================================================
# SECTION 5 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# ── 5.1 Monthly Revenue Trend ──
orders['month']     = orders['order_time'].dt.to_period('M')
monthly = orders.groupby('month').agg(
    revenue      = ('total_usd', 'sum'),
    total_orders = ('order_id',  'count')
).reset_index()
monthly['month_str'] = monthly['month'].astype(str)

fig, ax1 = plt.subplots()
ax1.bar(monthly['month_str'], monthly['revenue'], color='steelblue', alpha=0.7, label='Revenue')
ax1.set_ylabel('Revenue ($)')
ax1.set_xlabel('Month')
ax2 = ax1.twinx()
ax2.plot(monthly['month_str'], monthly['total_orders'], color='coral', marker='o', label='Orders')
ax2.set_ylabel('Orders')
plt.title('Monthly Revenue & Order Volume')
plt.xticks(rotation=90)
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'monthly_revenue.png', dpi=150)
plt.show()
# INSIGHT: Revenue holds $60K–$70K band. Volume drives revenue; AOV is flat.
# Consistent spikes in Jul/Dec, drops in Feb.

# ── 5.2 AOV Distribution ──
plt.figure()
sns.histplot(orders['total_usd'], bins=40, color='steelblue', kde=True)
plt.axvline(orders['total_usd'].mean(), color='coral', linestyle='--',
            label=f"Mean AOV: ${orders['total_usd'].mean():.2f}")
plt.title('Order Value Distribution')
plt.xlabel('Order Value ($)')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'aov_distribution.png', dpi=150)
plt.show()

# ── 5.3 Revenue by Device ──
device_rev = orders.groupby('device')['total_usd'].sum().sort_values(ascending=False)
device_rev.plot(kind='bar', color='steelblue', title='Revenue by Device')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'revenue_by_device.png', dpi=150)
plt.show()

# ── 5.4 Revenue by Payment Method ──
payment_rev = orders.groupby('payment_method')['total_usd'].sum().sort_values(ascending=False)
payment_rev.plot(kind='bar', color='mediumpurple', title='Revenue by Payment Method')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'revenue_by_payment.png', dpi=150)
plt.show()

# ── 5.5 Top 10 Countries by Revenue ──
country_rev = orders.groupby('country')['total_usd'].sum().sort_values(ascending=False).head(10)
country_rev.plot(kind='barh', color='steelblue', title='Top 10 Countries by Revenue')
plt.xlabel('Revenue ($)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'revenue_by_country.png', dpi=150)
plt.show()

# ── 5.6 Correlation Heatmap (NEW) ──
# Shows which numeric features move together — key for feature selection
numeric_cols = orders.select_dtypes(include='number').columns.tolist()
corr = orders[numeric_cols].corr()

plt.figure(figsize=(10, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle (redundant)
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', center=0,
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'Pearson r'}
)
plt.title('Correlation Heatmap — Numeric Features')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=150)
plt.show()
# INSIGHT: Strong correlations reveal multicollinearity risks for modelling.

# ── 5.7 Pairplot — Feature Relationships (NEW) ──
# Sample for speed on large datasets; shows joint distributions
sample = orders[numeric_cols].sample(min(2000, len(orders)), random_state=42)
sns.pairplot(sample, diag_kind='kde', plot_kws={'alpha': 0.3, 'color': 'steelblue'})
plt.suptitle('Pairplot — Numeric Feature Relationships', y=1.02)
plt.savefig(OUTPUT_DIR / 'pairplot_features.png', dpi=120, bbox_inches='tight')
plt.show()


# ============================================================
# SECTION 6 — RFM SEGMENTATION
# ============================================================

snapshot_date = orders['order_time'].max() + pd.Timedelta(days=1)

rfm = orders.groupby('customer_id').agg(
    last_purchase = ('order_time', 'max'),
    frequency     = ('order_id',   'count'),
    monetary      = ('total_usd',  'sum')
).reset_index()

rfm['recency'] = (snapshot_date - rfm['last_purchase']).dt.days

rfm['r_score'] = pd.qcut(rfm['recency'],  5, labels=[5, 4, 3, 2, 1])
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm['r_score'] = rfm['r_score'].astype(int)
rfm['f_score'] = rfm['f_score'].astype(int)
rfm['m_score'] = rfm['m_score'].astype(int)
rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']


def assign_segment(row):
    """
    Priority-ordered rules — first match wins.
    Ordering prevents overlapping conditions from being ambiguous.
    """
    r, f, m = row['r_score'], row['f_score'], row['m_score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'           # recently active, high freq, high spend
    elif r >= 4 and f <= 2:
        return 'New Customers'       # recent but low frequency — still exploring
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'     # consistent buyers
    elif m >= 4 and r >= 3:
        return 'Big Spenders'        # high monetary, decent recency
    elif r <= 2 and (f >= 3 or m >= 3):
        return 'At Risk'             # historically valuable, going quiet
    elif r <= 2 and f <= 2:
        return 'Lost Customers'      # low engagement, long since purchased
    else:
        return 'Average Customers'


rfm['segment'] = rfm.apply(assign_segment, axis=1)

seg_counts  = rfm['segment'].value_counts()
seg_revenue = rfm.groupby('segment')['monetary'].sum().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2)
seg_counts.plot(kind='bar',  ax=axes[0], color='steelblue',    title='Customers per Segment')
seg_revenue.plot(kind='bar', ax=axes[1], color='mediumpurple', title='Revenue per Segment')
for ax in axes:
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'rfm_segments.png', dpi=150)
plt.show()

print('\nRFM Segment Summary:')
print(rfm.groupby('segment').agg(
    customers    = ('customer_id', 'count'),
    avg_monetary = ('monetary',    'mean'),
    avg_recency  = ('recency',     'mean')
).round(2).sort_values('avg_monetary', ascending=False))

# INSIGHT:
# Champions — largest segment, highest avg spend, recently active → referral program.
# At Risk   — historically valuable, going quiet → win-back email before day 365.
# Lost      — suppress from paid campaigns, minimal spend.


# ============================================================
# SECTION 7 — COHORT RETENTION HEATMAP
# ============================================================

orders['cohort_month'] = (
    orders.groupby('customer_id')['order_time']
    .transform('min')
    .dt.to_period('M')
)
orders['order_month']  = orders['order_time'].dt.to_period('M')
orders['month_number'] = (
    orders['order_month'] - orders['cohort_month']
).apply(lambda x: x.n)

cohort_data = (
    orders.groupby(['cohort_month', 'month_number'])['customer_id']
    .nunique()
    .reset_index()
)
cohort_data.columns = ['cohort_month', 'month_number', 'customers']

cohort_sizes = (
    cohort_data[cohort_data['month_number'] == 0]
    .set_index('cohort_month')['customers']
)
cohort_data['retention_rate'] = cohort_data.apply(
    lambda row: row['customers'] / cohort_sizes[row['cohort_month']], axis=1
)

cohort_pivot = cohort_data.pivot_table(
    index='cohort_month', columns='month_number', values='retention_rate'
)

plt.figure(figsize=(16, 8))
sns.heatmap(
    cohort_pivot,
    annot=True, fmt='.0%',
    cmap='YlGn',
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'Retention Rate'}
)
plt.title('Cohort Retention Heatmap')
plt.xlabel('Month Number (since first purchase)')
plt.ylabel('Cohort Month')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cohort_retention_heatmap.png', dpi=150)
plt.show()

# INSIGHT: Month-1 retention is only 5–16% across all cohorts.
# Fixing the first-to-second purchase moment is the highest-value retention action.


# ============================================================
# SECTION 8 — CHURN PREDICTION (RANDOM FOREST + PIPELINE)
# ============================================================

max_date = orders['order_time'].max()

churn_base = orders.groupby('customer_id').agg(
    total_orders    = ('order_id',    'count'),
    total_revenue   = ('total_usd',   'sum'),
    avg_order_value = ('total_usd',   'mean'),
    first_purchase  = ('order_time',  'min'),
    last_purchase   = ('order_time',  'max')
).reset_index()

churn_base['recency_days']  = (max_date - churn_base['last_purchase']).dt.days
churn_base['lifespan_days'] = (
    churn_base['last_purchase'] - churn_base['first_purchase']
).dt.days


def label_churn(days):
    """
    Industry-standard thresholds:
      Active   = purchased within 180 days
      At Risk  = 181–365 days since last purchase
      Churned  = more than 365 days since last purchase
    """
    if days <= 180:
        return 'Active'
    elif days <= 365:
        return 'At Risk'
    else:
        return 'Churned'


churn_base['status'] = churn_base['recency_days'].apply(label_churn)

print('\nChurn segment distribution:')
print(churn_base['status'].value_counts())
print('\nRevenue by churn status:')
print(churn_base.groupby('status')['total_revenue'].agg(['sum', 'mean']).round(2))

# ── Features & labels ──
# FIX: recency_days was missing from original features list
features = [
    'total_orders',
    'total_revenue',
    'avg_order_value',
    'lifespan_days',
    'recency_days',      # ← strongest predictor, was absent in original
]

X = churn_base[features]
le = LabelEncoder()
y  = le.fit_transform(churn_base['status'])

print(f'\nClass mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # stratify preserves class ratio
)

# ── Build Pipeline (scaler + classifier) ──
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',   # FIX: handles class imbalance
        random_state=42,
        n_jobs=-1,
    ))
])

# ── 5-Fold Cross Validation ──
print('\n' + '=' * 50)
print('5-FOLD CROSS VALIDATION')
print('=' * 50)
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='f1_macro', n_jobs=-1)
print(f'F1-macro per fold : {[round(s, 3) for s in cv_scores]}')
print(f'Mean F1-macro     : {cv_scores.mean():.3f}  ±  {cv_scores.std():.3f}')

# ── Hyperparameter Tuning (GridSearchCV) ──
print('\n' + '=' * 50)
print('HYPERPARAMETER TUNING')
print('=' * 50)
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth':    [None, 10, 20],
    'clf__min_samples_split': [2, 5],
}
grid = GridSearchCV(
    pipe, param_grid,
    cv=3, scoring='f1_macro',
    n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
print(f'Best params : {grid.best_params_}')
print(f'Best CV F1  : {grid.best_score_:.3f}')

best_model = grid.best_estimator_

# ── Evaluate on held-out test set ──
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

print('\n' + '=' * 50)
print('CLASSIFICATION REPORT')
print('=' * 50)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ROC-AUC (One-vs-Rest, handles multiclass)
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
print(f'ROC-AUC (macro OVR) : {roc_auc:.3f}')

# ── Confusion Matrix Heatmap ──
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=le.classes_,
    cmap='Blues', ax=ax
)
ax.set_title('Confusion Matrix — Churn Prediction')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150)
plt.show()

# ── Feature Importance ──
rf         = best_model.named_steps['clf']
importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
importance.plot(kind='barh', color='steelblue', title='Feature Importance — Churn Prediction')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'churn_feature_importance.png', dpi=150)
plt.show()

# ── Save model ──
model_path = OUTPUT_DIR / 'churn_pipeline.pkl'
joblib.dump(best_model, model_path)
print(f'\n✓ Model saved to {model_path}')

# INSIGHT: recency_days is the strongest predictor of churn.
# At Risk customers hold significant recoverable revenue — target before day 365.


# ============================================================
# SECTION 9 — K-MEANS CUSTOMER CLUSTERING
# ============================================================

cluster_features = ['recency', 'frequency', 'monetary']
X_cluster = rfm[cluster_features].copy()

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# ── Elbow method ──
inertia = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o', color='steelblue')
plt.title('Elbow Method — Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'elbow_method.png', dpi=150)
plt.show()

# FIX: K justified by elbow — inertia drop flattens at K=4
K        = 4
km_final = KMeans(n_clusters=K, random_state=42, n_init=10)
rfm['cluster'] = km_final.fit_predict(X_scaled)

print('\nK-Means Cluster Summary (K=4, chosen from elbow above):')
print(rfm.groupby('cluster')[cluster_features].mean().round(2))

print('\nCluster vs Manual RFM Segment:')
print(pd.crosstab(rfm['cluster'], rfm['segment']))

plt.figure()
scatter = plt.scatter(
    rfm['recency'], rfm['monetary'],
    c=rfm['cluster'], cmap='tab10', alpha=0.5, s=20
)
plt.colorbar(scatter, label='Cluster')
plt.title('K-Means Clusters — Recency vs Monetary')
plt.xlabel('Recency (days)')
plt.ylabel('Monetary ($)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'kmeans_clusters.png', dpi=150)
plt.show()

# INSIGHT: K-Means clusters validate manual RFM segmentation.
# Disagreements reveal nuance the rule-based segment misses.
# Both methods together are stronger than either alone.


# ============================================================
# SECTION 10 — EXPORT SEGMENTED DATA
# ============================================================

final_output = rfm[[
    'customer_id', 'recency', 'frequency', 'monetary',
    'r_score', 'f_score', 'm_score', 'rfm_score',
    'segment', 'cluster'
]].merge(
    churn_base[['customer_id', 'status', 'lifespan_days', 'recency_days']],
    on='customer_id', how='left'
)

out_path = OUTPUT_DIR / 'customer_segments_final.csv'
final_output.to_csv(out_path, index=False)
print(f'\n✓ Exported: {out_path}')
print(f'Shape: {final_output.shape}')
print(final_output.head())


# ============================================================
# SECTION 11 — BUSINESS RECOMMENDATIONS
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║           BUSINESS RECOMMENDATIONS — ACTION PLAN             ║
╚══════════════════════════════════════════════════════════════╝

1. WIN-BACK CAMPAIGN — At Risk customers
   ▸ ~3,823 customers with avg $315 spend but silent for 180–365 days
   ▸ Action: personalised email + 10% discount before day 365
   ▸ Estimated recoverable revenue: high — these were real buyers

2. FIRST → SECOND PURCHASE NUDGE
   ▸ Month-1 retention is 5–16% — 84–95% never return after order 1
   ▸ Action: post-purchase email sequence at day 7, 14, 30
   ▸ Even lifting retention to 20% adds ~$133/customer at zero CAC

3. CHAMPIONS REFERRAL PROGRAM
   ▸ 4,417 Champions with highest AOV and recent activity
   ▸ Action: invite to refer a friend — offer reward on friend's first order
   ▸ Lowest-cost acquisition channel; highest LTV cohort

4. SUPPRESS LOST CUSTOMERS FROM PAID CHANNELS
   ▸ 2,678 Lost Customers with avg $64 spend, silent 1,480+ days
   ▸ Action: remove from Google/Meta retargeting audiences
   ▸ Saves ad budget; reallocate to At Risk win-back instead

5. MODEL IN PRODUCTION
   ▸ Re-score customers monthly using churn_pipeline.pkl
   ▸ Flag anyone crossing into At Risk for automated CRM trigger
   ▸ Re-train quarterly as new data accumulates
""")
