"""
Extract exact model coefficients — ultra low-memory version.
Uses scipy sparse matrices directly instead of pd.get_dummies.
"""
import kagglehub
import pandas as pd
import numpy as np
import os
import json
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import sparse

# ──────────────────────────────────────────────
# STEP 1: Load data (only needed columns)
# ──────────────────────────────────────────────
print("STEP 1: Loading data...")
path = kagglehub.dataset_download("joebeachcapital/airbnb")
files = [f for f in os.listdir(path) if f.endswith('.csv')]
source_path = os.path.join(path, files[0])

needed = ['ID','Price','First Review','Last Review',
          'Accommodates','Bathrooms','Bedrooms','Beds',
          'Review Scores Rating','Number of Reviews','Availability 365',
          'Room Type','Neighbourhood Cleansed']

df = pd.read_csv(source_path, sep=';', quotechar='"',
                 on_bad_lines='skip', engine='python', usecols=needed)
raw_rows = df.shape[0]
print(f"  Raw: {df.shape}")

# ──────────────────────────────────────────────
# STEP 2: Clean
# ──────────────────────────────────────────────
print("STEP 2: Cleaning...")
df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
dedup_rows = df.shape[0]
df.drop('ID', axis=1, inplace=True)

# Price is currency string
if df['Price'].dtype == 'object':
    df['Price'] = df['Price'].str.replace('$','',regex=False).str.replace(',','',regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

for col in ['First Review','Last Review']:
    df[col] = pd.to_datetime(df[col], errors='coerce')

print(f"  After dedup: {dedup_rows}")

# ──────────────────────────────────────────────
# STEP 3: Feature Engineering
# ──────────────────────────────────────────────
print("STEP 3: Feature engineering...")
bed_clean = df['Bedrooms'].replace(0,1).fillna(1)
df['Price per Bedroom'] = (df['Price'] / bed_clean).replace([np.inf,-np.inf], np.nan)
df['Review Month'] = df['Last Review'].dt.month
days = (df['Last Review'] - df['First Review']).dt.days.replace(0,1).fillna(1)
df['Booking Density'] = ((df['Number of Reviews'] / days) * 30.44).replace([np.inf,-np.inf], np.nan)
df.drop(['First Review','Last Review'], axis=1, inplace=True)
gc.collect()

# Price stats
price_stats = {
    'mean': round(float(df['Price'].mean()), 2),
    'median': round(float(df['Price'].median()), 2),
    'std': round(float(df['Price'].std()), 2),
    'min': round(float(df['Price'].min()), 2),
    'max': round(float(df['Price'].max()), 2),
}
print(f"  Price: mean=${price_stats['mean']}, median=${price_stats['median']}")

# ──────────────────────────────────────────────
# STEP 4: EDA
# ──────────────────────────────────────────────
print("STEP 4: EDA...")
avg_by_region = df.groupby('Neighbourhood Cleansed')['Price'].mean().sort_values(ascending=False)
top_10_regions = avg_by_region.head(10)
print("  Top 10 Regions:")
for r, p in top_10_regions.items():
    print(f"    {r}: ${p:.2f}")

rt_stats = df.groupby('Room Type')['Price'].agg(['mean','median','count']).sort_values('mean',ascending=False)
print("  Room Type Stats:")
print(rt_stats.to_string())

# ──────────────────────────────────────────────
# STEP 5: K-Means Clustering
# ──────────────────────────────────────────────
print("STEP 5: K-Means...")
clust_feats = ['Price','Accommodates','Bathrooms','Bedrooms','Beds',
               'Review Scores Rating','Number of Reviews','Price per Bedroom','Booking Density']
df_c = df[clust_feats].dropna().copy()
print(f"  Clustering rows: {len(df_c)}")

sc_km = StandardScaler()
X_km = sc_km.fit_transform(df_c).astype(np.float32)
km = KMeans(n_clusters=3, random_state=42, n_init=10)
df_c['Cluster'] = km.fit_predict(X_km)
del X_km; gc.collect()

cluster_means = df_c.groupby('Cluster')[clust_feats].mean()
cluster_counts = df_c['Cluster'].value_counts().sort_index()
print(f"  Cluster counts: {dict(cluster_counts)}")

df = df.merge(df_c[['Cluster']], left_index=True, right_index=True, how='left')
del df_c; gc.collect()

# ──────────────────────────────────────────────
# STEP 6: Regression — Build sparse matrix manually
# ──────────────────────────────────────────────
print("STEP 6: Building regression matrix...")

# Numerical features for regression
num_feats = ['Accommodates','Bathrooms','Bedrooms','Beds',
             'Review Scores Rating','Number of Reviews','Availability 365',
             'Price per Bedroom','Booking Density','Review Month','Cluster']

# Build a clean DataFrame with only what we need
reg_df = df[num_feats + ['Room Type','Neighbourhood Cleansed','Price']].copy()
del df; gc.collect()

# Drop NaN
reg_df.dropna(inplace=True)
n_rows = len(reg_df)
print(f"  Rows after NaN drop: {n_rows}")

y = reg_df['Price'].values.astype(np.float32)

# Scale numerical
scaler_reg = StandardScaler()
X_num = scaler_reg.fit_transform(reg_df[num_feats].values.astype(np.float32)).astype(np.float32)
num_means = dict(zip(num_feats, [float(m) for m in scaler_reg.mean_]))
num_stds = dict(zip(num_feats, [float(s) for s in scaler_reg.scale_]))
print(f"  Numerical block: {X_num.shape}")

# Encode categoricals manually as sparse one-hot
# Room Type
rt_le = LabelEncoder()
rt_codes = rt_le.fit_transform(reg_df['Room Type'].values)
rt_classes = list(rt_le.classes_)
n_rt = len(rt_classes)
rt_sparse = sparse.coo_matrix(
    (np.ones(n_rows, dtype=np.float32), (np.arange(n_rows), rt_codes)),
    shape=(n_rows, n_rt)
).tocsr()
print(f"  Room Type classes ({n_rt}): {rt_classes}")

# Neighbourhood Cleansed
ne_le = LabelEncoder()
ne_codes = ne_le.fit_transform(reg_df['Neighbourhood Cleansed'].values)
ne_classes = list(ne_le.classes_)
n_ne = len(ne_classes)
ne_sparse = sparse.coo_matrix(
    (np.ones(n_rows, dtype=np.float32), (np.arange(n_rows), ne_codes)),
    shape=(n_rows, n_ne)
).tocsr()
print(f"  Neighbourhood classes ({n_ne}): showing first 10: {ne_classes[:10]}...")

del reg_df; gc.collect()

# Combine: [scaled_numerical | room_type_onehot | neighbourhood_onehot]
X_num_sparse = sparse.csr_matrix(X_num)
del X_num; gc.collect()

X_all = sparse.hstack([X_num_sparse, rt_sparse, ne_sparse], format='csr')
del X_num_sparse, rt_sparse, ne_sparse; gc.collect()

all_columns = num_feats + [f'Room Type_{c}' for c in rt_classes] + [f'Neighbourhood Cleansed_{c}' for c in ne_classes]
total_features = len(all_columns)
print(f"  Final matrix: {X_all.shape}, features={total_features}, nnz={X_all.nnz}")
print(f"  Memory: ~{X_all.data.nbytes / 1024 / 1024:.0f} MB sparse data")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
train_n = X_train.shape[0]
test_n = X_test.shape[0]
print(f"  Train: {train_n} | Test: {test_n}")
del X_all, y; gc.collect()

# Train
print("  Training LinearRegression...")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = float(r2_score(y_test, y_pred))
intercept = float(model.intercept_)

print(f"\n{'='*50}")
print(f"  R²        = {r2:.4f}")
print(f"  RMSE      = ${rmse:.2f}")
print(f"  Intercept = {intercept:.4f}")
print(f"  Train     = {train_n}")
print(f"  Test      = {test_n}")
print(f"{'='*50}")

# ──────────────────────────────────────────────
# Extract all coefficients
# ──────────────────────────────────────────────
coefs = [float(c) for c in model.coef_]
coef_dict = dict(zip(all_columns, coefs))

sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nTop 30 by |coefficient|:")
for f, c in sorted_coefs[:30]:
    print(f"  {f}: {c:.4f}")

# Numerical
print("\nNumerical features:")
for f in num_feats:
    print(f"  {f}: coef={coef_dict[f]:.4f}, mean={num_means[f]:.4f}, std={num_stds[f]:.4f}")

# Room Type
room_type_coefs = {c: coef_dict[f'Room Type_{c}'] for c in rt_classes}
print("\nRoom Type:", {k: round(v,4) for k,v in room_type_coefs.items()})

# Neighbourhood (top 15)
neigh_coefs = {c: coef_dict[f'Neighbourhood Cleansed_{c}'] for c in ne_classes}
sorted_n = sorted(neigh_coefs.items(), key=lambda x: abs(x[1]), reverse=True)
top_neighs = dict(sorted_n[:15])
print("\nTop 15 Neighbourhoods:", {k: round(v,4) for k,v in top_neighs.items()})

# ──────────────────────────────────────────────
# Export JSON
# ──────────────────────────────────────────────
export = {
    "r2": round(r2, 4),
    "rmse": round(rmse, 2),
    "intercept": round(intercept, 4),
    "raw_rows": raw_rows,
    "dedup_rows": dedup_rows,
    "rows_after_nan_drop": train_n + test_n,
    "training_samples": train_n,
    "test_samples": test_n,
    "total_features_after_encoding": total_features,
    "numerical_features": [
        {"name": f, "coef": round(coef_dict[f],4),
         "mean": round(num_means[f],4), "std": round(num_stds[f],4)}
        for f in num_feats
    ],
    "room_type_coefs": {k: round(v,4) for k,v in room_type_coefs.items()},
    "top_neighbourhood_coefs": {k: round(v,4) for k,v in top_neighs.items()},
    "all_neighbourhood_coefs": {k: round(v,4) for k,v in neigh_coefs.items()},
    "cluster_coef": round(coef_dict.get('Cluster',0),4),
    "cluster_means": {
        int(cid): {f: round(float(cluster_means.loc[cid,f]),2) for f in clust_feats}
        for cid in sorted(cluster_means.index)
    },
    "cluster_counts": {int(k): int(v) for k,v in cluster_counts.items()},
    "top_10_regions": {r: round(float(p),2) for r,p in top_10_regions.items()},
    "room_type_stats": {
        rt: {"mean": round(float(rt_stats.loc[rt,'mean']),2),
             "median": round(float(rt_stats.loc[rt,'median']),2),
             "count": int(rt_stats.loc[rt,'count'])}
        for rt in rt_stats.index
    },
    "top_20_features": [
        {"feature": f, "coefficient": round(c,4)} for f,c in sorted_coefs[:20]
    ],
    "price_mean": price_stats['mean'],
    "price_median": price_stats['median'],
    "price_std": price_stats['std'],
    "price_min": price_stats['min'],
    "price_max": price_stats['max'],
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data.json")
with open(out, 'w') as f:
    json.dump(export, f, indent=2)

print(f"\n✅ Exported to: {out}")
