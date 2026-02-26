# 🏠 Airbnb Price Prediction & Market Analysis

> **Live Demo:** [https://your-site-name.netlify.app](https://your-site-name.netlify.app) _(update with your actual Netlify URL after deployment)_

![Dashboard](https://img.shields.io/badge/Dashboard-Interactive-FF5A5F?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-R²%200.8799-00A699?style=for-the-badge)
![Deploy](https://img.shields.io/badge/Deployed-Netlify-00C7B7?style=for-the-badge)

---

## 📌 Project Overview

This project analyzes **494,954 Airbnb listings** (sourced from Kaggle via [joebeachcapital/airbnb](https://www.kaggle.com/datasets/joebeachcapital/airbnb)) to predict nightly listing prices and uncover market dynamics using machine learning.

- **Linear Regression** for price prediction — **R² = 0.8799, RMSE = $50.40**
- **K-Means Clustering** (K = 3) for market segmentation
- **Interactive web dashboard** deployed on Netlify with in-browser predictions using exact trained coefficients

---

## 📊 Exact Model Results

All values below were extracted by running the full pipeline on the dataset via `extract_model.py` and exported to `model_data.json`.

### Model Performance

| Metric | Exact Value |
|---|---|
| **R² Score** | 0.8799 |
| **RMSE** | $50.40 |
| **Intercept** | 217.7575 |
| **Raw Dataset Rows** | 494,954 |
| **After Deduplication** | 494,952 |
| **After NaN Removal** | 360,644 |
| **Training Samples (80%)** | 288,515 |
| **Test Samples (20%)** | 72,129 |
| **Total Features (after encoding)** | 2,349 |
| **Numerical Features** | 11 |
| **Room Type Categories** | 3 |
| **Unique Neighbourhoods** | 2,335 |
| **K-Means Clusters** | 3 (`random_state=42`, `n_init=10`) |

### Price Statistics

| Stat | Value |
|---|---|
| Mean | $138.07 |
| Median | $90.00 |
| Std Dev | $145.48 |

### Exact Numerical Feature Coefficients

| Feature | Coefficient | Mean | Std Dev |
|---|---|---|---|
| Price per Bedroom | +99.3079 | 99.4818 | 107.8403 |
| Bedrooms | +42.6715 | 1.3357 | 0.8743 |
| Accommodates | +11.1565 | 3.2985 | 2.0111 |
| Bathrooms | +9.3944 | 1.2202 | 0.5416 |
| Availability 365 | +2.7395 | 173.2570 | 136.7705 |
| Review Scores Rating | +1.5790 | 92.8982 | 8.5508 |
| Booking Density | +0.8931 | 6.1417 | 9.8640 |
| Review Month | +0.3292 | 5.1491 | 3.1680 |
| Beds | −1.9397 | 1.9050 | 1.4293 |
| Number of Reviews | −1.8881 | 22.6037 | 35.7939 |
| Cluster | −8.5854 | 1.7381 | 0.5329 |

### Room Type Coefficients

| Room Type | Coefficient | Mean Price | Median Price | Count |
|---|---|---|---|---|
| Entire home/apt | −87.6389 | $173.43 | $120.00 | 306,113 |
| Private room | −100.7526 | $79.16 | $55.00 | 171,965 |
| Shared room | −111.4595 | $60.50 | $37.00 | 8,915 |

> Note: All room type coefficients are negative because they are relative to the intercept ($217.76). Effective price = intercept + room_type_coef + other features.

### K-Means Cluster Characteristics (on 360,764 rows)

| Cluster | Count | Avg Price | Avg Accommodates | Avg Bedrooms | Avg Beds | Avg Review Score | Avg Price/Bedroom |
|---|---|---|---|---|---|---|---|
| **0 (Luxury)** | 16,351 | $613.88 | 3.06 | 1.27 | 1.68 | 93.52 | $508.86 |
| **1 (Mid-range)** | 61,840 | $220.79 | 6.48 | 2.72 | 4.10 | 92.89 | $83.41 |
| **2 (Budget)** | 282,573 | $87.66 | 2.62 | 1.04 | 1.44 | 92.86 | $79.29 |

### Top 10 Neighbourhoods by Average Price

| Rank | Neighbourhood | Avg Price |
|---|---|---|
| 1 | Indre By | $692.63 |
| 2 | Hiller Highlands | $650.00 |
| 3 | Southern | $643.26 |
| 4 | Vesterbro-Kongens Enghave | $635.53 |
| 5 | Frederiksberg | $620.50 |
| 6 | Amager Vest | $615.67 |
| 7 | Central & Western | $600.43 |
| 8 | Islands | $595.63 |
| 9 | Østerbro | $584.93 |
| 10 | Nørrebro | $582.94 |

---

## 🚀 Deployed Application Features

### 📊 Interactive Dashboard
- Price distribution histogram
- Top 10 regions by average price (exact values from data)
- Price distribution by room type (box plots)
- Price vs. Number of Reviews scatter plot
- Price vs. Availability scatter plot
- K-Means Clustering visualization (3 clusters)
- Feature importance chart (exact trained coefficients)

### 🔮 Prediction Interface
Users input listing details and receive an **instant price prediction** using the exact trained model coefficients embedded in JavaScript.

### 💡 Business Insights
- Coefficient analysis with exact values
- Cluster segment characteristics (exact means)
- Revenue optimization strategies
- Risk analysis: overpricing vs. underpricing

---

## 🛠️ Methodology

### Step 1: Data Acquisition & Cleaning
- **494,954 raw rows** → **494,952 after deduplication** (by `ID`)
- Cleaned 6 currency columns, converted `Host Response Rate`, parsed 4 date columns
- Semicolon-delimited CSV loaded with `on_bad_lines='skip'`

### Step 2: Feature Engineering
- **Price per Bedroom** = `Price / max(Bedrooms, 1)`
- **Review Month** = month from `Last Review` date
- **Booking Density** = `(Number of Reviews / days_active) × 30.44`

### Step 3: K-Means Clustering
- 9 features scaled with `StandardScaler`
- Elbow method → K=3
- **360,764 rows** after NaN drop for clustering
- `KMeans(n_clusters=3, random_state=42, n_init=10)`

### Step 4: Linear Regression
- 13 selected features → **2,349 after one-hot encoding** (3 Room Types + 2,335 Neighbourhoods)
- `StandardScaler` on numerical columns
- 80/20 split: **288,515 train / 72,129 test** (`random_state=42`)
- `sklearn.linear_model.LinearRegression`
- **R² = 0.8799 | RMSE = $50.40**

### Step 5: Deployment
- Exact coefficients extracted via `extract_model.py` → `model_data.json`
- Coefficients embedded in `site/index.html` JavaScript
- Static site deployed on Netlify

---

## 📁 Repository Structure

```
Business_Applied_AI/
├── site/
│   └── index.html               # Interactive dashboard (deployed to Netlify)
├── Business_Applied_AI.ipynb    # Jupyter notebook (full analysis)
├── extract_model.py             # Script to extract exact model coefficients
├── model_data.json              # Exact extracted values (R², RMSE, coefficients, etc.)
├── netlify.toml                 # Netlify deployment configuration
└── README.md                    # This file
```

---

## 🌐 Step-by-Step Deployment on Netlify

This is a **real production deployment** on Netlify's CDN. Your site will be live on the internet with a real URL. Netlify serves static files — no server needed.

### What Gets Deployed

Only the `site/` folder (containing `index.html`) gets deployed. The notebook, Python scripts, and data files stay in your repo but are not served. The `netlify.toml` file tells Netlify which folder to publish.

---

### Option A: Deploy via GitHub (Recommended — Auto-Deploys on Push)

**Step 1: Push to GitHub**
```bash
cd /Users/shivammishra/Desktop/Business_Applied_AI

# Initialize git if not already done
git init

# Add all files
git add .
git commit -m "Airbnb Price Prediction - initial commit"

# Create a new repository on GitHub (go to github.com/new)
# Then link and push:
git remote add origin https://github.com/YOUR_USERNAME/Business_Applied_AI.git
git branch -M main
git push -u origin main
```

**Step 2: Connect to Netlify**
1. Go to **[app.netlify.com](https://app.netlify.com)** and sign up / log in (you can use your GitHub account)
2. Click **"Add new site"** → **"Import an existing project"**
3. Choose **"Deploy with GitHub"**
4. Authorize Netlify to access your GitHub account (if first time)
5. Select the repository: **`Business_Applied_AI`**

**Step 3: Configure Build Settings**
Netlify will auto-detect the `netlify.toml` config. Verify these settings:

| Setting | Value |
|---|---|
| **Branch to deploy** | `main` |
| **Build command** | _(leave empty — no build needed)_ |
| **Publish directory** | `site` |

> The `netlify.toml` file in your repo already specifies `publish = "site"`, so Netlify should auto-fill this.

**Step 4: Deploy**
1. Click **"Deploy site"**
2. Wait 30–60 seconds for deployment
3. Netlify assigns a random URL like `https://random-name-12345.netlify.app`
4. Your site is now **live on the internet** ✅

**Step 5: Customize URL (Optional)**
1. Go to **Site settings** → **Domain management** → **Custom domains**
2. Click **"Options"** → **"Edit site name"**
3. Change to something like `airbnb-price-prediction` → URL becomes `https://airbnb-price-prediction.netlify.app`

**Automatic Redeployment:** Every time you `git push` to the `main` branch, Netlify automatically rebuilds and redeploys your site.

---

### Option B: Deploy via Netlify CLI (Command Line)

**Step 1: Install the CLI**
```bash
npm install -g netlify-cli
```

**Step 2: Login to Netlify**
```bash
netlify login
```
This opens a browser window — authorize the CLI.

**Step 3: Deploy (First Time)**
```bash
cd /Users/shivammishra/Desktop/Business_Applied_AI

# Create a new site and deploy
netlify deploy --prod --dir=site
```

The CLI will ask:
- **"Create & configure a new site"** → Yes
- **Team** → Select your team
- **Site name** → Enter a name (e.g., `airbnb-price-prediction`)

**Step 4: Verify**
The CLI prints the live URL. Open it in your browser.

**Redeployment:** Run the same `netlify deploy --prod --dir=site` command after any changes.

---

### Option C: Drag & Drop (Quickest — No Git Required)

1. Open **[app.netlify.com/drop](https://app.netlify.com/drop)** in your browser
2. Open Finder and navigate to `/Users/shivammishra/Desktop/Business_Applied_AI/`
3. **Drag the `site` folder** (not individual files — the entire folder) onto the Netlify drop zone
4. Wait 10 seconds — your site is live! 🎉
5. Netlify gives you a random URL immediately

> ⚠️ With drag & drop, you'll need to manually re-upload after changes. For automatic updates, use Option A (GitHub).

---

### Verifying Your Deployment

After deployment, check:
- [ ] Site loads at the Netlify URL
- [ ] Dashboard tab shows all 7 interactive charts
- [ ] Prediction tab generates prices when you click "Generate Prediction"
- [ ] Insights tab shows business analysis with exact numbers
- [ ] KPI cards show: **494,954** listings, **0.8799** R², **$50.40** RMSE

---

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3** | Data analysis & ML pipeline |
| **Pandas / NumPy** | Data cleaning, manipulation, feature engineering |
| **Scikit-learn** | `LinearRegression`, `KMeans`, `StandardScaler`, `train_test_split` |
| **SciPy** | Sparse matrices for memory-efficient model training |
| **Matplotlib / Seaborn** | EDA visualizations (notebook) |
| **KaggleHub** | Programmatic dataset download |
| **HTML / CSS / JavaScript** | Deployed web application |
| **Plotly.js** | Interactive charts in the dashboard |
| **Netlify** | Static site hosting & global CDN deployment |

---

## 👤 Author

Shivam Mishra

---

## 📜 License

This project is for educational purposes as part of Business Applied AI coursework.
