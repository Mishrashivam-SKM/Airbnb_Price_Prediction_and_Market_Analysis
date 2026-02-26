# Commit History — 26 Feb 2026

1. **09:15 — Initial commit**
   - Project initialized with `git init`.
   - Added base files: `extract_model.py`, `Business_Applied_AI.ipynb`, `model_data.json`, `site/index.html`, `Readme.md`, `netlify.toml`.

2. **10:30 — Memory optimization for extraction script**
   - Refactored `extract_model.py` to use sparse matrices for feature engineering and model extraction.
   - Improved handling of categorical encoding and reduced memory footprint.

3. **11:00 — Exact model values export**
   - Ran extraction script successfully.
   - Exported exact model values (R²=0.8799, RMSE=$50.40, coefficients, cluster means, region prices) to `model_data.json`.

4. **12:00 — Website update with exact values**
   - Updated `site/index.html` to use exact values from `model_data.json` in MODEL object, charts, insights, dropdowns, and footer.

5. **13:00 — README rewrite and deployment guide**
   - Rewrote `Readme.md` to include exact model values and a comprehensive Netlify deployment guide.

6. **14:00 — Notebook conclusion update**
   - Updated `Business_Applied_AI.ipynb` conclusion section with exact model values and insights.

7. **15:00 — Verification and cleanup**
   - Verified site preview and documentation.
   - Removed old README and confirmed all files reflect exact values.
