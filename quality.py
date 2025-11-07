# quality.py
import pandas as pd, numpy as np
from collections import defaultdict

def normalize_category(x):
    if pd.isna(x): return x
    s = str(x).strip()
    return " ".join(s.split()).lower()

def is_datetime_series(s: pd.Series) -> bool:
    nonnull = s.dropna()
    if nonnull.empty: return False
    sample = nonnull.astype(str).head(50)
    return pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.8

def coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0: return pd.Series(False, index=series.index)
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return (series < lower) | (series > upper)

def percentile_outlier_mask_dt(series_dt: pd.Series, pct: float = 1.0) -> pd.Series:
    s = coerce_datetime(series_dt.copy())
    s_nonnull = s.dropna()
    if s_nonnull.empty or len(s_nonnull) < 10:
        return pd.Series(False, index=series_dt.index)
    q_low  = s_nonnull.quantile(pct/100.0)
    q_high = s_nonnull.quantile(1 - pct/100.0)
    return ((s < q_low) | (s > q_high)).fillna(False)

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0: return pd.DataFrame(columns=["Missing Count","Missing %"])
    miss = df.isna().sum()
    pct  = (miss/len(df)*100).round(2)
    out  = pd.DataFrame({"Missing Count": miss, "Missing %": pct})
    return out[out["Missing Count"]>0].sort_values("Missing %", ascending=False)

def extract_duplicates(df: pd.DataFrame):
    dup_mask = df.duplicated(keep=False)
    dups = df.loc[dup_mask].copy()
    if dups.empty:
        return (pd.DataFrame({"Duplicate Rows":[0],"Duplicate %":[0.0]}),
                pd.DataFrame(columns=["_dup_group_id","Group Size"]))
    key = df.astype(object).apply(lambda r: tuple(r.values.tolist()), axis=1)
    dup_key = key[dup_mask]
    dups["_dup_group_id"] = dup_key.factorize()[0] + 1
    summary = pd.DataFrame({
        "Duplicate Rows":[int(dup_mask.sum())],
        "Duplicate %":[round(dup_mask.mean()*100,2)],
        "Distinct Duplicate Groups":[int(dups["_dup_group_id"].nunique())],
    })
    sizes = dups["_dup_group_id"].value_counts().rename("Group Size").reset_index()
    sizes = sizes.rename(columns={"index":"_dup_group_id"}).sort_values("Group Size", ascending=False)
    return summary, sizes

def extract_outliers(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number])
    dt_cols = [c for c in df.columns if is_datetime_series(df[c])]
    counts, any_mask, flagged = {}, pd.Series(False, index=df.index), defaultdict(list)
    for col in numeric.columns:
        m = iqr_outlier_mask(numeric[col]); counts[col] = int(m.sum()); any_mask |= m
        for i in m[m].index: flagged[i].append(col)
    for col in dt_cols:
        m = percentile_outlier_mask_dt(df[col]); counts[col] = counts.get(col,0)+int(m.sum()); any_mask |= m
        for i in m[m].index: flagged[i].append(col)
    counts_df = (pd.DataFrame.from_dict(counts, orient="index", columns=["Outlier Count"])
                 .sort_values("Outlier Count", ascending=False)
                 if counts else pd.DataFrame(columns=["Outlier Count"]))
    if any_mask.any():
        rows = df.loc[any_mask].copy()
        rows["flagged_outlier_in"] = rows.index.map(lambda i: ", ".join(flagged.get(i, [])))
    else:
        rows = pd.DataFrame()
    return counts_df, rows

def find_label_issues(df: pd.DataFrame, rare_pct: float = 1.0) -> pd.DataFrame:
    issues, obj_cols = [], df.select_dtypes(include=["object","category"]).columns.tolist()
    for col in obj_cols:
        s = df[col]
        if s.dropna().empty: continue
        norm = s.map(normalize_category)
        mapping = pd.DataFrame({"raw": s, "norm": norm})
        variants = mapping.groupby("norm", dropna=False)["raw"].nunique()
        for norm_val, n_raw in variants[variants>1].items():
            samples = mapping.loc[mapping["norm"]==norm_val,"raw"].dropna().astype(str).value_counts().head(8).to_dict()
            issues.append({"Column":col,"Issue":"Case/Whitespace variants","Detail":str(samples)})
        stray = s.astype(str).str.match(r".*^\s+|\s+$|  +.*", na=False)
        if stray.any():
            examples = s[stray].astype(str).value_counts().head(8).to_dict()
            issues.append({"Column":col,"Issue":"Stray/double spaces","Detail":str(examples)})
        numeric_coerce = pd.to_numeric(s, errors="coerce")
        mixed_ratio = (~numeric_coerce.isna()).mean()
        if 0.05 < mixed_ratio < 0.95:
            examples = s.astype(str).value_counts().head(8).to_dict()
            issues.append({"Column":col,"Issue":"Mixed types (numeric/text)","Detail":str(examples)})
        vc = norm.value_counts(dropna=True)
        if len(vc)>0:
            total = len(norm.dropna())
            rare = vc[vc/total*100 < rare_pct]
            if not rare.empty:
                rare_examples = mapping[mapping["norm"].isin(rare.index)]["raw"].astype(str).value_counts().head(8).to_dict()
                issues.append({"Column":col,"Issue":f"Rare categories (<{rare_pct}%)","Detail":str(rare_examples)})
    return pd.DataFrame(issues, columns=["Column","Issue","Detail"]).fillna("")


    # --- HTML helpers for the standalone report ---
from html import escape
from pathlib import Path
import pandas as pd

RARE_CAT_PCT = 1.0  # used in the footnote text


def df_to_html_table(df: pd.DataFrame, index: bool = False) -> str:
    """Render a dataframe using the .tbl style used in the standalone report."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        df = pd.DataFrame({"Info": ["No rows."]})
    return df.to_html(classes="tbl", border=0, index=index, escape=False)

def build_quality_html(
    *,
    name: str,
    profile_path: str,
    overview: dict,
    missing_df: pd.DataFrame,
    dup_summary_df: pd.DataFrame,
    dup_groups_df: pd.DataFrame,
    out_counts_df: pd.DataFrame,
    out_rows_df: pd.DataFrame,
    label_issues_df: pd.DataFrame,
) -> str:
    """Return a complete HTML document (string) that matches the style you sent."""

from html import escape
from pathlib import Path
import pandas as pd

RARE_CAT_PCT = 1.0  # used in the footnote text

def df_to_html_table(df: pd.DataFrame, index: bool = False) -> str:
    """Render a dataframe using the .tbl style used in the standalone report."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        df = pd.DataFrame({"Info": ["No rows."]})
    return df.to_html(classes="tbl", border=0, index=index, escape=False)

def build_quality_html(name, profile_path, overview,
                       missing_df, dup_summary_df,
                       dup_groups_df, out_counts_df,
                       out_rows_df, label_issues_df,
                       RARE_CAT_PCT=1.0):
    """Return a complete HTML document (string) for the quality report."""

    def df_to_html_table(df, index=True):
        try:
            return df.to_html(classes="tbl", border=0, index=index)
        except Exception:
            return "<p class='note'>(No data)</p>"

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{escape(name)} — Enhanced Data Quality Report</title>
<style>
/* your CSS as before */
</style>
</head>

<body>
  <div class="header-bar">
    <div class="header-left">
      <div class="logo">🔷 Crystal LAB</div>
      <div class="file-info">
        <div class="filename">{escape(name)}</div>
        <div class="subtitle">Data Profile Report</div>
      </div>
    </div>
    <div class="header-right">
      <div class="user">Haifa Moha</div>
      <button class="quality-btn">⚠️ Quality issues</button>
    </div>
  </div>

  <div class="content">
    <h1>{escape(name)} — Enhanced Data Quality Report</h1>
    <p>
      <a class="report-link" href="{escape(profile_path)}" target="_blank">Open ydata_profiling dashboard</a>
      <span class="badge">HTML</span>
    </p>
    <div class="hr"></div>

    <h2>Overview</h2>
    <p class="note">Rows: {overview.get('rows','-')} &nbsp;&nbsp; Columns: {overview.get('cols','-')}</p>

    <h2>1) Missing Values</h2>
    {df_to_html_table(missing_df)}

    <h2>2) Duplicate Rows</h2>
    <h3>Summary</h3>
    {df_to_html_table(dup_summary_df, index=False)}
    <h3>Largest Duplicate Groups</h3>
    {df_to_html_table(dup_groups_df, index=False)}

    <h2>3) Outliers</h2>
    <h3>Per-column Outlier Counts</h3>
    {df_to_html_table(out_counts_df)}
    <h3>Rows Flagged as Outliers (any column)</h3>
    {df_to_html_table(out_rows_df, index=True)}

    <h2>4) Labeling Issues (Categorical Heuristics)</h2>
    {df_to_html_table(label_issues_df, index=False)}

    <div class="hr"></div>
    <p class="note">
      Notes: Outliers use IQR for numeric columns and top/bottom 1% tails for datetimes.
      Labeling issues include case/whitespace variants, stray spaces, mixed types, and very rare categories (&lt; {RARE_CAT_PCT}%).
    </p>
  </div>
</body>
</html>"""
    return html
