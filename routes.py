import os
from flask import session, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from app import app, db
from flask_login import current_user, login_required
from flask import Response  # add at the top with other imports
from models import Dataset
from auth import auth_bp
import pandas as pd
# REMOVED: from ydata_profiling import ProfileReport
from datetime import datetime


# NEW: helpers for quality page
from quality import (
    summarize_missing,
    extract_duplicates,
    extract_outliers,
    find_label_issues,
)

app.register_blueprint(auth_bp, url_prefix="/auth")

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def load_df(filepath: str, original_name: str) -> pd.DataFrame:
    ext = original_name.lower().rsplit('.', 1)[-1]
    if ext == 'csv':
        return pd.read_csv(filepath)
    if ext in ('xlsx', 'xls'):
        # pandas will use openpyxl for .xlsx automatically if installed
        # .xls needs xlrd installed; if you don't use .xls you can drop this
        return pd.read_excel(filepath)
    raise ValueError(f'Unsupported file format: {ext}')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def make_session_permanent():
    session.permanent = True

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('auth.login'))

@app.route('/dashboard')
@login_required
def dashboard():
    datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.uploaded_at.desc()).all()
    return render_template('dashboard.html', user=current_user, datasets=datasets)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{current_user.id}_{timestamp}_{filename}"
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            dataset = Dataset(
                user_id=current_user.id,
                filename=unique_filename,
                original_filename=filename,
                file_size=os.path.getsize(filepath)
            )
            db.session.add(dataset)
            db.session.commit()
            
            flash('File uploaded successfully!', 'success')
            return redirect(url_for('profile_dataset', dataset_id=dataset.id))
        else:
            flash('Invalid file type. Please upload CSV or Excel files.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html', user=current_user)

@app.route('/profile/<int:dataset_id>')
@login_required
def profile_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)

    if dataset.user_id != current_user.id:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))

    if not dataset.profile_generated:
        try:
            # LAZY IMPORT – only when needed
            from ydata_profiling import ProfileReport

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)

            # use the helper (handles csv/xlsx/xls)
            df = load_df(filepath, dataset.original_filename)

            profile = ProfileReport(
                df,
                title=f"Profile Report - {dataset.original_filename}",
                explorative=True
            )

            profile_filename = f"profile_{dataset.id}.html"
            profile_path = os.path.join(app.config['UPLOAD_FOLDER'], profile_filename)
            profile.to_file(profile_path)

            dataset.profile_generated = True
            dataset.profile_path = profile_filename
            db.session.commit()

            flash('Profile report generated successfully!', 'success')
        except Exception as e:
            flash(f'Error generating profile: {str(e)}', 'error')
            return redirect(url_for('dashboard'))

    return render_template('profile.html', dataset=dataset, user=current_user)


@app.route('/view_profile/<int:dataset_id>')
@login_required
def view_profile(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if dataset.user_id != current_user.id:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    if not dataset.profile_generated:
        return redirect(url_for('profile_dataset', dataset_id=dataset_id))
    
    return render_template('view_report.html', dataset=dataset, user=current_user)

@app.route('/profile_report/<int:dataset_id>')
@login_required
def profile_report(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if dataset.user_id != current_user.id:
        return "Access denied", 403
    
    if not dataset.profile_generated:
        return "Profile not generated", 404
    
    profile_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.profile_path)
    return send_file(profile_path)

@app.route('/delete/<int:dataset_id>', methods=['POST'])
@login_required
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if dataset.user_id != current_user.id:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        if dataset.profile_path:
            profile_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.profile_path)
            if os.path.exists(profile_path):
                os.remove(profile_path)
        
        db.session.delete(dataset)
        db.session.commit()
        flash('Dataset deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting dataset: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))




@app.route('/quality/<int:dataset_id>')
@login_required
def quality_issues(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != current_user.id:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
    if not os.path.exists(filepath):
        flash('File not found on server', 'error')
        return redirect(url_for('dashboard'))

    try:
        df = load_df(filepath, dataset.original_filename)
    except Exception as e:
        flash(f'Error reading file: {e}', 'error')
        return redirect(url_for('dashboard'))

    # --- compute the raw tables
    missing_df                  = summarize_missing(df)
    dup_summary_df, dup_groups  = extract_duplicates(df)
    out_counts_df, out_rows_df  = extract_outliers(df)
    labels_df                   = find_label_issues(df)

    # --- light formatting for clean headers & limits
    def _fmt_missing(d: pd.DataFrame) -> pd.DataFrame:
        if d is None or d.empty:
            return pd.DataFrame({"Info": ["No missing values found."]})
        t = d.reset_index().rename(columns={"index": "Column"})
        if "Missing %" in t.columns:
            t["Missing %"] = pd.to_numeric(t["Missing %"], errors="coerce").round(2)
        if "Missing Count" in t.columns:
            t["Missing Count"] = pd.to_numeric(t["Missing Count"], errors="coerce").fillna(0).astype(int)
        return t

    def _fmt_dup_groups(d: pd.DataFrame) -> pd.DataFrame:
        if d is None or d.empty:
            return pd.DataFrame({"Info": ["No duplicate groups."]})
        return d.rename(columns={"_dup_group_id": "Group ID", "Group Size": "Size"}).head(20)

    def _fmt_out_counts(d: pd.DataFrame) -> pd.DataFrame:
        if d is None or d.empty:
            return pd.DataFrame({"Info": ["No outliers detected."]})
        t = d.reset_index().rename(columns={"index": "Column"})
        if "Outlier Count" in t.columns:
            t["Outlier Count"] = pd.to_numeric(t["Outlier Count"], errors="coerce").fillna(0).astype(int)
        return t

    def _fmt_out_rows(d: pd.DataFrame, max_rows: int = 50) -> pd.DataFrame:
        if d is None or d.empty:
            return pd.DataFrame({"Info": ["No rows flagged as outliers."]})
        cols = list(d.columns)
        if "flagged_outlier_in" in cols:
            cols = [c for c in cols if c != "flagged_outlier_in"] + ["flagged_outlier_in"]
        t = d[cols].head(max_rows).copy()
        # tidy long text (no ellipsis in counts/ids)
        for c in t.columns:
            if t[c].dtype == "object":
                t[c] = (
                    t[c].astype(str)
                        .str.replace(r"\s+", " ", regex=True)
                        .str.slice(0, 220)
                )
        t.index.name = "Row #"
        return t.reset_index()

    def _fmt_labels(d: pd.DataFrame) -> pd.DataFrame:
        if d is None or d.empty:
            return pd.DataFrame({"Info": ["No labeling issues found."]})
        t = d.copy()
        if "Detail" in t.columns:
            t["Detail"] = (
                t["Detail"].astype(str)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.slice(0, 500)
            )
        return t

    # formatted views
    missing_tbl    = _fmt_missing(missing_df)
    dup_sum_tbl    = dup_summary_df if dup_summary_df is not None else pd.DataFrame()
    dup_groups_tbl = _fmt_dup_groups(dup_groups)
    out_cnt_tbl    = _fmt_out_counts(out_counts_df)
    out_rows_tbl   = _fmt_out_rows(out_rows_df, max_rows=50)
    labels_tbl     = _fmt_labels(labels_df)

    # build the single-file HTML you want
    from quality import build_quality_html
    profile_url = url_for('profile_report', dataset_id=dataset.id)
    overview    = {"rows": len(df), "cols": df.shape[1]}

    html = build_quality_html(
        name=dataset.original_filename,
        profile_path=profile_url,
        overview=overview,
        missing_df=missing_tbl,
        dup_summary_df=dup_sum_tbl,
        dup_groups_df=dup_groups_tbl,
        out_counts_df=out_cnt_tbl,
        out_rows_df=out_rows_tbl,
        label_issues_df=labels_tbl,
    )

    return Response(html, mimetype='text/html')