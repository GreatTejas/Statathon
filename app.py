from flask import Flask, request, send_file, render_template
import pandas as pd
import numpy as np
from io import StringIO, BytesIO

app = Flask(__name__)

# -------------- CLEANING LOGIC -----------------
def safe_read_csv(fileobj):
    fileobj.seek(0)
    try:
        return pd.read_csv(fileobj)
    except Exception:
        fileobj.seek(0)
        return pd.read_csv(fileobj, encoding='latin1', sep=None, engine='python')

def clean_df(df):
    df = df.copy()
    # Normalize column names
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    # Drop empty rows/cols
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    # Trim strings
    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
        df[c].replace({'': pd.NA, 'nan': pd.NA}, inplace=True)
    # Convert to numeric if possible
    for c in df.columns:
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().sum() >= len(df) * 0.5:
                df[c] = coerced
    # Fill missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c].fillna(df[c].median(), inplace=True)
    non_num_cols = [c for c in df.columns if c not in num_cols]
    for c in non_num_cols:
        if df[c].dropna().shape[0] > 0:
            mode = df[c].mode(dropna=True)
            if not mode.empty:
                df[c].fillna(mode.iloc[0], inplace=True)
            else:
                df[c].fillna('Unknown', inplace=True)
        else:
            df[c].fillna('Unknown', inplace=True)
    # Drop duplicates
    df = df.drop_duplicates()
    # Cap outliers
    for c in num_cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[c] = df[c].clip(lower, upper)
    return df

# -------------- ROUTES -----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file provided", 400

    f = request.files['file']
    try:
        df = safe_read_csv(f)
    except Exception as e:
        return f"Could not parse CSV: {e}", 400

    cleaned = clean_df(df)

    s = StringIO()
    cleaned.to_csv(s, index=False)
    mem = BytesIO()
    mem.write(s.getvalue().encode('utf-8'))
    mem.seek(0)

    return send_file(mem, mimetype='text/csv',
                     as_attachment=True,
                     download_name='cleaned.csv')

if __name__ == '__main__':
    app.run(debug=True)
