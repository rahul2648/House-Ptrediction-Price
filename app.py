from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
#  DATASET GENERATION  (realistic synthetic data)
# ─────────────────────────────────────────────
np.random.seed(42)
N = 500

area        = np.random.randint(500, 5000, N)
bedrooms    = np.random.randint(1, 7, N)
bathrooms   = np.random.randint(1, 5, N)
floors      = np.random.randint(1, 4, N)
age         = np.random.randint(0, 50, N)
garage      = np.random.randint(0, 3, N)
location    = np.random.choice(['Premium', 'Good', 'Average', 'Economy'], N,
                                p=[0.2, 0.3, 0.35, 0.15])
pool        = np.random.choice([0, 1], N, p=[0.7, 0.3])
school_dist = np.random.uniform(0.5, 10, N)

loc_map = {'Premium': 4, 'Good': 3, 'Average': 2, 'Economy': 1}
loc_num = np.array([loc_map[l] for l in location])

# Price formula (realistic)
price = (
    area * 150
    + bedrooms * 25000
    + bathrooms * 18000
    + floors * 20000
    - age * 3000
    + garage * 15000
    + loc_num * 80000
    + pool * 45000
    - school_dist * 8000
    + np.random.normal(0, 30000, N)
)
price = np.clip(price, 80000, 2500000)

df = pd.DataFrame({
    'Area_sqft':     area,
    'Bedrooms':      bedrooms,
    'Bathrooms':     bathrooms,
    'Floors':        floors,
    'Age_years':     age,
    'Garage_spaces': garage,
    'Location':      location,
    'Pool':          pool,
    'School_dist_km':np.round(school_dist, 2),
    'Price_USD':     np.round(price, -3).astype(int)
})

# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────
features = ['Area_sqft','Bedrooms','Bathrooms','Floors',
            'Age_years','Garage_spaces','Pool','School_dist_km','loc_num']
df['loc_num'] = loc_num

X = df[features]
y = df['Price_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_sc, y_train)

y_pred  = model.predict(X_test_sc)
y_pred_train = model.predict(X_train_sc)

r2      = round(r2_score(y_test, y_pred), 4)
mse     = round(mean_squared_error(y_test, y_pred), 2)
rmse    = round(np.sqrt(mse), 2)
mae     = round(mean_absolute_error(y_test, y_pred), 2)
train_r2= round(r2_score(y_train, y_pred_train), 4)

print(f"✅ Model Trained  |  R²={r2}  |  RMSE={rmse:,.0f}  |  MAE={mae:,.0f}")

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/metrics')
def metrics():
    coeff = dict(zip(features, model.coef_.tolist()))
    return jsonify({
        'r2':       r2,
        'train_r2': train_r2,
        'mse':      mse,
        'rmse':     rmse,
        'mae':      mae,
        'accuracy': round(r2 * 100, 2),
        'intercept': round(model.intercept_, 2),
        'coefficients': {k: round(v, 4) for k, v in coeff.items()},
        'train_size': len(X_train),
        'test_size':  len(X_test),
        'total_samples': N
    })


@app.route('/api/raw-data')
def raw_data():
    sample = df.drop(columns=['loc_num']).head(50)
    return jsonify({
        'columns': sample.columns.tolist(),
        'data':    sample.values.tolist(),
        'total':   len(df)
    })


@app.route('/api/chart-data')
def chart_data():
    # Scatter: actual vs predicted (test set)
    scatter_actual    = y_test.tolist()
    scatter_predicted = y_pred.tolist()

    # Feature importance (abs coeff)
    importance = sorted(
        zip(features, np.abs(model.coef_).tolist()),
        key=lambda x: x[1], reverse=True
    )
    feat_names  = [i[0].replace('_', ' ') for i in importance]
    feat_values = [round(i[1], 2) for i in importance]

    # Price distribution histogram
    hist_vals, hist_edges = np.histogram(df['Price_USD'] / 1000, bins=20)
    hist_labels = [f"${e:.0f}K" for e in hist_edges[:-1]]

    # Price by location
    loc_avg = df.groupby('Location')['Price_USD'].mean().round(-3).astype(int)
    loc_labels = loc_avg.index.tolist()
    loc_values = loc_avg.values.tolist()

    # Area vs Price sample (200 pts)
    sample = df.sample(200, random_state=1)

    # Correlation with price
    corr_df  = df[features[:-1] + ['Price_USD']].corr()['Price_USD'].drop('Price_USD')
    corr_feats = corr_df.index.tolist()
    corr_vals  = [round(v, 3) for v in corr_df.values.tolist()]

    return jsonify({
        'scatter': {
            'actual':    [round(v / 1000, 1) for v in scatter_actual],
            'predicted': [round(v / 1000, 1) for v in scatter_predicted]
        },
        'feature_importance': {'labels': feat_names, 'values': feat_values},
        'price_distribution': {'labels': hist_labels, 'values': hist_vals.tolist()},
        'price_by_location':  {'labels': loc_labels, 'values': loc_values},
        'area_vs_price': {
            'area':  sample['Area_sqft'].tolist(),
            'price': (sample['Price_USD'] / 1000).round(1).tolist()
        },
        'correlation': {'labels': corr_feats, 'values': corr_vals}
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        d = request.json
        loc_num_val = {'Premium': 4, 'Good': 3, 'Average': 2, 'Economy': 1}.get(
            d.get('location', 'Average'), 2)
        inp = np.array([[
            float(d['area']),
            float(d['bedrooms']),
            float(d['bathrooms']),
            float(d['floors']),
            float(d['age']),
            float(d['garage']),
            int(d.get('pool', 0)),
            float(d['school_dist']),
            loc_num_val
        ]])
        inp_sc = scaler.transform(inp)
        price_pred = model.predict(inp_sc)[0]
        price_pred = max(50000, price_pred)

        # Confidence interval (±15% simplified)
        low  = round(price_pred * 0.88, -3)
        high = round(price_pred * 1.12, -3)

        return jsonify({
            'success':    True,
            'price':      round(price_pred, -3),
            'price_low':  int(low),
            'price_high': int(high),
            'price_fmt':  f"${price_pred:,.0f}",
            'low_fmt':    f"${low:,.0f}",
            'high_fmt':   f"${high:,.0f}"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
