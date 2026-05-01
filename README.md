# 🏠 HouseIQ — House Price Predictor

> AI-powered house price prediction using Linear Regression + Flask + Chart.js

---

## ⚡ Quick Start (VS Code mein run karo)

### Step 1 — Dependencies Install karo

```bash
pip install -r requirements.txt
```

### Step 2 — Server Run karo

```bash
python app.py
```

### Step 3 — Browser mein kholo

```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
house-price-prediction/
├── app.py                  ← Flask backend + ML model
├── requirements.txt        ← Python dependencies
├── README.md               ← Ye file
└── templates/
    └── index.html          ← Frontend (animations, charts, UI)
```

---

## 🧠 ML Model Details

| Property         | Value                         |
|------------------|-------------------------------|
| Algorithm        | Linear Regression (sklearn)   |
| Training samples | 400 (80%)                     |
| Testing samples  | 100 (20%)                     |
| Features         | 9                             |
| Preprocessing    | StandardScaler                |

### Features Used:
- Area (sq ft)
- Bedrooms
- Bathrooms
- Floors
- Age (years)
- Garage Spaces
- Location (Premium/Good/Average/Economy)
- Swimming Pool (Yes/No)
- School Distance (km)

---

## 🌐 API Endpoints

| Endpoint          | Method | Description                        |
|-------------------|--------|------------------------------------|
| `/`               | GET    | Frontend HTML page                 |
| `/api/metrics`    | GET    | Model performance metrics          |
| `/api/raw-data`   | GET    | First 50 rows of training data     |
| `/api/chart-data` | GET    | Data for all 5 charts              |
| `/api/predict`    | POST   | Predict house price                |

### Predict API Example:
```json
POST /api/predict
{
  "area": 2500,
  "bedrooms": 4,
  "bathrooms": 3,
  "floors": 2,
  "age": 5,
  "garage": 2,
  "school_dist": 1.5,
  "location": "Premium",
  "pool": 1
}
```

---

## 🎨 Frontend Sections

1. **Hero** — Animated particle background, live stats
2. **Features** — 6 feature cards with hover effects  
3. **Model Metrics** — R², RMSE, MAE, SVG gauge, coefficient bars
4. **Charts** — 5 interactive Chart.js visualizations
5. **Raw Data** — Scrollable data table with 50 rows
6. **Predictor** — 9-field form with instant AI prediction

---

## 🔧 VS Code Tips

- Install **Python** extension
- Press `Ctrl+`` ` to open terminal
- Run `python app.py` and open `http://127.0.0.1:5000`
