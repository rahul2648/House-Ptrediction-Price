# 🏠 HouseIQ — House Price Predictor

> AI-powered house price prediction using Linear Regression + Flask + Chart.js

---

## ⚡ Quick Start (VS Code mein run karo)
<img width="1533" height="784" alt="image" src="https://github.com/user-attachments/assets/906fe21a-0d51-4c3b-82e5-7e3fa655ce24" />

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
- ## 📸 Screenshots

### Hero
<img width="1523" height="784" alt="Hero section" src="https://github.com/user-attachments/assets/9e168098-6f18-405a-b3a4-6fd9c4d89d04" />

### Predictor
<img width="1530" height="784" alt="image" src="https://github.com/user-attachments/assets/195e821b-22d3-490a-9b65-31f71c598b4c" />

### Model Metrics & Coefficients
<img width="1533" height="784" alt="Model performance metrics and feature coefficients" src="https://github.com/user-attachments/assets/ae59b422-fe8b-4869-afd0-2fb9105085d9" />

### Raw Training Data
<img width="1533" height="784" alt="image" src="https://github.com/user-attachments/assets/af58879b-42fd-4dc3-9af9-8e1535cdaa8e" />
