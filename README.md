# 🛒 QuickCart — Smart POS & Business Suite

**QuickCart** is a full-featured Point-of-Sale, CRM, and Inventory Management web app built with Python and Streamlit.

> Built by Nabisab Nadaf as an individual project · v1.0.0

---

## ✨ Features

| Module | What it does |
|---|---|
| 📊 **Dashboard** | KPI cards, daily revenue trend, top/least selling charts, payment mix |
| 💳 **POS & Billing** | Cart, discount, GST, cash/online/credit payment, PDF receipt download |
| 📦 **Inventory** | Add/edit/delete items, category filter, bulk CSV import, low-stock alerts |
| 👥 **CRM** | Customer profiles, loyalty points, visit counter, spending insights |
| 🏭 **Suppliers** | Supplier directory with reliability score |
| 🧍 **Employees** | Employee records, daily attendance marking, attendance report |
| 💰 **Expenses** | Log business expenses by category, net profit calculation |
| 📈 **Reports** | Filtered sales report, profit analysis scatter chart, CSV export |
| 🤖 **AI Forecast** | Multi-model ensemble demand prediction (Polynomial + Linear + WMA) |

---

## 🤖 AI Demand Forecasting

QuickCart uses a **3-model ensemble** to predict demand for the next 7 days:

- **Polynomial Regression** — captures non-linear growth curves and seasonal spikes
- **Linear Regression** — captures the overall long-term sales trend
- **Weighted Moving Average** — emphasises recent sales for short-term accuracy

Each prediction also includes a **confidence score** (High / Medium / Low) based on how consistent the item's sales pattern is.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/nabisab/quickcart.git
cd quickcart

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run quickcart.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
quickcart/
├── quickcart.py          # Main application
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── LICENSE               # MIT License
└── data/                 # Auto-created on first run
    ├── inventory.csv
    ├── sales.csv
    ├── customers.csv
    ├── suppliers.csv
    ├── employees.csv
    ├── attendance.csv
    ├── expenses.csv
    └── backup/           # Daily automatic backups
```

> **Note:** The `data/` folder is auto-created when you first run the app. It is excluded from version control via `.gitignore` to protect business data.

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo, set main file as `quickcart.py`
5. Click **Deploy** — your app gets a public URL instantly

> **Important:** Streamlit Cloud has an ephemeral filesystem. For persistent data on the cloud, migrate the CSV storage to [Supabase](https://supabase.com) (free tier available).

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI framework |
| [Pandas](https://pandas.pydata.org) | Data handling |
| [NumPy](https://numpy.org) | Numerical computation |
| [Altair](https://altair-viz.github.io) | Interactive charts |
| [scikit-learn](https://scikit-learn.org) | AI / ML forecasting |
| [ReportLab](https://www.reportlab.com) | PDF receipt generation |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Nabisab Nadaf**  
GitHub: [@nabisab](https://github.com/nabisab)

---

*Made with ❤️ as this project represents my first independent hands-on implementation in data-related development.*
