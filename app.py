# ================================================================
# QuickCart — Smart POS + CRM + Inventory + AI Suite
# Author  : Nabisab Nadaf
# Version : 1.0.0
# GitHub  : github.com/nabisab/quickcart
# License : MIT
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os, io, shutil
from datetime import datetime, date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

# ──────────────────────────────────────────────
# PAGE CONFIG & THEME
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="QuickCart — Smart POS",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY = "#0f172a"
ACCENT  = "#b68c2a"
ACCENT2 = "#1d4ed8"
BG      = "#f8f9fc"
CARD    = "#ffffff"
SUCCESS = "#16a34a"
DANGER  = "#dc2626"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{ background:{BG}; }}
h1,h2,h3,h4,h5 {{ color:{PRIMARY}; }}
[data-testid="stSidebar"] > div:first-child {{
    background:{CARD}; border-right:1px solid #e5e7eb;
}}
[data-testid="stMetricValue"] {{ color:{PRIMARY}; font-size:1.4rem !important; font-weight:700; }}
.stButton > button {{
    background:{ACCENT}; color:white; border:none;
    border-radius:8px; padding:6px 16px; font-weight:600;
    transition:background .2s;
}}
.stButton > button:hover {{ background:#caa03d; color:white; }}
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; }}
[data-testid="stExpander"] {{ border-radius:10px; border:1px solid #e5e7eb; }}
.stTabs [data-baseweb="tab-list"] {{ gap:8px; }}
.stTabs [data-baseweb="tab"] {{ border-radius:8px 8px 0 0; }}
.kpi-card {{
    background:{CARD}; border-radius:14px; padding:18px 22px;
    border:1px solid #e5e7eb; box-shadow:0 1px 4px rgba(0,0,0,.04);
    margin-bottom:4px;
}}
.kpi-label {{ font-size:11px; color:#6b7280; font-weight:600;
              text-transform:uppercase; letter-spacing:.06em; }}
.kpi-value {{ font-size:26px; font-weight:700; color:{PRIMARY}; margin:4px 0 2px; }}
.kpi-sub   {{ font-size:11px; color:#9ca3af; }}
.badge-green {{ display:inline-block; background:#dcfce7; color:#16a34a;
                font-size:11px; font-weight:600; padding:2px 8px; border-radius:20px; }}
.badge-red   {{ display:inline-block; background:#fee2e2; color:#dc2626;
                font-size:11px; font-weight:600; padding:2px 8px; border-radius:20px; }}
.badge-amber {{ display:inline-block; background:#fef9c3; color:#b45309;
                font-size:11px; font-weight:600; padding:2px 8px; border-radius:20px; }}
.badge-blue  {{ display:inline-block; background:#dbeafe; color:#1d4ed8;
                font-size:11px; font-weight:600; padding:2px 8px; border-radius:20px; }}
.ai-card {{
    background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
    border-radius:14px; padding:20px 24px; color:white; margin-bottom:12px;
}}
.ai-card h4 {{ color:white !important; margin:0 0 6px; }}
.ai-card p  {{ color:#94a3b8; font-size:13px; margin:0; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PATHS & DIRECTORIES
# ──────────────────────────────────────────────
DATA_DIR   = "data"
BACKUP_DIR = os.path.join(DATA_DIR, "backup")
for d in [DATA_DIR, BACKUP_DIR]:
    os.makedirs(d, exist_ok=True)

INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.csv")
SALES_FILE     = os.path.join(DATA_DIR, "sales.csv")
CUSTOMERS_FILE = os.path.join(DATA_DIR, "customers.csv")
SUPPLIERS_FILE = os.path.join(DATA_DIR, "suppliers.csv")
EMPLOYEES_FILE = os.path.join(DATA_DIR, "employees.csv")
ATTEND_FILE    = os.path.join(DATA_DIR, "attendance.csv")
EXPENSES_FILE  = os.path.join(DATA_DIR, "expenses.csv")

# ──────────────────────────────────────────────
# COLUMN SCHEMAS
# ──────────────────────────────────────────────
INV_COLS   = ["item","category","cost_price","unit_price","stock","expiry","supplier","reorder_level","barcode"]
SALES_COLS = ["invoice","datetime","item","qty","unit_price","total","customer","payment","profit","discount_pct","gst_pct"]
CUST_COLS  = ["name","phone","email","points","total_spent","visits","last_purchase"]
SUP_COLS   = ["name","contact","email","reliability","items_supplied"]
EMP_COLS   = ["name","role","salary","branch","join_date"]
ATT_COLS   = ["name","date","present"]
EXP_COLS   = ["date","category","amount","note"]

# ──────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────
def load_csv(file, cols):
    if not os.path.exists(file):
        pd.DataFrame(columns=cols).to_csv(file, index=False)
    df = pd.read_csv(file)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def save_csv(df, file):
    df.to_csv(file, index=False)

def money(x):
    try:    return f"₹{float(x):,.2f}"
    except: return "₹0.00"

def today_str():
    return date.today().strftime("%Y-%m-%d")

def make_invoice_id():
    s = load_csv(SALES_FILE, SALES_COLS)
    if s.empty:
        seq = 1
    else:
        nums = []
        for v in s["invoice"].dropna().astype(str):
            try: nums.append(int(v.split("-")[-1]))
            except: pass
        seq = (max(nums) + 1) if nums else 1
    return f"QC-{datetime.now().strftime('%Y%m%d')}-{seq:05d}"

def backup_daily():
    for f in [INVENTORY_FILE, SALES_FILE, CUSTOMERS_FILE]:
        if os.path.exists(f):
            bfile = os.path.join(BACKUP_DIR, f"{today_str()}_{os.path.basename(f)}")
            if not os.path.exists(bfile):
                shutil.copy(f, bfile)

backup_daily()

# ──────────────────────────────────────────────
# CACHED DATA LOADER
# ──────────────────────────────────────────────
@st.cache_data(ttl=2)
def load_all():
    return {
        "inventory" : load_csv(INVENTORY_FILE, INV_COLS),
        "sales"     : load_csv(SALES_FILE,     SALES_COLS),
        "customers" : load_csv(CUSTOMERS_FILE, CUST_COLS),
        "suppliers" : load_csv(SUPPLIERS_FILE, SUP_COLS),
        "employees" : load_csv(EMPLOYEES_FILE, EMP_COLS),
        "attendance": load_csv(ATTEND_FILE,    ATT_COLS),
        "expenses"  : load_csv(EXPENSES_FILE,  EXP_COLS),
    }

def reload():
    st.cache_data.clear()
    st.rerun()

data       = load_all()
inventory  = data["inventory"]
sales      = data["sales"]
customers  = data["customers"]
suppliers  = data["suppliers"]
employees  = data["employees"]
attendance = data["attendance"]
expenses   = data["expenses"]

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
for k, v in {"cart":{}, "discount":0.0, "gst":18.0, "cash":0.0,
             "payment":"Cash", "customer":""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# AI DEMAND FORECAST ENGINE (v2 — Ensemble)
# ──────────────────────────────────────────────
def ai_forecast_ensemble(df):
    """
    Multi-model ensemble demand forecast.
    Uses Gradient Boosting + Random Forest + Linear Regression.
    Features: day-of-week, day-of-month, rolling avg, lag-1, lag-7.
    Falls back gracefully with limited data.
    Returns DataFrame with item, predicted_units, confidence, trend, restock_by.
    """
    if df.empty or "datetime" not in df.columns:
        return pd.DataFrame(columns=["item","predicted_units","confidence","trend","restock_by","model_used"])

    dfc = df.copy()
    dfc["date"] = pd.to_datetime(dfc["datetime"], errors="coerce").dt.date
    dfc["qty"]  = pd.to_numeric(dfc["qty"], errors="coerce").fillna(0)
    agg = dfc.groupby(["item","date"])["qty"].sum().reset_index()

    results = []
    for item, sub in agg.groupby("item"):
        sub = sub.sort_values("date").reset_index(drop=True)
        n   = len(sub)
        if n < 2:
            continue

        sub["dnum"]     = np.arange(n)
        sub["dow"]      = pd.to_datetime(sub["date"]).dt.dayofweek
        sub["dom"]      = pd.to_datetime(sub["date"]).dt.day
        sub["roll3"]    = sub["qty"].rolling(3, min_periods=1).mean()
        sub["lag1"]     = sub["qty"].shift(1).fillna(sub["qty"].mean())
        sub["lag7"]     = sub["qty"].shift(7).fillna(sub["qty"].mean())

        feature_cols = ["dnum","dow","dom","roll3","lag1","lag7"]
        X = sub[feature_cols].values
        y = sub["qty"].values

        # build future row (next 7 days ahead)
        last_dnum = n + 7
        future_dt = pd.Timestamp(sub["date"].iloc[-1]) + timedelta(days=7)
        fut_roll  = float(sub["qty"].tail(3).mean())
        fut_lag1  = float(sub["qty"].iloc[-1])
        fut_lag7  = float(sub["qty"].iloc[-7]) if n >= 7 else float(sub["qty"].mean())
        X_fut = np.array([[last_dnum, future_dt.dayofweek, future_dt.day,
                           fut_roll, fut_lag1, fut_lag7]])

        preds_all = []
        model_name = "Linear"

        # Linear Regression (always available)
        lr = LinearRegression().fit(X, y)
        preds_all.append(max(0, lr.predict(X_fut)[0]))

        if n >= 5:
            # Random Forest
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            preds_all.append(max(0, rf.predict(X_fut)[0]))
            model_name = "Random Forest"

        if n >= 8:
            # Gradient Boosting
            gb = GradientBoostingRegressor(n_estimators=80, learning_rate=0.1,
                                           max_depth=3, random_state=42)
            gb.fit(X, y)
            preds_all.append(max(0, gb.predict(X_fut)[0]))
            model_name = "Gradient Boosting"

        # Ensemble average
        final_pred = round(float(np.mean(preds_all)), 1)

        # Confidence: lower MAE on training = higher confidence
        y_hat = lr.predict(X)
        mae   = mean_absolute_error(y, y_hat)
        mean_y = y.mean() if y.mean() != 0 else 1
        conf_raw = max(0, 1 - mae / mean_y)
        confidence = f"{min(99, round(conf_raw * 100))}%"

        # Trend
        slope = lr.coef_[0]
        trend = "↑ Rising" if slope > 0.05 else ("↓ Falling" if slope < -0.05 else "→ Stable")

        # Restock suggestion: if predicted > current stock
        cur_stock = float(pd.to_numeric(
            inventory.loc[inventory["item"] == item, "stock"], errors="coerce"
        ).fillna(0).sum())
        restock_by = max(0, round(final_pred - cur_stock, 1))

        results.append({
            "item":            item,
            "predicted_units": final_pred,
            "current_stock":   int(cur_stock),
            "restock_by":      restock_by,
            "confidence":      confidence,
            "trend":           trend,
            "model_used":      model_name,
        })

    return pd.DataFrame(results)


def ai_smart_insights(sales_df, inv_df, exp_df):
    """Generate text-based AI insights from data patterns."""
    insights = []
    if sales_df.empty:
        return ["📭 No sales data yet — insights will appear after first sales."]

    sales_df = sales_df.copy()
    sales_df["total"]  = pd.to_numeric(sales_df["total"],  errors="coerce").fillna(0)
    sales_df["profit"] = pd.to_numeric(sales_df["profit"], errors="coerce").fillna(0)
    sales_df["qty"]    = pd.to_numeric(sales_df["qty"],    errors="coerce").fillna(0)
    sales_df["_date"]  = pd.to_datetime(sales_df["datetime"], errors="coerce").dt.date

    # Best selling item
    top = sales_df.groupby("item")["qty"].sum().idxmax()
    insights.append(f"🏆 **Best seller:** {top} — consider ensuring it's always well-stocked.")

    # Peak day
    sales_df["dow"] = pd.to_datetime(sales_df["datetime"], errors="coerce").dt.day_name()
    peak_day = sales_df.groupby("dow")["total"].sum().idxmax()
    insights.append(f"📅 **Peak sales day:** {peak_day} — schedule more staff and stock on this day.")

    # Most profitable item
    top_profit = sales_df.groupby("item")["profit"].sum().idxmax()
    insights.append(f"💰 **Most profitable item:** {top_profit} — prioritise this in your promotions.")

    # Low margin warning
    by_item = sales_df.groupby("item").agg(rev=("total","sum"), prof=("profit","sum"))
    by_item["margin"] = by_item["prof"] / by_item["rev"].replace(0, np.nan)
    low_margin = by_item[by_item["margin"] < 0.1].index.tolist()
    if low_margin:
        insights.append(f"⚠️ **Low margin items (<10%):** {', '.join(low_margin[:3])} — review pricing or costs.")

    # Expense ratio
    if not exp_df.empty:
        total_rev = sales_df["total"].sum()
        total_exp = pd.to_numeric(exp_df["amount"], errors="coerce").sum()
        ratio = (total_exp / total_rev * 100) if total_rev > 0 else 0
        if ratio > 40:
            insights.append(f"🔴 **Expense ratio is {ratio:.1f}%** of revenue — consider reducing operating costs.")
        else:
            insights.append(f"✅ **Expense ratio is {ratio:.1f}%** of revenue — healthy control.")

    # Low stock
    low_s = inv_df[pd.to_numeric(inv_df["stock"], errors="coerce").fillna(0) <=
                   pd.to_numeric(inv_df["reorder_level"], errors="coerce").fillna(0)]
    if not low_s.empty:
        insights.append(f"📦 **{len(low_s)} items need restocking:** {', '.join(low_s['item'].tolist()[:4])}.")

    # Payment mode
    if "payment" in sales_df.columns:
        top_pay = sales_df["payment"].value_counts().idxmax()
        insights.append(f"💳 **Most used payment mode:** {top_pay}.")

    return insights


# ──────────────────────────────────────────────
# PDF RECEIPT
# ──────────────────────────────────────────────
def make_receipt_pdf(invoice_id, now_dt, cart, subtotal, disc_pct, disc_amt,
                     gst_pct, gst_amt, grand, cash_given, change_due, payment, customer):
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = H - 20*mm

    c.setFont("Helvetica-Bold", 18)
    c.drawString(20*mm, y, "QuickCart")
    y -= 7*mm
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(.4,.4,.4)
    c.drawString(20*mm, y, "Smart POS — by Nabisab Nadaf")
    c.setFillColorRGB(0,0,0)
    y -= 5*mm
    c.line(20*mm, y, W-20*mm, y)
    y -= 6*mm

    c.setFont("Helvetica", 9)
    c.drawString(20*mm, y, f"Invoice: {invoice_id}")
    c.drawRightString(W-20*mm, y, f"Date: {now_dt}")
    y -= 5*mm
    c.drawString(20*mm, y, f"Customer: {customer or 'Walk-in'}   |   Payment: {payment}")
    y -= 5*mm
    c.line(20*mm, y, W-20*mm, y)
    y -= 7*mm

    c.setFont("Helvetica-Bold", 9)
    c.drawString(22*mm, y, "Item");        c.drawRightString(95*mm,  y, "Qty")
    c.drawRightString(130*mm, y, "Unit");  c.drawRightString(170*mm, y, "Total")
    y -= 5*mm
    c.line(20*mm, y, W-20*mm, y);  y -= 5*mm

    c.setFont("Helvetica", 9)
    for itm, d in cart.items():
        line_total = d["qty"] * d["price"]
        c.drawString(22*mm, y, itm[:32])
        c.drawRightString(95*mm,  y, str(d["qty"]))
        c.drawRightString(130*mm, y, f"{d['price']:.2f}")
        c.drawRightString(170*mm, y, f"{line_total:.2f}")
        y -= 5*mm

    y -= 3*mm;  c.line(20*mm, y, W-20*mm, y);  y -= 6*mm

    for label, val in [
        ("Subtotal",                money(subtotal)),
        (f"Discount ({disc_pct}%)", f"-{money(disc_amt)}"),
        (f"GST ({gst_pct}%)",       f"+{money(gst_amt)}"),
    ]:
        c.drawRightString(140*mm, y, label + ":"); c.drawRightString(170*mm, y, val); y -= 5*mm

    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(140*mm, y, "GRAND TOTAL:"); c.drawRightString(170*mm, y, money(grand)); y -= 6*mm

    c.setFont("Helvetica", 9)
    if payment == "Cash":
        c.drawRightString(140*mm, y, "Cash Given:"); c.drawRightString(170*mm, y, money(cash_given)); y -= 5*mm
        c.drawRightString(140*mm, y, "Change Due:"); c.drawRightString(170*mm, y, money(change_due)); y -= 5*mm

    y -= 5*mm;  c.line(20*mm, y, W-20*mm, y);  y -= 7*mm
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColorRGB(.4,.4,.4)
    c.drawCentredString(W/2, y, "Thank you for shopping at QuickCart! — Nabisab Nadaf")
    c.save();  buf.seek(0)
    return buf.getvalue()

# ──────────────────────────────────────────────
# LOW STOCK HELPER
# ──────────────────────────────────────────────
low_stock = inventory[
    pd.to_numeric(inventory["stock"], errors="coerce").fillna(0) <=
    pd.to_numeric(inventory["reorder_level"], errors="coerce").fillna(0)
]

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style="padding:10px 0 16px;">
  <div style="font-size:22px;font-weight:800;color:{PRIMARY};letter-spacing:-0.5px;">🛒 QuickCart</div>
  <div style="font-size:12px;color:#6b7280;margin-top:2px;">by Nabisab Nadaf · v1.0</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "💳 POS & Billing",
    "📦 Inventory",
    "👥 CRM",
    "🏭 Suppliers",
    "🧍 Employees",
    "💰 Expenses",
    "🤖 AI Insights",
    "📈 Reports",
])

st.sidebar.markdown("---")
if not low_stock.empty:
    st.sidebar.warning(f"⚠️ {len(low_stock)} item(s) low on stock")
st.sidebar.markdown(f"<div style='font-size:12px;color:#6b7280;margin-top:8px;'>📅 {today_str()}</div>",
                    unsafe_allow_html=True)

# ================================================================
# PAGE: DASHBOARD
# ================================================================
if page == "📊 Dashboard":
    st.title("📊 Dashboard")

    today_s = sales.copy()
    if not today_s.empty:
        today_s["_date"] = pd.to_datetime(today_s["datetime"], errors="coerce").dt.date
        today_s = today_s[today_s["_date"] == date.today()]

    def safe_sum(df, col):
        return pd.to_numeric(df[col], errors="coerce").sum() if not df.empty else 0

    total_today  = safe_sum(today_s, "total")
    profit_today = safe_sum(today_s, "profit")
    total_all    = safe_sum(sales, "total")
    profit_all   = safe_sum(sales, "profit")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    for col, label, val, sub in [
        (k1, "Today's Sales",   money(total_today),   "revenue"),
        (k2, "Today's Profit",  money(profit_today),  "net margin"),
        (k3, "All-Time Sales",  money(total_all),     "lifetime"),
        (k4, "All-Time Profit", money(profit_all),    "lifetime"),
        (k5, "Customers",       len(customers),       "registered"),
        (k6, "Low Stock",       len(low_stock),       "need reorder"),
    ]:
        col.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not low_stock.empty:
        with st.expander(f"⚠️ Low Stock Alert — {len(low_stock)} items", expanded=True):
            st.dataframe(low_stock[["item","stock","reorder_level","supplier"]],
                         use_container_width=True, hide_index=True)

    if not sales.empty:
        t1, t2, t3 = st.tabs(["Sales Trend", "Top Items", "Payment Mix"])

        with t1:
            sd = sales.copy()
            sd["date"]  = pd.to_datetime(sd["datetime"], errors="coerce").dt.date
            sd["total"] = pd.to_numeric(sd["total"], errors="coerce")
            daily = sd.groupby("date")["total"].sum().reset_index()
            daily["date"] = pd.to_datetime(daily["date"])
            area  = alt.Chart(daily).mark_area(color=ACCENT2, opacity=0.12).encode(x="date:T", y="total:Q")
            line  = alt.Chart(daily).mark_line(color=ACCENT2, strokeWidth=2.5).encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("total:Q", title="Revenue (₹)"),
                tooltip=["date","total"]
            )
            st.altair_chart(area + line, use_container_width=True)

        with t2:
            top_items = pd.to_numeric(sales["qty"], errors="coerce").groupby(sales["item"]).sum().nlargest(10).reset_index()
            top_items.columns = ["item","qty"]
            bar = alt.Chart(top_items).mark_bar(color=ACCENT).encode(
                x=alt.X("qty:Q", title="Units Sold"),
                y=alt.Y("item:N", sort="-x", title=""),
                tooltip=["item","qty"]
            ).properties(height=300)
            st.altair_chart(bar, use_container_width=True)

        with t3:
            pay = sales["payment"].value_counts().reset_index()
            pay.columns = ["mode","count"]
            pie = alt.Chart(pay).mark_arc(outerRadius=110).encode(
                theta="count:Q",
                color=alt.Color("mode:N", scale=alt.Scale(
                    domain=["Cash","Online","Credit"],
                    range=[ACCENT, ACCENT2, SUCCESS]
                )),
                tooltip=["mode","count"]
            ).properties(height=280)
            st.altair_chart(pie, use_container_width=True)
    else:
        st.info("No sales data yet. Head to POS & Billing to make your first sale!")

# ================================================================
# PAGE: POS & BILLING
# ================================================================
elif page == "💳 POS & Billing":
    st.title("💳 Point-of-Sale (Billing)")
    inv = load_csv(INVENTORY_FILE, INV_COLS)

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Product Catalogue")
        sc, cc = st.columns([3,2])
        query    = sc.text_input("🔍 Search", placeholder="Item name...").strip()
        cats     = ["All"] + sorted(inv["category"].dropna().unique().tolist())
        cat_filt = cc.selectbox("Category", cats)

        disp = inv.copy()
        if query:
            disp = disp[disp["item"].str.contains(query, case=False, na=False)]
        if cat_filt != "All":
            disp = disp[disp["category"] == cat_filt]

        st.dataframe(disp[["item","category","unit_price","stock"]].rename(
            columns={"unit_price":"Price","stock":"Stock"}),
            use_container_width=True, hide_index=True)

        for _, row in disp.iterrows():
            a,b,c,d,e = st.columns([4,2,2,2,1])
            a.write(f"**{row['item']}**")
            b.write(money(row["unit_price"]))
            
            val = pd.to_numeric(row["stock"], errors="coerce")
            stock_val = int(val) if pd.notna(val) else 0
            c.write(f"Stock: {stock_val}")
            if stock_val > 0:
                qty = d.number_input("Qty", 1, stock_val, 1, key=f"qty_{row['item']}")
                if e.button("➕", key=f"add_{row['item']}"):
                    existing = st.session_state.cart.get(row["item"], {}).get("qty", 0)
                    st.session_state.cart[row["item"]] = {
                        "qty":   existing + qty,
                        "price": float(row["unit_price"])
                    }
                    st.success(f"✓ Added {qty} × {row['item']}")
            else:
                d.write("")
                e.button("🚫", disabled=True, key=f"no_{row['item']}")

    with right:
        st.subheader("🧾 Current Bill")
        if not st.session_state.cart:
            st.info("Cart is empty.")
        else:
            cart_df = pd.DataFrame([
                {"Item":i,"Qty":d["qty"],"Price":d["price"],"Total":round(d["qty"]*d["price"],2)}
                for i,d in st.session_state.cart.items()
            ])
            st.dataframe(cart_df, hide_index=True, use_container_width=True)

            for itm in list(st.session_state.cart.keys()):
                ra, rb = st.columns([8,1])
                ra.write(f"• {itm}")
                if rb.button("🗑️", key=f"rm_{itm}"):
                    del st.session_state.cart[itm]
                    st.rerun()

            st.divider()
            d1, d2 = st.columns(2)
            st.session_state.discount = d1.number_input("Discount %", 0.0, 100.0, st.session_state.discount, 0.5)
            st.session_state.gst      = d2.number_input("GST %",      0.0, 100.0, st.session_state.gst,      0.5)

            subtotal = sum(d["qty"]*d["price"] for d in st.session_state.cart.values())
            disc_amt = subtotal * st.session_state.discount / 100
            gst_amt  = (subtotal - disc_amt) * st.session_state.gst / 100
            grand    = subtotal - disc_amt + gst_amt

            st.markdown(f"""
| | |
|---|---|
| Subtotal | {money(subtotal)} |
| Discount ({st.session_state.discount}%) | -{money(disc_amt)} |
| GST ({st.session_state.gst}%) | +{money(gst_amt)} |
| **Grand Total** | **{money(grand)}** |
""")
            pm1, pm2 = st.columns(2)
            st.session_state.payment = pm1.radio("Payment Mode",["Cash","Online","Credit"], horizontal=True)
            st.session_state.cash    = pm2.number_input("Cash Given",0.0,100000.0,st.session_state.cash,10.0)
            change = max(0.0, st.session_state.cash - grand) if st.session_state.payment=="Cash" else 0.0
            if st.session_state.payment == "Cash":
                st.write(f"**Change Due: {money(change)}**")

            st.divider()
            cust_names = ["Walk-in"] + load_csv(CUSTOMERS_FILE, CUST_COLS)["name"].tolist()
            chosen = st.selectbox("Customer", cust_names)
            if chosen == "Walk-in":
                new_name = st.text_input("Or type new customer name").strip()
                st.session_state.customer = new_name if new_name else "Walk-in"
            else:
                st.session_state.customer = chosen

            st.divider()
            fc, cc2 = st.columns(2)
            if cc2.button("🧹 Clear Cart"):
                st.session_state.cart = {}
                st.rerun()

            if fc.button("✅ Finalize & Print Bill"):
                cur_inv = load_csv(INVENTORY_FILE, INV_COLS)
                ok = True
                for itm, d in st.session_state.cart.items():
                    row = cur_inv[cur_inv["item"] == itm]
                    if row.empty:
                        st.error(f"'{itm}' not found."); ok = False
                    elif d["qty"] > int(pd.to_numeric(row.iloc[0]["stock"], errors="coerce") or 0):
                        st.error(f"Insufficient stock for '{itm}'"); ok = False

                if ok and st.session_state.cart:
                    inv_id = make_invoice_id()
                    now_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cur_s  = load_csv(SALES_FILE, SALES_COLS)

                    for itm, d in st.session_state.cart.items():
                        line_total = round(d["qty"] * d["price"], 2)
                        cost = float(pd.to_numeric(
                            cur_inv.loc[cur_inv["item"]==itm, "cost_price"], errors="coerce"
                        ).fillna(0).iloc[0])
                        profit = round(line_total - cost * d["qty"], 2)
                        cur_s = pd.concat([cur_s, pd.DataFrame([[
                            inv_id, now_dt, itm, d["qty"], d["price"], line_total,
                            st.session_state.customer, st.session_state.payment,
                            profit, st.session_state.discount, st.session_state.gst
                        ]], columns=SALES_COLS)], ignore_index=True)
                        cur_inv.loc[cur_inv["item"]==itm, "stock"] = (
                            pd.to_numeric(cur_inv.loc[cur_inv["item"]==itm,"stock"], errors="coerce") - d["qty"]
                        )

                    # loyalty points
                    cur_cust = load_csv(CUSTOMERS_FILE, CUST_COLS)
                    pts = int(grand // 100)
                    cname = st.session_state.customer
                    if cname and cname != "Walk-in":
                        if cname in cur_cust["name"].values:
                            for field, add in [("points",pts),("total_spent",grand),("visits",1)]:
                                cur_cust.loc[cur_cust["name"]==cname, field] = (
                                    pd.to_numeric(cur_cust.loc[cur_cust["name"]==cname, field], errors="coerce").fillna(0) + add
                                )
                            cur_cust.loc[cur_cust["name"]==cname, "last_purchase"] = today_str()
                        else:
                            cur_cust = pd.concat([cur_cust, pd.DataFrame([[
                                cname,"","",pts,grand,1,today_str()
                            ]], columns=CUST_COLS)], ignore_index=True)
                        save_csv(cur_cust, CUSTOMERS_FILE)

                    save_csv(cur_s,   SALES_FILE)
                    save_csv(cur_inv, INVENTORY_FILE)
                    st.success(f"✅ {inv_id} recorded! +{pts} loyalty points.")

                    pdf = make_receipt_pdf(inv_id, now_dt, st.session_state.cart,
                                          subtotal, st.session_state.discount, disc_amt,
                                          st.session_state.gst, gst_amt, grand,
                                          st.session_state.cash, change,
                                          st.session_state.payment, st.session_state.customer)
                    st.download_button("⬇️ Download Receipt PDF", data=pdf,
                                       file_name=f"{inv_id}.pdf", mime="application/pdf")
                    st.session_state.cart = {}
                    st.session_state.discount = 0.0
                    st.session_state.gst = 18.0
                    st.session_state.cash = 0.0
                    reload()

# ================================================================
# PAGE: INVENTORY
# ================================================================
elif page == "📦 Inventory":
    st.title("📦 Inventory Management")
    tab_view, tab_add, tab_edit, tab_bulk = st.tabs(["View", "Add Item", "Edit / Delete", "Bulk Import"])

    with tab_view:
        si = load_csv(INVENTORY_FILE, INV_COLS)
        c1,c2 = st.columns(2)
        c1.metric("Total Items", len(si))
        low_n = len(si[pd.to_numeric(si["stock"],errors="coerce").fillna(0) <=
                       pd.to_numeric(si["reorder_level"],errors="coerce").fillna(0)])
        c2.metric("Low Stock Items", low_n)
        st.dataframe(si, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export CSV", si.to_csv(index=False).encode(), "inventory.csv","text/csv")

    with tab_add:
        r1,r2,r3 = st.columns(3)
        item     = r1.text_input("Item Name *")
        category = r2.text_input("Category")
        barcode  = r3.text_input("Barcode")
        r4,r5,r6 = st.columns(3)
        cost  = r4.number_input("Cost Price ₹", 0.0, step=0.5)
        price = r5.number_input("Selling Price ₹", 0.0, step=0.5)
        qty   = r6.number_input("Stock Qty", 0, step=1)
        r7,r8,r9 = st.columns(3)
        expiry = r7.date_input("Expiry", value=date.today()+timedelta(days=365))
        sup    = r8.text_input("Supplier")
        reord  = r9.number_input("Reorder Level", 0, step=1, value=5)
        if st.button("💾 Save Item"):
            if not item.strip():
                st.warning("Item name required.")
            else:
                ci = load_csv(INVENTORY_FILE, INV_COLS)
                if item.strip() in ci["item"].values:
                    st.error("Item exists. Use Edit tab.")
                else:
                    ci = pd.concat([ci, pd.DataFrame([[item.strip(),category,cost,price,qty,
                        str(expiry),sup,reord,barcode]], columns=INV_COLS)], ignore_index=True)
                    save_csv(ci, INVENTORY_FILE); st.success("✅ Item added."); reload()

    with tab_edit:
        ci = load_csv(INVENTORY_FILE, INV_COLS)
        sel = st.selectbox("Select item", [""] + ci["item"].dropna().astype(str).tolist())
        
        if sel:
            filtered = ci[ci["item"].astype(str).str.strip() == str(sel).strip()]
            
            if not filtered.empty:
                row = filtered.iloc[0]
            else:
                st.error(f"Item '{sel}' not found in inventory")
                st.stop()

            ec1, ec2, ec3 = st.columns(3)
            nc = ec1.number_input("Cost Price", value=float(pd.to_numeric(row["cost_price"], errors="coerce") or 0), step=0.5)
            np_ = ec2.number_input("Selling Price", value=float(pd.to_numeric(row["unit_price"], errors="coerce") or 0), step=0.5)
            ns = ec3.number_input("Stock", value=int(pd.to_numeric(row["stock"], errors="coerce") or 0))
            nr = st.number_input("Reorder Level", value=int(pd.to_numeric(row["reorder_level"], errors="coerce") or 0))

            sc1, sc2 = st.columns(2)
            if sc1.button("💾 Save"):
                ci.loc[ci["item"] == sel, ["cost_price", "unit_price", "stock", "reorder_level"]] = [nc, np_, ns, nr]
                save_csv(ci, INVENTORY_FILE)
                st.success("Updated.")
                reload()

            if sc2.button("🗑️ Delete"):
                ci = ci[ci["item"] != sel]
                save_csv(ci, INVENTORY_FILE)
                st.success("Deleted.")
                reload()
    with tab_bulk:
        st.info("Upload CSV with columns: item, category, cost_price, unit_price, stock, expiry, supplier, reorder_level, barcode")
        up = st.file_uploader("Upload CSV", type="csv")
        if up:
            try:
                df_up = pd.read_csv(up)
                st.dataframe(df_up.head(), use_container_width=True)
                if st.button("✅ Confirm Import"):
                    ci = load_csv(INVENTORY_FILE, INV_COLS)
                    merged = pd.concat([ci, df_up], ignore_index=True).drop_duplicates(subset=["item"], keep="last")
                    save_csv(merged, INVENTORY_FILE); st.success(f"Imported {len(df_up)} rows."); reload()
            except Exception as e:
                st.error(f"Error: {e}")

# ================================================================
# PAGE: CRM
# ================================================================
elif page == "👥 CRM":
    st.title("👥 Customer Relationship Management")
    cc = load_csv(CUSTOMERS_FILE, CUST_COLS)
    t1,t2,t3 = st.tabs(["All Customers","Add / Edit","Insights"])

    with t1:
        st.metric("Total Customers", len(cc))
        st.dataframe(cc, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export", cc.to_csv(index=False).encode(),"customers.csv","text/csv")

    with t2:
        sel = st.selectbox("Edit or add new", ["— New —"]+cc["name"].tolist())
        pf  = cc[cc["name"]==sel].iloc[0] if sel != "— New —" else None
        c1,c2,c3 = st.columns(3)
        name  = c1.text_input("Name *",  value=str(pf["name"])  if pf is not None else "")
        phone = c2.text_input("Phone",   value=str(pf["phone"]) if pf is not None else "")
        email = c3.text_input("Email",   value=str(pf["email"]) if pf is not None else "")
        pts   = st.number_input("Loyalty Points", 0, 999999, int(pd.to_numeric(pf["points"],errors="coerce") or 0) if pf is not None else 0)
        if st.button("💾 Save Customer"):
            if not name.strip(): st.warning("Name required.")
            else:
                if name in cc["name"].values:
                    cc.loc[cc["name"]==name,["phone","email","points"]] = [phone,email,pts]
                else:
                    cc = pd.concat([cc,pd.DataFrame([[name,phone,email,pts,0,0,today_str()]],columns=CUST_COLS)],ignore_index=True)
                save_csv(cc,CUSTOMERS_FILE); st.success("Saved."); reload()

    with t3:
        if cc.empty: st.info("No data yet.")
        else:
            cc["total_spent"] = pd.to_numeric(cc["total_spent"],errors="coerce").fillna(0)
            cc["points"]      = pd.to_numeric(cc["points"],     errors="coerce").fillna(0)
            bar = alt.Chart(cc.sort_values("total_spent",ascending=False).head(10)).mark_bar(color=ACCENT).encode(
                x=alt.X("total_spent:Q",title="Total Spent ₹"),
                y=alt.Y("name:N",sort="-x",title=""),
                tooltip=["name","total_spent","points","visits"]
            ).properties(height=280,title="Top 10 Customers by Spending")
            st.altair_chart(bar, use_container_width=True)
            st.subheader("Loyalty Leaderboard")
            st.dataframe(cc.sort_values("points",ascending=False)[["name","points","total_spent","visits","last_purchase"]],
                         use_container_width=True,hide_index=True)

# ================================================================
# PAGE: SUPPLIERS
# ================================================================
elif page == "🏭 Suppliers":
    st.title("🏭 Suppliers Management")
    cs = load_csv(SUPPLIERS_FILE, SUP_COLS)
    st.dataframe(cs, use_container_width=True, hide_index=True)
    st.divider()
    s1,s2,s3,s4 = st.columns(4)
    sn = s1.text_input("Name *"); sc2 = s2.text_input("Contact")
    se = s3.text_input("Email"); sr = s4.slider("Reliability",1,10,7)
    si = st.text_input("Items Supplied")
    if st.button("💾 Save Supplier"):
        if not sn.strip(): st.warning("Name required.")
        else:
            if sn in cs["name"].values:
                cs.loc[cs["name"]==sn,["contact","email","reliability","items_supplied"]] = [sc2,se,sr,si]
            else:
                cs = pd.concat([cs,pd.DataFrame([[sn,sc2,se,sr,si]],columns=SUP_COLS)],ignore_index=True)
            save_csv(cs,SUPPLIERS_FILE); st.success("Saved."); reload()

# ================================================================
# PAGE: EMPLOYEES
# ================================================================
elif page == "🧍 Employees":
    st.title("🧍 Employee Management")
    ce = load_csv(EMPLOYEES_FILE, EMP_COLS)
    ca = load_csv(ATTEND_FILE,    ATT_COLS)
    t1,t2,t3 = st.tabs(["Employees","Mark Attendance","Report"])

    with t1:
        st.dataframe(ce, use_container_width=True, hide_index=True)
        e1,e2,e3,e4 = st.columns(4)
        en = e1.text_input("Name *"); er = e2.text_input("Role")
        es = e3.number_input("Salary ₹",0.0,step=500.0); eb = e4.text_input("Branch")
        if st.button("💾 Add Employee"):
            if not en.strip(): st.warning("Name required.")
            else:
                ce = pd.concat([ce,pd.DataFrame([[en,er,es,eb,today_str()]],columns=EMP_COLS)],ignore_index=True)
                save_csv(ce,EMPLOYEES_FILE); st.success("Added."); reload()

    with t2:
        if ce.empty: st.info("Add employees first.")
        else:
            today_present = ca[ca["date"]==today_str()]["name"].tolist()
            for emp in ce["name"].tolist():
                a1,a2 = st.columns([5,2])
                a1.write(f"{'✅' if emp in today_present else '⬜'} {emp}")
                if emp not in today_present:
                    if a2.button("Mark Present", key=f"att_{emp}"):
                        ca = pd.concat([ca,pd.DataFrame([[emp,today_str(),1]],columns=ATT_COLS)],ignore_index=True)
                        save_csv(ca,ATTEND_FILE); st.success(f"{emp} marked."); reload()
                else:
                    a2.markdown('<span class="badge-green">Present</span>', unsafe_allow_html=True)

    with t3:
        if ca.empty: st.info("No records yet.")
        else:
            sm = ca.groupby("name")["present"].sum().reset_index()
            sm.columns = ["name","days_present"]
            st.dataframe(sm, use_container_width=True, hide_index=True)
            bar = alt.Chart(sm).mark_bar(color=ACCENT).encode(
                x=alt.X("days_present:Q",title="Days Present"),
                y=alt.Y("name:N",sort="-x"),
                tooltip=["name","days_present"]
            ).properties(height=220)
            st.altair_chart(bar, use_container_width=True)

# ================================================================
# PAGE: EXPENSES
# ================================================================
elif page == "💰 Expenses":
    st.title("💰 Expense Tracker")
    cx = load_csv(EXPENSES_FILE, EXP_COLS)
    t1,t2 = st.tabs(["Log Expense","View & Export"])

    with t1:
        x1,x2,x3 = st.columns(3)
        xd = x1.date_input("Date", value=date.today())
        xc = x2.selectbox("Category",["Rent","Utilities","Salaries","Supplies","Marketing","Maintenance","Other"])
        xa = x3.number_input("Amount ₹",0.0,step=10.0)
        xn = st.text_input("Note")
        if st.button("💾 Log Expense"):
            if xa <= 0: st.warning("Enter valid amount.")
            else:
                cx = pd.concat([cx,pd.DataFrame([[str(xd),xc,xa,xn]],columns=EXP_COLS)],ignore_index=True)
                save_csv(cx,EXPENSES_FILE); st.success("Logged."); reload()

    with t2:
        if cx.empty: st.info("No expenses yet.")
        else:
            cx["amount"] = pd.to_numeric(cx["amount"],errors="coerce").fillna(0)
            st.metric("Total Expenses", money(cx["amount"].sum()))
            st.dataframe(cx, use_container_width=True, hide_index=True)
            by_cat = cx.groupby("category")["amount"].sum().reset_index()
            pie = alt.Chart(by_cat).mark_arc(outerRadius=100).encode(
                theta="amount:Q", color="category:N", tooltip=["category","amount"]
            ).properties(height=260,title="Expenses by Category")
            st.altair_chart(pie, use_container_width=True)
            st.download_button("⬇️ Export",cx.to_csv(index=False).encode(),"expenses.csv","text/csv")

# ================================================================
# PAGE: AI INSIGHTS (dedicated page)
# ================================================================
elif page == "🤖 AI Insights":
    st.title("🤖 AI-Powered Demand & Business Intelligence")

    st.markdown("""<div class="ai-card">
        <h4>🧠 QuickCart AI Engine</h4>
        <p>Multi-model ensemble: Linear Regression + Random Forest + Gradient Boosting.
        Automatically selects the best model based on available data.
        Features: day-of-week patterns, rolling averages, lag variables, trend detection.</p>
    </div>""", unsafe_allow_html=True)

    t1, t2 = st.tabs(["📦 Demand Forecast", "💡 Smart Insights"])

    with t1:
        st.subheader("Predicted Demand — Next 7 Days")
        if sales.empty:
            st.info("Need sales data to generate a forecast. Make some sales first!")
        else:
            with st.spinner("Running AI models..."):
                forecast = ai_forecast_ensemble(sales)

            if forecast.empty:
                st.warning("Not enough sales history yet (need ≥2 days of data per item).")
            else:
                # Summary metrics
                m1,m2,m3 = st.columns(3)
                m1.metric("Items Forecasted",   len(forecast))
                m2.metric("Need Restocking",    len(forecast[forecast["restock_by"] > 0]))
                rising = len(forecast[forecast["trend"]=="↑ Rising"])
                m3.metric("Rising Demand Items", rising)

                # Colour-coded table
                st.subheader("Forecast Table")
                st.dataframe(
                    forecast.style.applymap(
                        lambda v: "color: #16a34a; font-weight:600" if "Rising" in str(v)
                        else ("color: #dc2626; font-weight:600" if "Falling" in str(v) else ""),
                        subset=["trend"]
                    ),
                    use_container_width=True, hide_index=True
                )

                # Chart
                chart = alt.Chart(forecast).mark_bar().encode(
                    x=alt.X("predicted_units:Q", title="Predicted Units"),
                    y=alt.Y("item:N", sort="-x", title=""),
                    color=alt.Color("trend:N", scale=alt.Scale(
                        domain=["↑ Rising","→ Stable","↓ Falling"],
                        range=[SUCCESS, ACCENT, DANGER]
                    )),
                    tooltip=["item","predicted_units","current_stock","restock_by","confidence","trend","model_used"]
                ).properties(height=max(300, len(forecast)*30), title="AI Demand Forecast by Item")
                st.altair_chart(chart, use_container_width=True)

                # Restock alerts
                restock = forecast[forecast["restock_by"] > 0].sort_values("restock_by", ascending=False)
                if not restock.empty:
                    st.subheader("🔴 Restock Recommendations")
                    for _, row in restock.iterrows():
                        st.markdown(
                            f'<span class="badge-red">Restock</span> &nbsp; '
                            f'<b>{row["item"]}</b> — order <b>{row["restock_by"]} units</b> '
                            f'(current stock: {row["current_stock"]}, predicted demand: {row["predicted_units"]})',
                            unsafe_allow_html=True
                        )
                        st.markdown("")

                st.caption("Model used per item depends on available data: 2+ days → Linear, 5+ days → Random Forest, 8+ days → Gradient Boosting.")

    with t2:
        st.subheader("💡 Business Intelligence Insights")
        if sales.empty:
            st.info("Make some sales to generate insights.")
        else:
            with st.spinner("Analysing your data..."):
                insights = ai_smart_insights(sales, inventory, expenses)
            for insight in insights:
                st.markdown(f"- {insight}")

            # Customer segmentation
            if not customers.empty:
                st.subheader("👥 Customer Segmentation")
                cc2 = customers.copy()
                cc2["total_spent"] = pd.to_numeric(cc2["total_spent"],errors="coerce").fillna(0)
                cc2["visits"]      = pd.to_numeric(cc2["visits"],     errors="coerce").fillna(0)

                def segment(row):
                    if row["total_spent"] > 5000 or row["visits"] > 10: return "VIP 🏆"
                    elif row["total_spent"] > 1000 or row["visits"] > 3: return "Regular ⭐"
                    else: return "New 🆕"

                cc2["segment"] = cc2.apply(segment, axis=1)
                seg_count = cc2["segment"].value_counts().reset_index()
                seg_count.columns = ["segment","count"]
                pie = alt.Chart(seg_count).mark_arc(outerRadius=100).encode(
                    theta="count:Q",
                    color=alt.Color("segment:N", scale=alt.Scale(
                        domain=["VIP 🏆","Regular ⭐","New 🆕"],
                        range=[ACCENT, ACCENT2, "#6b7280"]
                    )),
                    tooltip=["segment","count"]
                ).properties(height=260, title="Customer Segments")
                st.altair_chart(pie, use_container_width=True)
                st.dataframe(cc2[["name","segment","total_spent","visits","points"]],
                             use_container_width=True, hide_index=True)

# ================================================================
# PAGE: REPORTS
# ================================================================
elif page == "📈 Reports":
    st.title("📈 Reports & Export")
    cur_s = load_csv(SALES_FILE, SALES_COLS)

    t1,t2,t3 = st.tabs(["Sales Report","Profit Analysis","Export All"])

    with t1:
        r1,r2 = st.columns(2)
        start = r1.date_input("From", value=date.today()-timedelta(days=30))
        end   = r2.date_input("To",   value=date.today())

        if cur_s.empty: st.info("No sales yet.")
        else:
            cur_s["_date"] = pd.to_datetime(cur_s["datetime"],errors="coerce").dt.date
            cur_s["total"] = pd.to_numeric(cur_s["total"],errors="coerce")
            period = cur_s[(cur_s["_date"]>=start) & (cur_s["_date"]<=end)].drop(columns=["_date"])

            m1,m2,m3 = st.columns(3)
            m1.metric("Revenue",      money(period["total"].sum()))
            m2.metric("Transactions", len(period["invoice"].unique()) if not period.empty else 0)
            m3.metric("Avg Sale",     money(period["total"].mean()) if not period.empty else "₹0.00")

            st.dataframe(period, use_container_width=True, hide_index=True)
            if not period.empty:
                export = pd.concat([period, pd.DataFrame([{**{c:"" for c in period.columns},"item":"TOTAL","total":period["total"].sum()}])], ignore_index=True)
                st.download_button("⬇️ Export Sales CSV", export.to_csv(index=False).encode(), "sales_report.csv","text/csv")

    with t2:
        if cur_s.empty: st.info("No data.")
        else:
            cur_s["profit"] = pd.to_numeric(cur_s["profit"],errors="coerce").fillna(0)
            cur_s["total"]  = pd.to_numeric(cur_s["total"], errors="coerce").fillna(0)
            by_item = cur_s.groupby("item").agg(revenue=("total","sum"),profit=("profit","sum"),units=("qty","sum")).reset_index()
            by_item["margin_%"] = (by_item["profit"]/by_item["revenue"].replace(0,np.nan)*100).round(1)
            st.dataframe(by_item.sort_values("profit",ascending=False), use_container_width=True,hide_index=True)

            scatter = alt.Chart(by_item).mark_circle(size=80).encode(
                x=alt.X("revenue:Q",title="Revenue ₹"),
                y=alt.Y("profit:Q", title="Profit ₹"),
                color=alt.Color("margin_%:Q",scale=alt.Scale(scheme="goldred")),
                size=alt.Size("units:Q"),
                tooltip=["item","revenue","profit","margin_%","units"]
            ).properties(height=320,title="Revenue vs Profit")
            st.altair_chart(scatter, use_container_width=True)

            cx = load_csv(EXPENSES_FILE, EXP_COLS)
            total_exp = pd.to_numeric(cx["amount"],errors="coerce").sum() if not cx.empty else 0
            net = cur_s["profit"].sum() - total_exp
            n1,n2,n3 = st.columns(3)
            n1.metric("Gross Profit",   money(cur_s["profit"].sum()))
            n2.metric("Total Expenses", money(total_exp))
            n3.metric("Net Profit",     money(net))

    with t3:
        st.subheader("Export All Data")
        for label, df, fname in [
            ("Sales",     load_csv(SALES_FILE,     SALES_COLS), "sales.csv"),
            ("Inventory", load_csv(INVENTORY_FILE, INV_COLS),   "inventory.csv"),
            ("Customers", load_csv(CUSTOMERS_FILE, CUST_COLS),  "customers.csv"),
            ("Expenses",  load_csv(EXPENSES_FILE,  EXP_COLS),   "expenses.csv"),
        ]:
            st.download_button(f"⬇️ {label} CSV", df.to_csv(index=False).encode(), fname, "text/csv")
