import streamlit as st
import pickle
import pandas as pd
import math
from datetime import timedelta
from xgboost import XGBRegressor

st.set_page_config(page_title="Olist Store - Delivery Prediction", layout="wide")

# ---------- LOAD PKL ----------
model = XGBRegressor()
model.load_model('model.json')

with open('catalog.pkl','rb') as f:
    CATALOG = pickle.load(f)
with open('seller_avg.pkl','rb') as f:
    seller_avg = pickle.load(f)
with open('category_avg.pkl','rb') as f:
    category_avg = pickle.load(f)
with open('config.pkl','rb') as f:
    config = pickle.load(f)

# ---------- CITY COORDS ----------
CITY_COORDS = {
    "Sao Paulo, SP": (-23.55, -46.63),
    "Rio de Janeiro, RJ": (-22.90, -43.20),
    "Belo Horizonte, MG": (-19.91, -43.93),
    "Curitiba, PR": (-25.42, -49.27),
    "Porto Alegre, RS": (-30.03, -51.23),
    "Salvador, BA": (-12.97, -38.50),
    "Fortaleza, CE": (-3.73, -38.52),
    "Brasilia, DF": (-15.79, -47.88),
    "Recife, PE": (-8.05, -34.90),
    "Manaus, AM": (-3.11, -60.02),
    "Goiania, GO": (-16.68, -49.25),
    "Campinas, SP": (-22.90, -47.06),
    "Santo Andre, SP": (-23.65, -46.53),
    "Santos, SP": (-23.96, -46.33),
    "Sao Jose dos Campos, SP": (-23.18, -45.88),
}

def get_coords(city_state):
    if city_state in CITY_COORDS:
        return CITY_COORDS[city_state]
    return CITY_COORDS["Sao Paulo, SP"]

def haversine(c1, c2):
    lat1, lon1 = c1
    lat2, lon2 = c2
    R = 6371
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def calc_freight(distance, weight):
    if distance < 100: base = 10
    elif distance < 500: base = 18
    elif distance < 1500: base = 28
    else: base = 45
    if weight > 1500: base += 8
    return float(base)

# ---------- CSS - SQUARE BOXES ----------
st.markdown("""
<style>
.item-box {
    border: 2px solid #E0E0E0;
    border-radius: 12px;
    padding: 18px 10px;
    text-align: center;
    background: white;
    margin-bottom: 8px;
}
.item-box-selected {
    border: 2px solid #2874F0;
    background: #E8F0FE;
    border-radius: 12px;
    padding: 18px 10px;
    text-align: center;
    margin-bottom: 8px;
}
.item-box h3, .item-box-selected h3 { margin: 0; color: #111; }
.item-box p, .item-box-selected p { margin: 4px 0 0 0; color: #333; }
</style>
""", unsafe_allow_html=True)

# ---------- UI ----------
st.title("Olist Store - Delivery Time Prediction")

left, right = st.columns([1.2, 1], gap="large")

# ================= LEFT : CUSTOMER =================
with left:
    st.subheader("1. Choose Category")
    category = st.selectbox("Product Category", options=sorted(list(CATALOG.keys())), key="cat_select")

    # Category change hui toh selection reset karo
    if 'last_category' not in st.session_state or st.session_state.last_category != category:
        st.session_state.last_category = category
        st.session_state.selected_idx = 0
        if 'result' in st.session_state:
            del st.session_state.result

    raw = CATALOG[category]
    if isinstance(raw, dict):
        options = [raw]
    else:
        options = raw

    # ---------- CHOOSE ITEM - SQUARE BOXES ----------
    st.subheader("Choose Item")

    # 3 boxes ek line me
    cols = st.columns(len(options))

    for i, o in enumerate(options):
        with cols[i]:
            is_selected = (st.session_state.get('selected_idx', 0) == i)
            # Box ka look
            box_class = "item-box-selected" if is_selected else "item-box"
            tick = "✅ Selected" if is_selected else "Click to Select"
            st.markdown(f"""
            <div class="{box_class}">
                <h3>Item {i+1}</h3>
                <p><b>R$ {float(o['median_price']):.2f}</b></p>
                <p style="font-size:13px;">{tick}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select Item {i+1}", key=f"item_btn_{category}_{i}", use_container_width=True):
                st.session_state.selected_idx = i
                if 'result' in st.session_state:
                    del st.session_state.result
                st.rerun()

    # Selected item ka data - customer ko sirf price dikhega, baaki kuch nahi
    selected_idx = st.session_state.get('selected_idx', 0)
    if selected_idx >= len(options):
        selected_idx = 0
    info = options[selected_idx]

    price = float(info['median_price'])
    weight = float(info['median_weight'])
    volume = float(info['median_volume'])
    seller_id = info['seller_id']
    seller_city = f"{info['seller_city']}, {info['seller_state']}"

    st.write(f"You selected: **Item {selected_idx+1}** - **R$ {price:.2f}**")

    st.subheader("2. Place Order")
    customer_city = st.selectbox("Deliver To - Your City", options=sorted(list(CITY_COORDS.keys())))
    qty = st.number_input("Quantity", min_value=1, max_value=5, value=1)
    purchase_date = st.date_input("Order Date")

    place_order = st.button("Place Order - Check Delivery Date", type="primary", use_container_width=True)

    if place_order:
        seller_coords = get_coords(seller_city)
        customer_coords = CITY_COORDS[customer_city]
        distance_km = haversine(seller_coords, customer_coords)
        freight = calc_freight(distance_km, weight)

        s_avg = seller_avg.get(seller_id, config['global_avg'])
        c_avg = category_avg.get(category, config['global_avg'])

        data = pd.DataFrame([{
            'distance_km': distance_km,
            'purchase_month': purchase_date.month,
            'same_city': 1 if seller_city == customer_city else 0,
            'total_freight': freight,
            'total_items': qty,
            'product_volume': volume,
            'product_weight_g': weight,
            'freight_per_price': freight / price if price else 0,
            'purchase_day': purchase_date.day,
            'seller_avg_delivery': s_avg,
            'category_avg_delivery': c_avg
        }])[config['features']]

        pred = float(model.predict(data)[0])
        delivery_date = purchase_date + timedelta(days=int(round(pred)))

        st.session_state.result = {
            'pred': pred, 'delivery_date': delivery_date, 'distance': distance_km,
            's_avg': s_avg, 'c_avg': c_avg, 'seller_city': seller_city,
            'customer_city': customer_city, 'category': category,
            'price': price, 'weight': weight, 'volume': volume,
            'freight': freight, 'purchase_date': purchase_date, 'qty': qty,
            'seller_id': seller_id, 'selected_item': f"Item {selected_idx+1}"
        }

    if 'result' in st.session_state:
        r = st.session_state.result
        st.success(f"### Delivery Expected By: {r['delivery_date'].strftime('%A, %d %B %Y')}")
        st.write(f"In **{r['pred']:.1f} days** | Order on {r['purchase_date'].strftime('%d %b %Y')} | Deliver to **{r['customer_city']}**")
        st.write(f"Total to Pay: **R$ {r['price']*r['qty'] + r['freight']:.2f}** (Product R$ {r['price']*r['qty']:.2f} + Freight R$ {r['freight']:.0f})")

# ================= RIGHT : MANAGER =================
with right:
    st.subheader("Manager Panel - Logistics Details")

    if 'result' not in st.session_state:
        st.write("Place an order from left side. Manager details will appear here.")
        st.write("- Distance & Route")
        st.write("- Freight calculation")
        st.write("- Seller vs Category average")
        st.write("- Delay risk check")
    else:
        r = st.session_state.result
        is_delay = r['pred'] > r['s_avg'] + 2
        status = "Delay Risk" if is_delay else "On Time"

        st.write(f"**Status:** {status}")
        st.write(f"**Category:** {r['category']}")
        st.write(f"**Selected:** {r['selected_item']}")
        st.write(f"**From (Seller):** {r['seller_city']}")
        st.write(f"**Seller ID:** `{r['seller_id']}`")
        st.write(f"**To (Customer):** {r['customer_city']}")
        st.write("---")
        st.write(f"**Distance:** {r['distance']:.0f} km")
        st.write(f"**Weight:** {r['weight']:.0f} g | **Volume:** {r['volume']:.0f} cm³ | **Qty:** {r['qty']}")
        st.write(f"**Price (Median):** R$ {r['price']:.2f}")
        st.write(f"**Freight (Auto):** R$ {r['freight']:.0f}")
        st.write(f"**Seller Avg Delivery:** {r['s_avg']:.1f} days")
        st.write(f"**Category Avg Delivery:** {r['c_avg']:.1f} days")
        st.write(f"**Global Avg:** {config['global_avg']:.1f} days")
        st.write("---")
        st.write(f"**Model Prediction:** {r['pred']:.1f} days")
        st.write(f"**Order:** {r['purchase_date'].strftime('%d %b %Y')} -> **Delivery:** {r['delivery_date'].strftime('%d %b %Y')}")
        st.caption("Note: Price / Weight / Volume are dataset medians from catalog.pkl for the chosen seller, not user input.")