import streamlit as st
import pandas as pd
import os
import tempfile
import cv2
import numpy as np
from model import predict_mango_agent
import re
import requests

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="Mango Intelligent System", layout="wide")
IMAGE_DIR = "images"
st.markdown("""
<style>

/* Card style */
.card {
    padding: 12px;
    border-radius: 12px;
    background: #1e1e1e;
    border: 1px solid #333;
    transition: 0.3s ease;
}

/* Hover effect */
.card:hover {
    transform: scale(1.05);
    border: 1px solid #4CAF50;
    box-shadow: 0px 4px 15px rgba(76, 175, 80, 0.3);
}

/* Badge style */
.badge-ripe {
    background-color: #4CAF50;
    color: white;
    padding: 3px 8px;
    border-radius: 8px;
    font-size: 12px;
}

.badge-unripe {
    background-color: #FF9800;
    color: white;
    padding: 3px 8px;
    border-radius: 8px;
    font-size: 12px;
}

/* Price box glow */
.price-box {
    padding: 10px;
    border-radius: 10px;
    background: #1e1e1e;
    border: 1px solid #4CAF50;
    transition: 0.3s;
}

.price-box:hover {
    box-shadow: 0px 0px 10px rgba(76, 175, 80, 0.6);
}

</style>
""", unsafe_allow_html=True)
# ==========================
# INFO PAGE STATE
# ==========================
if "started" not in st.session_state:
    st.session_state.started = False
# ==========================
# 💱 LIVE INR → MYR
# ==========================
@st.cache_data(ttl=3600)
def get_myr_rate():
    try:
        url = "https://open.er-api.com/v6/latest/INR"
        res = requests.get(url).json()
        return res["rates"]["MYR"]
    except:
        return 1/23

def convert_inr_to_myr(inr):
    rate = get_myr_rate()
    return round(inr * rate, 2), rate
# ==========================
# 🌐 LIVE EXCHANGE RATE
#  ==========================
# @st.cache_data(ttl=3600)
# def get_myr_rate():
#     try:
#         url = "https://api.exchangerate.host/latest?base=INR&symbols=MYR"
#         res = requests.get(url).json()
#         return res["rates"]["MYR"]
#     except:
#         return 0.042
# ==========================
# 🧾 INFO PAGE
# ==========================
if not st.session_state.started:

    st.title("🥭 Agent-Based Multimodal Mango Quality Assessment and Recommendation System")

    # 🔥 TWO COLUMN LAYOUT
    left, right = st.columns([2.5, 1])

    # ==========================
    # LEFT SIDE (YOUR CURRENT CONTENT)
    # ==========================
    with left:

        st.markdown("""
### 📌 About This Project

This system is an **Agent-Based Multimodal Mango Quality Assessment and Recommendation System**.

### 🔍 What it does:
- 🧠 Uses multiple AI agents for validation, prediction, and recommendation
- 📸 Analyzes mango images (front + back)
- ⚖️ Considers weight for pricing decisions
- 🥭 Predicts:
    - Ripeness (Ripe / Unripe)
    - Quality Grade (Premium / Good / Average)
- 💰 Recommends price in:
    - INR (Indian Rupees)
    - RM (Malaysian Ringgit)
""")

        st.markdown("### 💱 Live Currency Conversion")

        rate = get_myr_rate()

        st.markdown(f"""
• Converts INR → RM using real-time exchange rates  
• Current Rate: **1 INR = {round(rate,4)} MYR**  
• Ensures accurate international pricing  
""")

        st.markdown("""
### ⚙️ System Pipeline:
Input → Validation Agent → Prediction Agent → Recommendation Agent → Output

### 🚀 Modes Available:
- Requirement Mode (text-based query)
- Upload Mode (image-based analysis)

---
👉 Click below to start using the system
""")

        if st.button("🚀 Proceed to System"):
            st.session_state.started = True
            st.rerun()

    with right:

    # Push content slightly up
        st.markdown("<div style='margin-top:-50px'></div>", unsafe_allow_html=True)

    # 🖼️ IMAGE (smaller + aligned)
        img_path = os.path.join("images", "harumanis.jpg")

        if os.path.exists(img_path):
            st.image(
                img_path,
                width=290  # 🔽 smaller size (adjust 240–280 if needed)
            )

    # small spacing control
        st.markdown("<div style='margin-top:-10px'></div>", unsafe_allow_html=True)

        st.markdown("### 🧠 Did You Know?")

        st.markdown("""
<div style="
    padding:14px;
    border-radius:12px;
    background:#1e1e1e;
    border:1px solid #FFA500;
    line-height:1.6;
    font-size:13.5px;
">

🥭 <b>Harumanis Mango</b> is known as the "King of Mangoes" in Malaysia.<br><br>

🌏 It is mainly grown in <b>Perlis, Malaysia</b> and is highly seasonal.<br><br>

💎 Premium Harumanis mangoes are often exported due to their high demand.<br><br>

⏳ Unripe mangoes last longer because they continue ripening after harvest.<br><br>

🧪 Color, texture, and weight are key indicators of mango quality.<br><br>

💰 Prices vary significantly based on <b>grade + ripeness + weight</b>.

</div>
""", unsafe_allow_html=True)

    st.stop()


# ==========================
# LABEL MAP
# ==========================
RIPENESS_MAP = {"K": "Ripe", "P": "Unripe"}
GRADE_MAP = {"P": "Premium", "1": "Good", "2": "Average"}

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    df = pd.read_excel("Harumanis_mango_weight_grade.xlsx")
    df["Color_K-Yellow_P_Green"] = df["Color_K-Yellow_P_Green"].astype(str).str.strip().str.upper()
    df["Fruit Grade"] = df["Fruit Grade"].astype(str).str.strip().str.upper()
    return df

df = load_data()

# ==========================
# PRICE & SHELF
# ==========================
def get_price(g, r):
    if r == "K":
        return {"P": 2000, "1": 1800, "2": 1500}.get(g)
    else:
        return {"P": 1800, "1": 1600, "2": 1300}.get(g)

def get_shelf(r, g):
    shelf_map = {
        ("P", "P"): "10–12 days",  # Premium Unripe
        ("P", "1"): "8–10 days",   # Good Unripe
        ("P", "2"): "6–8 days",   # Average Unripe

        ("K", "P"): "5–7 days",    # Premium Ripe
        ("K", "1"): "3–5 days",    # Good Ripe
        ("K", "2"): "2–3 days"     # Average Ripe
    }

    return shelf_map.get((r, g), "N/A")



# ==========================
# TITLE
# ==========================
st.title("🥭 Agent-Based Multimodal Mango Quality Assessment and Recommendation System")

mode = st.radio("Select Mode", ["Requirement Mode", "Upload Mode"])

# =========================================================
# 🟢 REQUIREMENT MODE
# =========================================================
if mode == "Requirement Mode":

    st.subheader("💬 Describe Your Requirement")

    if "step" not in st.session_state:
        st.session_state.step = 1
    if "budget" not in st.session_state:
        st.session_state.budget = None
    if "ripeness" not in st.session_state:
        st.session_state.ripeness = None

    prompt = st.text_input("Enter your requirement(in ₹)", placeholder="e.g. I want mango under 2000")

    if st.button("Submit Requirement"):

        match = re.search(r'\d+', prompt)

        if match:
            st.session_state.budget = int(match.group())
            st.session_state.step = 2
        else:
            st.error("❌ Please mention a budget (e.g. 1500)")

    if st.session_state.step == 2:

        st.info("🤖 Do you prefer ripe or unripe mango?")

        st.markdown("""
<style>
.chip {
    display: inline-block;
    padding: 10px 18px;
    margin-right: 10px;
    border-radius: 20px;
    background: #1e1e1e;
    border: 1px solid #444;
    cursor: pointer;
    transition: 0.3s;
    font-weight: 500;
}
.chip:hover {
    border: 1px solid #4CAF50;
    transform: scale(1.05);
}
.chip-active-ripe {
    background: linear-gradient(45deg, #FF9800, #FFC107);
    color: white;
    border: none;
}
.chip-active-unripe {
    background: linear-gradient(45deg, #4CAF50, #81C784);
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# default
        if "choice" not in st.session_state:
            st.session_state.choice = "Ripe"

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🟡 Ripe", use_container_width=True):
                st.session_state.choice = "Ripe"

        with col2:
            if st.button("🟢 Unripe", use_container_width=True):
                st.session_state.choice = "Unripe"

# show selected chip
        if st.session_state.choice == "Ripe":
            st.markdown('<div class="chip chip-active-ripe">🟡 Ripe Selected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chip chip-active-unripe">🟢 Unripe Selected</div>', unsafe_allow_html=True)

        choice = st.session_state.choice

        if st.button("Confirm Preference"):
            st.session_state.ripeness = "K" if choice == "Ripe" else "P"
            st.session_state.step = 3

    if st.session_state.step == 3:

        data = df.copy()

        data["Price"] = data.apply(
            lambda r: get_price(r["Fruit Grade"], r["Color_K-Yellow_P_Green"]),
            axis=1
        )

        category_exists = data[data["Color_K-Yellow_P_Green"] == st.session_state.ripeness]

        filtered = data[
            (data["Price"] <= st.session_state.budget) &
            (data["Color_K-Yellow_P_Green"] == st.session_state.ripeness)
        ]

        if len(category_exists) > 0 and len(filtered) == 0:
            st.error(f"❌ No mango found under ₹{st.session_state.budget}")
            st.stop()

        if len(category_exists) == 0:
            st.warning("⚠️ No mango available for selected preference")
            st.stop()

        selected_type = "Ripe" if st.session_state.ripeness == "K" else "Unripe"
        badge_color = "#FF9800" if selected_type == "Ripe" else "#4CAF50"

        st.markdown(f"""
<h2>🥭 Recommended Mangoes 
<span style="
    background:{badge_color};
    color:white;
    padding:4px 10px;
    border-radius:10px;
    font-size:14px;
    margin-left:10px;
">
{selected_type}
</span>
</h2>
""", unsafe_allow_html=True)
        st.info(f"Showing {selected_type} mangoes under ₹{st.session_state.budget}")
        for grade in ["P", "1", "2"]:

            group = filtered[filtered["Fruit Grade"] == grade]

            if len(group) == 0:
                continue

            st.subheader(f"🏆 {GRADE_MAP[grade]} Quality")

            c1, c2 = st.columns([3, 1])

            with c1:
                cols = st.columns(5)
                for i, name in enumerate(group["Fruit No"].head(5)):
                    path = os.path.join(IMAGE_DIR, str(name))
                    if os.path.exists(path):
                        cols[i].image(path, width=120)

            with c2:
                price = int(group["Price"].iloc[0])
                price_rm, rate = convert_inr_to_myr(price)

                st.markdown(f"""
<div class="price-box">
💰 <b>₹{price}/kg</b> &nbsp; | &nbsp; 💱 <b>RM {price_rm}/kg</b>
<br><span style="font-size:12px;color:#aaa;">
(1 INR = {round(rate,4)} MYR)
</span>
</div>
""", unsafe_allow_html=True)

                st.caption("💱 RM = Malaysian Ringgit (Live conversion)")

                st.write(f"🧊 Shelf Life: {get_shelf(st.session_state.ripeness, grade)}")
                st.caption("Shelf life is estimated from the day of delivery under normal storage conditions.")
                if st.button("🛒 Order This Mango", key=f"req_order_{grade}",use_container_width=True):
                    st.success(f"🎉 Order placed for {GRADE_MAP[grade]} mango!")
                    st.info("📦 Delivery in 2–3 days")
                    st.balloons()

    if st.button("🔄 Reset"):
        st.session_state.step = 1
        st.session_state.budget = None
        st.session_state.ripeness = None

# =========================================================
# 🔵 UPLOAD MODE
# =========================================================
else:

    st.subheader("📸 Upload Mango Images")

    role = st.radio("Select Role", ["Buyer", "Seller"])

    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "validation_done" not in st.session_state:
        st.session_state.validation_done = False
    if "last_inputs" not in st.session_state:
        st.session_state.last_inputs = None

    col1, col2 = st.columns(2)

    with col1:
        front = st.file_uploader("Front Image", type=["jpg", "jpeg", "png"])

    with col2:
        back = st.file_uploader("Back Image", type=["jpg", "jpeg", "png"])

    weight = st.number_input("Weight (grams)", 300, 700, 500)

    current_inputs = (
        role,
        front.name if front else None,
        back.name if back else None,
        weight
    )

    if st.session_state.last_inputs is None:
        st.session_state.last_inputs = current_inputs

    if current_inputs != st.session_state.last_inputs:
        st.session_state.analysis = None
        st.session_state.validation_done = False
        st.session_state.last_inputs = current_inputs
        st.info("ℹ️ Inputs changed. Please run validation again.")

    if front and back:
        p1, p2 = st.columns(2)
        p1.image(front, width=220)
        p2.image(back, width=220)

    def save_file(file):
        bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp.name, img)
        return temp.name

    if st.button("🚀 Run Input Validation Agent"):

        if not front or not back:
            st.error("❌ Please upload both images.")
        else:
            f = save_file(front)
            b = save_file(back)

            res = predict_mango_agent(f, b, weight)

            st.session_state.analysis = res
            st.session_state.validation_done = True

    if st.session_state.validation_done:

        res = st.session_state.analysis

        st.divider()
        st.markdown("## 🧠 Input Validation Agent")

        if res["error"]:
            st.error(res["message"])
        else:
            st.success("✅ All validation checks passed")

            st.markdown("""
<div style="padding:18px;border-radius:14px;background:#1e1e1e;border:1px solid #4CAF50;">
✔ Blur Check Passed<br>
✔ Brightness Check Passed<br>
✔ Duplicate Check Passed<br>
✔ Same Mango Check Passed
</div>
""", unsafe_allow_html=True)

            st.divider()
            st.markdown("## ⚡ Next Step: Prediction Agent")

            run_pred = st.radio("Proceed with prediction?", ["No", "Yes"], horizontal=True)

            if run_pred == "Yes":

                st.markdown("## 🤖 Prediction Results")
                def result_card(title, value):
                    st.markdown(f"""
    <div style="
    padding:15px;
    border-radius:12px;
    background:#1e1e1e;
    border:1px solid #333;
    text-align:center;
    ">
    <h4>{title}</h4>
    <h3>{value}</h3>
    </div>
    """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    result_card("🍃 Ripeness", RIPENESS_MAP[res['ripeness']])
                with c2:
                    result_card("🏆 Grade", GRADE_MAP[res['grade']])
                with c3:
                    result_card("🎯 Confidence", f"{round(res['confidence']*100,2)}%")

                st.divider()
                st.markdown("## ⚡ Next Step: Recommendation Agent")

                run_rec = st.radio("Proceed with recommendation?", ["No", "Yes"], horizontal=True)

                if run_rec == "Yes":

                    st.markdown("## 🛒 Recommendation Summary")

                    price = res["price"]
                    price_rm, rate = convert_inr_to_myr(price)

                    c1, c2 = st.columns(2)
                    c1.markdown(f"### 💰 Price (INR)\n**₹{price}/kg**")
                    c2.markdown(f"""
### 💱 Price (RM)
**RM {price_rm}/kg**

<span style="font-size:12px;color:gray;">
(1 INR = {round(rate,4)} MYR)
</span>
""", unsafe_allow_html=True)

                    st.caption("💱 RM = Malaysian Ringgit (Live conversion from INR)")

                    st.success(f"🧊 Shelf Life: {get_shelf(res['ripeness'], res['grade'])}")

                    st.caption("Shelf life is estimated from the day of delivery under normal storage conditions.")
                    st.subheader("🥭 Similar Mangoes")

                    similar = df[
                        (df["Fruit Grade"] == res["grade"]) &
                        (df["Color_K-Yellow_P_Green"] == res["ripeness"])
                    ]

                    cols = st.columns(5)

                    for i, name in enumerate(similar["Fruit No"].head(5)):
                        for ext in ["", ".jpg", ".png"]:
                            path = os.path.join(IMAGE_DIR, str(name) + ext)
                            if os.path.exists(path):
                                cols[i].image(path, width=120)
                                break

                    st.divider()

                    if role == "Buyer":

                        st.markdown("## 🛒 Buyer Insight")

                        if res["grade"] == "P":
                            st.success("⭐ Premium quality. Highly recommended.")
                        elif res["grade"] == "1":
                            st.info("👍 Good quality. Worth buying.")
                        else:
                            st.warning("⚠️ Average quality. Negotiate price.")

                        if st.button("🛒 Order This Mango", key="upload_order"):
                            st.success("🎉 Order placed successfully!")
                            st.info("📦 Delivery in 2–3 days")
                            st.balloons()

                    else:

                        st.markdown("## 📈 Seller Insight")

                        if res["grade"] == "P":
                            st.success("🚀 Premium quality. Target export market.")
                        elif res["grade"] == "1":
                            st.info("📦 Good quality. Sell in retail.")
                        else:
                            st.warning("⚠️ Lower grade. Sell quickly.")