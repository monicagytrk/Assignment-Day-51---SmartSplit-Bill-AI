"""
Streamlit Controller
--------------------
3-step flow: Upload, Assign, and Report
"""

import os
import streamlit as st
from PIL import Image

from modules.receipt_reader import read_receipt
from modules.splitter import calculate_split
from modules.models import BillData, BillItem, AdditionalCharge



VIRIDIS = ["#440154", "#31688e", "#35b779", "#fde725", "#21918c", "#90d743"]


def _text_color(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"


def _get_groq_key() -> str:
    return "INSERT_YOUR_API_KEY"



def _setup_page():
    st.set_page_config(
        page_title="SplitSmart — AI Bill Splitter",
        page_icon="🧾",
        layout="wide",
    )
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800; }
    .stButton > button {
        background-color: #111; color: #fff;
        border-radius: 8px; font-family: 'Syne', sans-serif;
        font-weight: 700; padding: 0.5rem 1.5rem;
        border: none; transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #e91e8c; color: #fff; }
    .step-badge {
        display: inline-block; background: #111; color: #ffffff;
        font-family: 'Syne', sans-serif; font-weight: 800;
        font-size: 0.75rem; letter-spacing: 2px;
        padding: 2px 10px; border-radius: 20px;
        margin-bottom: 8px; text-transform: uppercase;
    }
    .person-card {
        color: #fff; border-radius: 12px;
        padding: 1rem 1.5rem; margin-bottom: 0.75rem;
    }
    .person-name { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.1rem; }
    .person-amount { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.6rem; }
    </style>
    """, unsafe_allow_html=True)


def _init_state():
    defaults = {
        "step": 1,
        "bill": None,
        "participants": [],
        "split_result": None,
        "inference_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _fmt(amount: float) -> str:
    return f"Rp {amount:,.0f}"


def _fmt_qty(qty: float) -> str:
    return f"{qty:.2f}"


def _step_upload():
    st.markdown('<div class="step-badge">Step 1 of 3</div>', unsafe_allow_html=True)
    st.title("📸 Upload Receipt")
    st.caption("Upload a photo of your receipt and let AI read it automatically.")

    uploaded = st.file_uploader(
        "Upload receipt photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Receipt", use_container_width=True)

        st.divider()
        if st.button("🤖 Read Receipt with AI", use_container_width=True):
            with st.spinner("AI is reading your receipt..."):
                try:
                    groq_key = _get_groq_key()
                    bill, elapsed = read_receipt(
                        image=image,
                        backend="groq",
                        groq_api_key=groq_key,
                    )
                    st.session_state.bill = bill
                    st.session_state.inference_time = elapsed
                    st.session_state.step = 2
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")


def _step_assign():
    bill: BillData = st.session_state.bill

    st.markdown('<div class="step-badge">Step 2 of 3</div>', unsafe_allow_html=True)
    st.title("✏️ Review & Assign Items")

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.subheader("🧾 Bill Items")
        if st.session_state.inference_time:
            st.caption(f"AI read time: {st.session_state.inference_time:.2f}s")

        participants = st.session_state.participants

        for i, item in enumerate(bill.items):
            with st.expander(f"**{item.name}** — {_fmt(item.total_price)}", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    item.name = st.text_input("Item name", item.name, key=f"iname_{i}")
                with c2:
                    item.quantity = st.number_input("Qty", value=item.quantity, min_value=0.0, step=0.5, key=f"iqty_{i}")
                with c3:
                    item.price_per_item = st.number_input("Price/item", value=item.price_per_item, min_value=0.0, step=100.0, key=f"ipp_{i}")
                item.total_price = item.quantity * item.price_per_item

                if participants:
                    item.assigned_to = st.multiselect(
                        "Assigned to (empty = split equally among all)",
                        options=participants,
                        default=[p for p in item.assigned_to if p in participants],
                        key=f"assign_{i}",
                    )
                else:
                    st.caption("⚠️ Add participants on the right panel first.")

        st.divider()
        st.subheader("➕ Additional Charges")
        for j, charge in enumerate(bill.additional_charges):
            c1, c2 = st.columns([2, 1])
            with c1:
                charge.name = st.text_input("Charge name", charge.name, key=f"cname_{j}")
            with c2:
                charge.amount = st.number_input("Amount", value=charge.amount, step=100.0, key=f"camt_{j}")

        if st.button("➕ Add Charge"):
            bill.additional_charges.append(AdditionalCharge("New Charge", 0.0))
            st.rerun()

        st.divider()
        bill.subtotal = sum(it.total_price for it in bill.items)
        calculated_total = bill.subtotal + sum(c.amount for c in bill.additional_charges)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Subtotal", _fmt(bill.subtotal))
        with c2:
            bill.total = st.number_input(
                "Grand Total (editable)",
                value=calculated_total,
                step=100.0,
                key="grand_total",
            )

    with col_right:
        st.subheader("👥 Participants")
        new_name = st.text_input("Add participant name", key="new_participant")
        if st.button("➕ Add Person") and new_name.strip():
            name = new_name.strip()
            if name not in st.session_state.participants:
                st.session_state.participants.append(name)
                st.rerun()
            else:
                st.warning(f"'{name}' already added.")

        if st.session_state.participants:
            st.divider()
            for p in list(st.session_state.participants):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.write(f"👤 {p}")
                with c2:
                    if st.button("✕", key=f"del_{p}"):
                        st.session_state.participants.remove(p)
                        for item in bill.items:
                            if p in item.assigned_to:
                                item.assigned_to.remove(p)
                        st.rerun()
        else:
            st.info("No participants yet. Add names above.")

    st.divider()
    c1, _, c2 = st.columns([1, 2, 1])
    with c1:
        if st.button("← Back"):
            st.session_state.step = 1
            st.rerun()
    with c2:
        if st.button("Calculate Split →", use_container_width=True):
            if not st.session_state.participants:
                st.error("Please add at least one participant.")
            else:
                result = calculate_split(bill, st.session_state.participants)
                st.session_state.split_result = result
                st.session_state.step = 3
                st.rerun()


def _step_report():
    bill: BillData = st.session_state.bill
    split: dict = st.session_state.split_result

    st.markdown('<div class="step-badge">Step 3 of 3</div>', unsafe_allow_html=True)
    st.title("📊 Split Report")
    st.divider()

    cols = st.columns(min(len(split), 3))
    for idx, (person, amount) in enumerate(split.items()):
        color = VIRIDIS[idx % len(VIRIDIS)]
        text_color = _text_color(color)
        with cols[idx % len(cols)]:
            st.markdown(f"""
            <div class="person-card" style="background: {color};">
                <div class="person-name" style="color: {text_color};">👤 {person}</div>
                <div class="person-amount" style="color: {text_color};">{_fmt(amount)}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.subheader("📋 Item Breakdown")
    table_data = []
    for item in bill.items:
        table_data.append({
            "Item": item.name,
            "Qty": _fmt_qty(item.quantity),
            "Price/Item": _fmt(item.price_per_item),
            "Total": _fmt(item.total_price),
            "Paid By": ", ".join(item.assigned_to) if item.assigned_to else "All (equal split)",
        })
    st.table(table_data)

    if bill.additional_charges:
        st.subheader("💸 Additional Charges")
        for c in bill.additional_charges:
            st.write(f"- **{c.name}**: {_fmt(c.amount)}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Subtotal", _fmt(bill.subtotal))
    with c2:
        st.metric("Grand Total", _fmt(bill.total))

    st.divider()
    c1, _, c2 = st.columns([1, 2, 1])
    with c1:
        if st.button("← Edit Assignments"):
            st.session_state.step = 2
            st.rerun()
    with c2:
        if st.button("🔄 Start Over", use_container_width=True):
            for key in ["bill", "participants", "split_result", "inference_time"]:
                st.session_state.pop(key, None)
            st.session_state.step = 1
            st.rerun()


def controller():
    _setup_page()
    _init_state()

    with st.sidebar:
        st.markdown("### 🧾 SplitSmart")
        st.caption("AI-powered bill splitting")
        st.caption("Powered by **Groq + LLaMA 4 Scout**")
        st.divider()
        steps = ["📸 Upload Receipt", "✏️ Assign Items", "📊 Final Report"]
        for i, label in enumerate(steps, 1):
            if st.session_state.step > i:
                icon = "✅"
            elif st.session_state.step == i:
                icon = "▶️"
            else:
                icon = "⬜"
            st.markdown(f"{icon} **Step {i}:** {label}")

        if st.session_state.bill:
            st.divider()
            st.caption(f"Items found: {len(st.session_state.bill.items)}")
            st.caption(f"Participants: {len(st.session_state.participants)}")
            if st.session_state.inference_time:
                st.caption(f"AI read time: {st.session_state.inference_time:.2f}s")

    step = st.session_state.step
    if step == 1:
        _step_upload()
    elif step == 2:
        _step_assign()
    elif step == 3:
        _step_report()