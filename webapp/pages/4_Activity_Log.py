import pandas as pd
import streamlit as st

from services import api_client
from services.ui import ensure_session, show_api_error, sync_pending_view

st.title("Activity Log")
st.caption("Every implicit signal recorded for a user — impressions, views, wishlist/cart/checkout, ratings, and the ten negative signals (bounces, dismissals, etc).")

ensure_session()
sync_pending_view()

users_result = api_client.list_users(limit=300)
if "error" in users_result:
    show_api_error(users_result)
    st.stop()
demo_users = users_result.get("users", [])
if not demo_users:
    st.info("No users yet — sign up or seed the database first.")
    st.stop()

current_user = st.session_state.get("user_id")
default_index = demo_users.index(current_user) if current_user in demo_users else 0
selected_user = st.selectbox("Select a user", demo_users, index=default_index)

limit = st.slider("How many recent events to load", min_value=20, max_value=1000, value=200, step=20)

result = api_client.get_user_events(selected_user, limit=limit)
if "error" in result:
    show_api_error(result)
    st.stop()

events = result["events"]
if not events:
    st.caption(f"No events recorded yet for **{selected_user}**.")
    st.stop()

df = pd.DataFrame(events)
df["created_at"] = pd.to_datetime(df["created_at"])

counts = df["signal"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total events", len(df))
c2.metric("Positive", int(counts.get("positive", 0)))
c3.metric("Negative", int(counts.get("negative", 0)))
c4.metric("Neutral", int(counts.get("neutral", 0)))

st.divider()
signal_filter = st.multiselect("Filter by signal", options=["positive", "negative", "neutral"], default=["positive", "negative", "neutral"])
filtered = df[df["signal"].isin(signal_filter)].sort_values("created_at", ascending=False)

st.subheader("By event type")
st.bar_chart(filtered["event_type"].value_counts())

st.subheader(f"Event log ({len(filtered)} of {len(df)})")
display_cols = ["created_at", "event_type", "signal", "reason", "item_title", "parent_asin", "position", "value"]
st.dataframe(
    filtered[display_cols], width="stretch", hide_index=True,
    column_config={"reason": st.column_config.TextColumn("Why", width="large")},
)
