import pandas as pd
import streamlit as st

from services import api_client
from services.ui import ensure_session, show_api_error, sync_pending_view

st.title("Signal Rules")
st.caption("The full rulebook — every condition this app records as a positive, negative, or neutral implicit signal.")

ensure_session()
sync_pending_view()

result = api_client.get_signal_catalog()
if "error" in result:
    show_api_error(result)
    st.stop()

df = pd.DataFrame(result["rules"])

SIGNAL_SECTIONS = [
    ("positive", "✅ Positive signals", "Ways a user shows they like or want a book."),
    ("negative", "🚫 Negative signals", "Ways a user shows disinterest — the ten rules for implicit negative feedback."),
    ("neutral", "➖ Neutral (raw signals)", "Not scored themselves — the raw substrate other signals are derived from."),
]

for signal, heading, caption in SIGNAL_SECTIONS:
    st.subheader(f"{heading} ({len(df[df['signal'] == signal])})")
    st.caption(caption)
    section = df[df["signal"] == signal][["event_type", "kind", "reason"]].rename(
        columns={"event_type": "Event type", "kind": "Implicit / Explicit", "reason": "Condition"}
    )
    st.dataframe(section, width="stretch", hide_index=True)
    st.divider()
