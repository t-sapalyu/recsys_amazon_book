import streamlit as st

from services.api_client import get_because_you_liked, get_personalized, get_popular, get_similar_users_liked
from services.ui import render_item_row, show_api_error

st.title("Homepage")

user_id = st.session_state.get("user_id")

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh recommendations"):
        st.rerun()
st.caption("Recommendations are temperature-sampled from a wider candidate pool — refresh to see them vary.")

if user_id is None:
    st.subheader("Trending right now")
    st.caption("Log in from the sidebar to see personalized recommendations instead.")
    result = get_popular(k=10)
    if "error" in result:
        show_api_error(result)
    else:
        render_item_row(result["items"], result["source_model"])

else:
    st.subheader(f"Recommended for you")
    result = get_personalized(user_id, k=10)
    if "error" in result:
        show_api_error(result)
    else:
        render_item_row(result["items"], result["source_model"])

    st.divider()
    st.subheader("Because you liked...")
    byl = get_because_you_liked(user_id, k=5)
    if "error" in byl:
        show_api_error(byl)
    else:
        because_of = byl["because_of"]
        st.caption(f"Based on: **{because_of.get('title') or because_of['parent_asin']}**")
        render_item_row(byl["items"], byl["source_model"])

    st.divider()
    st.subheader("People like you also like")
    st.caption("Readers whose taste (by learned embedding) resembles yours — not items close to your own embedding, but what *they* actually liked.")
    similar_users = get_similar_users_liked(user_id, k=5)
    if "error" in similar_users:
        show_api_error(similar_users)
    else:
        render_item_row(similar_users["items"], similar_users["source_model"])
