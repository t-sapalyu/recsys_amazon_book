import streamlit as st

from services.api_client import get_because_you_liked, get_discover, get_personalized, get_popular, get_recent, get_similar_users_liked
from services.ui import render_item_row, show_api_error, sync_pending_view

st.title("Homepage")

sync_pending_view()

user_id = st.session_state.get("user_id")

# Recommendations are fetched once and cached per user, recomputed only on
# an explicit refresh. Without this, EVERY rerun of this page — including
# the one triggered just by clicking a card's "View details" button —
# re-ran all of these model-backed calls (two-tower retrieval + LightGBM
# ranking, real inference against the full catalog) from scratch. Against
# the real models that's slow enough that clicking a card could get
# superseded by a later interaction before Streamlit ever finished
# processing that click — which looked exactly like "clicking just
# refreshes the homepage". The "temperature-sampled ... refresh to see them
# vary" caption already implies this is meant to vary on explicit refresh,
# not on every incidental rerun.
cache_key = f"homepage_cache_{user_id or 'guest'}"

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh recommendations"):
        st.session_state.pop(cache_key, None)
        st.rerun()
st.caption("Recommendations are temperature-sampled from a wider candidate pool — refresh to see them vary.")

if cache_key not in st.session_state:
    cached = {}
    if user_id is None:
        result = get_discover(trending_k=20, random_k=10)
        if "error" in result:
            cached["discover_error"] = result
            cached["popular_fallback"] = get_popular(k=10)
        else:
            cached["discover"] = result
    else:
        cached["personalized"] = get_personalized(user_id, k=10)
        cached["because_you_liked"] = get_because_you_liked(user_id, k=5)
        cached["similar_users_liked"] = get_similar_users_liked(user_id, k=5)
    cached["recent"] = get_recent(k=10)
    st.session_state[cache_key] = cached

cached = st.session_state[cache_key]

if user_id is None:
    st.subheader("Trending right now")
    st.caption("Log in from the sidebar to see personalized recommendations instead.")
    if "discover_error" in cached:
        show_api_error(cached["discover_error"])
        popular_fallback = cached.get("popular_fallback", {})
        if "error" not in popular_fallback:
            render_item_row(popular_fallback["items"], popular_fallback["source_model"])
    else:
        result = cached["discover"]
        render_item_row(result["trending"], result["trending_source_model"])

        st.divider()
        st.subheader("Something different")
        st.caption("From categories none of the books above belong to — a bit of catalog breadth beyond what's trending.")
        render_item_row(result["discover"], result["discover_source_model"])

else:
    st.subheader("Recommended for you")
    result = cached["personalized"]
    if "error" in result:
        show_api_error(result)
    else:
        render_item_row(result["items"], result["source_model"])

    st.divider()
    st.subheader("Because you liked...")
    byl = cached["because_you_liked"]
    if "error" in byl:
        show_api_error(byl)
    else:
        because_of = byl["because_of"]
        st.caption(f"Based on: **{because_of.get('title') or because_of['parent_asin']}**")
        render_item_row(byl["items"], byl["source_model"])

    st.divider()
    st.subheader("People like you also like")
    st.caption("Readers whose taste (by learned embedding) resembles yours — not items close to your own embedding, but what *they* actually liked.")
    similar_users = cached["similar_users_liked"]
    if "error" in similar_users:
        # A brand-new user (or one with very little history yet) genuinely
        # has no similar-taste readers to compare against — an expected,
        # temporary state, not a system error, so no red error box here.
        st.caption("Not enough activity yet to find readers with similar taste — check back after browsing or liking a few more books.")
    else:
        render_item_row(similar_users["items"], similar_users["source_model"])

st.divider()
st.subheader("Recently added")
st.caption("A random pick from the 100 most recently added books — changes on every reload.")
recent = cached["recent"]
if "error" in recent:
    show_api_error(recent)
else:
    render_item_row(recent["items"], recent["source_model"])
