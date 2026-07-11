import streamlit as st

from services import ui
from services.api_client import check_health, list_users

st.set_page_config(
    page_title="Amazon Books Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None  # None = "not logged in" -> Popularity homepage

ui.ensure_session()


def show_main_content():
    ui.sync_pending_view()
    st.title("📚 Amazon Books Recommender")
    st.markdown("""
    A real-time recommendation system for Amazon Books, built on MF-BPR,
    two-tower retrieval, and a LightGBM ranker — all trained from scratch.

    - **Homepage** — trending + discovery picks (logged out) or a personalized
      feed re-ranked by LightGBM (logged in), via the sidebar.
    - **Item pages** — "similar items" from learned embeddings.

    Log in as a demo user in the sidebar, or sign up as a new one, to see
    personalization in action — each recommendation shows which model
    produced it.
    """)
    st.info("Select a page from the navigation menu on the left.")


# `st.navigation` must be constructed before any `st.page_link` call below can
# resolve those same pages — page_link looks up the target in the navigation
# registry, which doesn't exist until this runs.
main_page = st.Page(show_main_content, title="Main", default=True)
homepage = st.Page("pages/1_Homepage.py", title="Homepage")
item_page = st.Page("pages/2_Item.py", title="Item Detail")
signup_page = st.Page("pages/0_Signup.py", title="Sign Up")
account_page = st.Page("pages/3_My_Account.py", title="My Account")
activity_log_page = st.Page("pages/4_Activity_Log.py", title="Activity Log")
signal_rules_page = st.Page("pages/5_Signal_Rules.py", title="Signal Rules")

pg = st.navigation([main_page, homepage, item_page, signup_page, account_page, activity_log_page, signal_rules_page])


# --- Sidebar: mock login + live health check ---
# Not a real auth system — a user_id picker over the demo-able users seeded
# by prestart.py, simulating a logged-in session per project.md's
# "Homepage (logged in)" requirement. New users created via the Sign Up page
# join this same picker once created.
with st.sidebar:
    st.markdown("### Account")
    # Ordered most-recent-first server-side, but still capped — the 300
    # pre-seeded demo users alone would already fill a smaller limit and
    # push new signups off the list, so this asks for the API's max.
    users_result = list_users(limit=1000)
    if "error" in users_result:
        st.error("Could not load demo users — is the API running?")
        demo_users = []
    else:
        demo_users = users_result.get("users", [])

    options = ["Guest (not logged in)"] + demo_users
    current = st.session_state["user_id"] or "Guest (not logged in)"
    choice = st.selectbox("Log in as", options, index=options.index(current) if current in options else 0)
    st.session_state["user_id"] = None if choice == "Guest (not logged in)" else choice

    if st.session_state["user_id"] is None:
        st.page_link(signup_page, label="➕ Sign up", icon="➕")
    else:
        st.page_link(account_page, label="My wishlist & cart", icon="🛍️")
    st.page_link(activity_log_page, label="Activity log (recorded signals)", icon="📋")
    st.page_link(signal_rules_page, label="Signal rules (the full rulebook)", icon="📜")

    @st.fragment(run_every="20s")
    def live_health_check():
        st.markdown("---")
        st.markdown("### System status")
        health = check_health()
        if health["status"] == "online":
            st.success("🟢 API: Online")
        else:
            st.error(f"🔴 API: {health['status']}")
        st.caption("Updates automatically every 20s.")

    live_health_check()


pg.run()
