import streamlit as st

from services.api_client import check_health, list_users

st.set_page_config(
    page_title="Amazon Books Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None  # None = "not logged in" -> Popularity homepage


# --- Sidebar: mock login + live health check ---
# Not a real auth system — a user_id picker over the demo-able users seeded
# by prestart.py, simulating a logged-in session per project.md's
# "Homepage (logged in)" requirement.
with st.sidebar:
    st.markdown("### Account")
    users_result = list_users(limit=300)
    if "error" in users_result:
        st.error("Could not load demo users — is the API running?")
        demo_users = []
    else:
        demo_users = users_result.get("users", [])

    options = ["Guest (not logged in)"] + demo_users
    current = st.session_state["user_id"] or "Guest (not logged in)"
    choice = st.selectbox("Log in as", options, index=options.index(current) if current in options else 0)
    st.session_state["user_id"] = None if choice == "Guest (not logged in)" else choice

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


def show_main_content():
    st.title("📚 Amazon Books Recommender")
    st.markdown("""
    A real-time recommendation system for Amazon Books, built on MF-BPR,
    two-tower retrieval, and a LightGBM ranker — all trained from scratch.

    - **Homepage** — trending books (logged out) or a personalized feed
      re-ranked by LightGBM (logged in), via the sidebar.
    - **Item pages** — "similar items" from learned embeddings.

    Log in as a demo user in the sidebar to see personalization in action —
    each recommendation shows which model produced it.
    """)
    st.info("Select a page from the navigation menu on the left.")


main_page = st.Page(show_main_content, title="Main", default=True)
homepage = st.Page("pages/1_Homepage.py", title="Homepage")
item_page = st.Page("pages/2_Item.py", title="Item Detail")

pg = st.navigation([main_page, homepage, item_page])
pg.run()
