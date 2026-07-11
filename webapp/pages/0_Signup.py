import streamlit as st

from services import api_client
from services.ui import clamped_title_html, cover_button, ensure_session, show_api_error

st.title("Sign Up")

ensure_session()

if st.session_state.get("user_id"):
    st.info(f"You're already logged in as **{st.session_state['user_id']}**. Choose \"Guest\" in the sidebar to create a different account.")
    st.stop()

if "signup_step" not in st.session_state:
    st.session_state["signup_step"] = 1


def _reset_signup():
    for key in ("signup_step", "signup_new_user_id", "signup_categories", "signup_suggestions", "signup_selected"):
        st.session_state.pop(key, None)


# --- Step 1: pick a username + at least 2 categories ---
# Nothing is persisted here — no account, no preferences — so an abandoned
# signup never leaves a half-created "ghost" user behind. Everything is only
# written to the database on the Finish step below.
if st.session_state["signup_step"] == 1:
    st.subheader("Step 1 of 3 — Tell us a bit about you")
    new_user_id = st.text_input("Choose a username", value=st.session_state.get("signup_new_user_id", ""))

    categories_result = api_client.list_categories()
    if "error" in categories_result:
        show_api_error(categories_result)
        st.stop()

    st.caption("Pick at least 2 categories you enjoy — click to select, click again to deselect.")
    categories = st.pills(
        "Categories", options=categories_result["categories"], default=st.session_state.get("signup_categories", []),
        selection_mode="multi", label_visibility="collapsed",
    )

    if st.button("Continue", type="primary"):
        if not new_user_id.strip():
            st.warning("Please choose a username.")
        elif len(categories) < 2:
            st.warning("Please select at least 2 categories.")
        else:
            st.session_state["signup_new_user_id"] = new_user_id.strip()
            st.session_state["signup_categories"] = categories
            st.session_state.pop("signup_suggestions", None)  # categories may have changed
            st.session_state["signup_step"] = 2
            st.rerun()


# --- Step 2: suggest books from those categories, ask which they like ---
elif st.session_state["signup_step"] == 2:
    new_user_id = st.session_state["signup_new_user_id"]
    categories = st.session_state["signup_categories"]
    st.subheader("Step 2 of 3 — Which of these do you like?")
    st.caption("Suggested from: " + ", ".join(categories) + ". Click a cover to select it, click again to deselect.")

    # Fetched once and cached — the underlying recommendation is temperature-
    # sampled (randomized), so re-fetching on every rerun (e.g. every time a
    # cover is clicked) would reshuffle the grid out from under the user. No
    # account exists yet, so this asks by category directly rather than by user_id.
    if "signup_suggestions" not in st.session_state:
        suggestions = api_client.get_onboarding_suggestions_by_categories(categories, k=12)
        if "error" in suggestions:
            show_api_error(suggestions)
            st.stop()
        st.session_state["signup_suggestions"] = suggestions["items"]
    st.session_state.setdefault("signup_selected", set())

    items = st.session_state["signup_suggestions"]
    selected = st.session_state["signup_selected"]

    cols = st.columns(4)
    for i, rec in enumerate(items):
        item = rec["item"]
        asin = item["parent_asin"]
        with cols[i % 4]:
            is_selected = asin in selected
            if cover_button(item.get("image_url"), key=f"onboard_toggle_{asin}", selected=is_selected):
                if is_selected:
                    selected.discard(asin)
                else:
                    selected.add(asin)
                st.rerun()
            title = (item.get("title") or asin)[:50]
            st.markdown(clamped_title_html(title), unsafe_allow_html=True)

    st.divider()
    if st.button("← Back", key="signup_step2_back"):
        st.session_state["signup_step"] = 1
        st.rerun()

    if st.button("Finish", type="primary"):
        result = api_client.signup(new_user_id)
        if "error" in result:
            # Username was taken (or another signup error) — send back to
            # step 1 to fix it, keeping everything else (categories,
            # suggestions, selections) intact so nothing has to be redone.
            show_api_error(result)
            st.session_state["signup_step"] = 1
            st.rerun()
        else:
            prefs_result = api_client.set_category_preferences(new_user_id, categories)
            if "error" in prefs_result:
                show_api_error(prefs_result)
                st.stop()
            # Logged directly against the new user_id (not yet the active
            # session user) so `onboarding_like` can backfill an Interaction
            # for them immediately — see POST /events. Logged once here, from
            # the final selection state, rather than per-click, so toggling
            # back and forth doesn't spam duplicate events.
            for rec in items:
                asin = rec["item"]["parent_asin"]
                event_type = "onboarding_like" if asin in selected else "onboarding_skip"
                api_client.log_event(
                    session_id=st.session_state["session_id"], event_type=event_type,
                    user_id=new_user_id, parent_asin=asin,
                )
            st.session_state["signup_step"] = 3
            st.rerun()


# --- Step 3: done ---
elif st.session_state["signup_step"] == 3:
    st.subheader("Step 3 of 3 — All set!")
    st.success(f"Welcome, **{st.session_state['signup_new_user_id']}**!")
    if st.button("Go to my homepage", type="primary"):
        st.session_state["user_id"] = st.session_state["signup_new_user_id"]
        _reset_signup()
        st.switch_page("pages/1_Homepage.py")
