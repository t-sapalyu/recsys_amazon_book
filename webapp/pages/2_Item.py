import streamlit as st

from services.api_client import add_to_cart, add_to_wishlist, get_item, get_same_author, get_similar_authors, get_similar_items, submit_rating
from services.ui import PLACEHOLDER_IMAGE, clear_pending_view, render_item_row, show_api_error, sync_pending_view

st.title("Item Detail")

item_id = st.query_params.get("item_id")

if not item_id:
    st.info("Pick a book from the Homepage to see its details here.")
    st.stop()

# Fetched once per item_id and cached — otherwise EVERY rerun of this page
# (including the one triggered just by clicking "Add to wishlist"/"Add to
# cart"/submitting a rating, or by a "Similar items" card's "View details"
# button) re-ran every one of these model-backed calls (FAISS similarity
# search, author-centroid search) from scratch. Against the real models
# that's slow enough that a click could get superseded by a later
# interaction before Streamlit ever finished processing it — see
# 1_Homepage.py's cache for the full explanation of that race.
cache_key = f"item_cache_{item_id}"
if cache_key not in st.session_state:
    cached = {"item": get_item(item_id)}
    if "error" not in cached["item"]:
        cached["similar_items"] = get_similar_items(item_id, k=10)
        if cached["item"].get("author"):
            cached["same_author"] = get_same_author(item_id, k=10)
            cached["similar_authors"] = get_similar_authors(item_id, k=10)
    st.session_state[cache_key] = cached

cached = st.session_state[cache_key]
item = cached["item"]
if "error" in item:
    show_api_error(item)
    st.stop()

sync_pending_view(item_id)
user_id = st.session_state.get("user_id")

col_img, col_info = st.columns([1, 3])
with col_img:
    st.image(item.get("image_url") or PLACEHOLDER_IMAGE, width="stretch")
with col_info:
    st.header(item.get("title") or item["parent_asin"])
    if item.get("author"):
        st.caption(f"by {item['author']}")
    if item.get("category"):
        st.caption(item["category"])
    if item.get("average_rating"):
        st.write(f"⭐ {item['average_rating']:.1f}")
    if item.get("price"):
        st.write(f"${item['price']:.2f}")

    if user_id:
        b1, b2 = st.columns(2)
        if b1.button("🤍 Add to wishlist"):
            add_to_wishlist(user_id, item_id, st.session_state["session_id"])
            clear_pending_view()
            st.toast("Added to wishlist")
        if b2.button("🛒 Add to cart"):
            add_to_cart(user_id, item_id, st.session_state["session_id"])
            clear_pending_view()
            st.toast("Added to cart")

        with st.form(key=f"rating_form_{item_id}"):
            rating = st.slider("Rate this book", min_value=1, max_value=5, value=4)
            if st.form_submit_button("Submit rating"):
                submit_rating(user_id, item_id, float(rating), st.session_state["session_id"])
                clear_pending_view()
                st.toast("Thanks for rating!")
    else:
        st.caption("Log in or sign up from the sidebar to save this book or rate it.")

st.divider()
st.subheader("Similar items")
sims = cached["similar_items"]
if "error" in sims:
    show_api_error(sims)
else:
    render_item_row(sims["items"], sims["source_model"])

# Author-based sections only apply to items with a listed author (~95% of the
# catalog) — skip silently rather than showing an error for the ~5% without one.
if item.get("author"):
    st.divider()
    st.subheader(f"More by {item['author']}")
    same_author = cached["same_author"]
    if "error" in same_author:
        show_api_error(same_author)
    elif same_author["items"]:
        render_item_row(same_author["items"], same_author["source_model"])
    else:
        st.caption("No other books by this author in the catalog.")

    st.divider()
    st.subheader("Readers who liked this also read")
    similar_authors = cached["similar_authors"]
    if "error" in similar_authors:
        show_api_error(similar_authors)
    else:
        st.caption("By authors with a similar overall body of work: " + ", ".join(similar_authors["similar_authors"]))
        render_item_row(similar_authors["items"], similar_authors["source_model"])
