import streamlit as st

from services.api_client import get_item, get_same_author, get_similar_authors, get_similar_items
from services.ui import PLACEHOLDER_IMAGE, render_item_row, show_api_error

st.title("Item Detail")

item_id = st.query_params.get("item_id")

if not item_id:
    st.info("Pick a book from the Homepage to see its details here.")
    st.stop()

item = get_item(item_id)
if "error" in item:
    show_api_error(item)
    st.stop()

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

st.divider()
st.subheader("Similar items")
sims = get_similar_items(item_id, k=10)
if "error" in sims:
    show_api_error(sims)
else:
    render_item_row(sims["items"], sims["source_model"])

# Author-based sections only apply to items with a listed author (~95% of the
# catalog) — skip silently rather than showing an error for the ~5% without one.
if item.get("author"):
    st.divider()
    st.subheader(f"More by {item['author']}")
    same_author = get_same_author(item_id, k=10)
    if "error" in same_author:
        show_api_error(same_author)
    elif same_author["items"]:
        render_item_row(same_author["items"], same_author["source_model"])
    else:
        st.caption("No other books by this author in the catalog.")

    st.divider()
    st.subheader("Readers who liked this also read")
    similar_authors = get_similar_authors(item_id, k=10)
    if "error" in similar_authors:
        show_api_error(similar_authors)
    else:
        st.caption("By authors with a similar overall body of work: " + ", ".join(similar_authors["similar_authors"]))
        render_item_row(similar_authors["items"], similar_authors["source_model"])
