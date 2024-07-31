import streamlit as st
from firebase_admin import firestore
import datetime
import matplotlib.pyplot as plt
import pandas as pd

def get_classification_text(score):
    classifications = {
        0: "Clear",
        0.5: "Clear",
        1: "Mild",
        1.5: "Mild",
        2: "Moderate",
        2.5: "Moderate",
        3: "Severe"
    }
    return classifications.get(score, "Unknown")

def fetch_journal_entries(db):
    user_id = "4WWZhb92MKMMjciIIYlr92cr1UM2"
    entries_ref = db.collection("users").document(user_id).collection("entries")
    docs = entries_ref.order_by("date", direction=firestore.Query.DESCENDING).get()
    
    entries = []
    for doc in docs:
        entry_data = doc.to_dict()
        entry_data['id'] = doc.id  # Add document ID to the entry data
        entries.append(entry_data)
    
    return entries

def calculate_delta(current_score, previous_score):
    if current_score is None or previous_score is None:
        return None
    return current_score - previous_score

def format_count_and_delta(count, delta):
    if count is None:
        return "N/A"
    count_str = f"{count} blemishes"
    if delta is not None:
        delta_str = f" ({delta:+.1f})" if delta != 0 else " (no change)"
        return f"{count_str}{delta_str}"
    return count_str

def display_journal_entry(entry, previous_entry):
    st.subheader(f"Entry on {entry['date'].strftime('%B %d, %Y')}")
    
    if 'body' in entry:
        for item in entry['body']:
            st.write(item)
    
    if 'media' in entry and entry['media']:
        columns = st.columns(len(entry['media']))
        for i, media_url in enumerate(entry['media']):
            with columns[i]:
                st.image(media_url, use_column_width=True)
    
    if 'scores' in entry:
        scores = entry['scores']
        prev_scores = previous_entry.get('scores', {}) if previous_entry else {}
        
        st.write("Acne Assessment:")
        
        left_class = get_classification_text(scores.get('leftClassification'))
        right_class = get_classification_text(scores.get('rightClassification'))
        total_class = get_classification_text(scores.get('totalClassification'))
        
        left_count = scores.get('leftPimpleCount')
        right_count = scores.get('rightPimpleCount')
        total_count = scores.get('totalPimpleCount')
        
        left_delta = calculate_delta(left_count, prev_scores.get('leftPimpleCount'))
        right_delta = calculate_delta(right_count, prev_scores.get('rightPimpleCount'))
        total_delta = calculate_delta(total_count, prev_scores.get('totalPimpleCount'))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Left Side",
                value=left_class,
                delta=format_count_and_delta(left_count, left_delta),
                delta_color="normal" if left_delta <= 0 else "inverse"
            )
        with col2:
            st.metric(
                label="Right Side",
                value=right_class,
                delta=format_count_and_delta(right_count, right_delta),
                delta_color="normal" if right_delta <= 0 else "inverse"
            )
        with col3:
            st.metric(
                label="Overall",
                value=total_class,
                delta=format_count_and_delta(total_count, total_delta),
                delta_color="normal" if total_delta <= 0 else "inverse"
            )
    
    st.markdown("---")

def show_total_score_chart(entries):
    # Extract dates and total pimple counts, handling missing data
    dates = []
    total_counts = []
    for entry in entries:
        date = entry.get('date')
        scores = entry.get('scores', {})
        total_count = scores.get('totalPimpleCount', 0)  # Default to 0 if missing
        if date:
            dates.append(date)
            total_counts.append(total_count)

    # Create a dataframe for plotting
    data = pd.DataFrame({'Date': dates, 'Total Pimple Count': total_counts})
    data['Date'] = pd.to_datetime(data['Date'])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Total Pimple Count'], marker='o', linestyle='-')
    plt.title('Total Blemish Count Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Blemish Count')
    plt.grid(True)
    st.pyplot(plt)



def show_andreas_journey(db):
    st.title("Andrea's Journal Entries")
    st.write("Below are my daily journal entries, tracking my skincare journey:")

    entries = fetch_journal_entries(db)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min(entry['date'].date() for entry in entries))
    with col2:
        end_date = st.date_input("End Date", max(entry['date'].date() for entry in entries))

    filtered_entries = [
        entry for entry in entries 
        if start_date <= entry['date'].date() <= end_date
    ]

    show_total_score_chart(filtered_entries)

    for i, entry in enumerate(filtered_entries):
        previous_entry = filtered_entries[i+1] if i+1 < len(filtered_entries) else None
        display_journal_entry(entry, previous_entry)
