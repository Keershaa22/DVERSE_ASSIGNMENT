import streamlit as st
import json

# Load museum data from JSON file
with open("museum_db.json", "r") as file:
    museum_data = json.load(file)

# Set page configuration
st.set_page_config(
    page_title="GD Car Museum",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add title at the top
st.title("GD CAR MUSEUM")

# Add image (full width)
st.image("gd_car_museum.jpg", use_container_width=True)

# Add description
st.write("""
**Gedee Car Museum** is the only classic car museum of its kind in South India, located in Coimbatore, Tamil Nadu. 
It has an impressive collection of more than 100 vintage cars, each with a history or a unique technology.

The cars are a private collection of **G D Naidu Charities**, a social trust founded by (late) Sri. G D Naidu. 
Like his father, Sri. G D Gopal, who is also an avid auto enthusiast, purchased and collected several cars, 
especially those that had unique mechanical features or the ones that had significantly influenced the evolution of the automobile.
""")

# Add a button to navigate to the chatbot page
if st.button("Go to Chatbot"):
    st.switch_page("pages/2_Queries.py")

# Sidebar for additional information
st.sidebar.header("About the Museum")
st.sidebar.write(f"**Name:** {museum_data['museum_info']['name']}")
st.sidebar.write(f"**Location:** {museum_data['museum_info']['location']}")
st.sidebar.write(f"**Contact:** {museum_data['museum_info']['contact']}")
st.sidebar.write(f"**Website:** {museum_data['museum_info']['website']}")