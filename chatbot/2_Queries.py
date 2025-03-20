import streamlit as st
import json
import spacy

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load museum data from JSON file
with open("museum_db.json", "r") as file:
    museum_data = json.load(file)

# Function to fetch structured museum details
def fetch_museum_info(query):
    doc = nlp(query.lower())

    if "museum name" in query or "name of the museum" in query:
        return f"The museum's name is {museum_data['museum_info']['name']}."

    if "location" in query or "where is the museum" in query:
        return f"The museum is located at {museum_data['museum_info']['location']}."

    if "contact" in query:
        return f"You can contact the museum at {museum_data['museum_info']['contact']}."

    if "website" in query:
        return f"Visit the museum's website: {museum_data['museum_info']['website']}"

    if "history" in query or "about the museum" in query:
        return f"Museum History: {museum_data['museum_info']['history']}"

    if "facilities" in query or "amenities" in query:
        facilities = museum_data['museum_info']['facilities']
        return (f"Museum Facilities:\n"
                f"event_space: {facilities['event_space']}\n"
                f"stem_Shop: {facilities['stem_shop']}\n"
                f"Wheelchair Access: {facilities['wheelchair_access']}\n"
                f"Parking: {facilities['parking']}")

    if "special collections" in query or "permanent exhibits" in query:
        collections = museum_data['museum_info']['special_collections']
        return "Special Collections:\n" + "\n".join([f"{col['name']}: {col['description']} (Located in {col['location']})" for col in collections])

    if "guided tours" in query or "tours" in query:
        tours = museum_data['museum_info']['guided_tours']
        return (f"Guided Tours:\n"
                f"Availability: {tours['availability']}\n"
                f"Price: {tours['price']}\n"
                f"Duration: {tours['duration']}")

    if "membership" in query or "membership benefits" in query:
        membership = museum_data['museum_info']['membership']
        benefits = "\n".join(membership['benefits'])
        return (f"Membership Options:\n"
                f"Individual: {membership['individual']}\n"
                f"Family: {membership['family']}\n"
                f"Benefits:\n{benefits}")

    if "accessibility" in query or "disabled access" in query:
        accessibility = museum_data['museum_info']['accessibility']
        return (f"Accessibility Information:\n"
                f"Wheelchair Rental: {accessibility['wheelchair_rental']}\n"
                f"Sign Language Tours: {accessibility['sign_language_tours']}\n"
                f"Elevators: {accessibility['elevators']}")

    if "open" in query or "timing" in query:
        for token in doc:
            if token.text.capitalize() in museum_data["opening_hours"]:
                return f"Opening hours on {token.text.capitalize()}: {museum_data['opening_hours'][token.text.capitalize()]}"
        return "Please specify a day to check opening hours."

    if "ticket" in query or "price" in query or "cost" in query:
        for token in doc:
            if token.text.lower() in museum_data["ticket_prices"]:
                return f"The price for {token.text.capitalize()} is {museum_data['ticket_prices'][token.text.lower()]}"
        return "Please specify a category (Adult, Child, Student, Senior)."

    if "exhibition" in query or "event" in query:
        exhibitions = museum_data["exhibitions"]
        if exhibitions:
            return "Upcoming Exhibitions:\n" + "\n".join([f"{ex['name']} ({ex['date']}) - {ex['location']}" for ex in exhibitions])
        return "No exhibitions are currently available."

    return "I'm not sure about that. Please ask something else about the museum!"

# Streamlit App
def main():
    # Set custom CSS for black text
    st.markdown(
        """
        <style>
        h1 {
            color: lightyellow !important;
        }
        .welcome-message {
            color: lightyellow;
            font-size: 150px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title in black color
    st.title("🤖 Museum Chatbot")

    # Welcome message in black color
    st.markdown('<p class="welcome-message">Welcome to the Museum Chatbot! Ask me anything about the museum.</p>', unsafe_allow_html=True)

    # Input box for user query
    user_query = st.text_input("You:", placeholder="Type your question here...")

    # Submit button
    if st.button("Ask"):
        if user_query.strip() == "":
            st.warning("Please type something!")
        else:
            response = fetch_museum_info(user_query)
            # Increase font size for the response
            st.markdown(f"<p style='font-size: 20px;'><b>Bot:</b> {response}</p>", unsafe_allow_html=True)

    # Sidebar for additional information
    st.sidebar.header("About the Museum")
    st.sidebar.write(f"**Name:** {museum_data['museum_info']['name']}")
    st.sidebar.write(f"**Location:** {museum_data['museum_info']['location']}")
    st.sidebar.write(f"**Contact:** {museum_data['museum_info']['contact']}")
    st.sidebar.write(f"**Website:** {museum_data['museum_info']['website']}")

# Run the Streamlit app
if __name__ == "__main__":
    main()