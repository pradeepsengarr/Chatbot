import streamlit as st
from streamlit_chat import message
import logging
from text_app import process_answer  # Import the process_answer function

# Configure logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(layout="wide")  # Set Streamlit layout to wide

def display_conversation(history):
    """Display the conversation history in the chat interface."""
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

def main():
    """Main function to run the Streamlit app."""
    logging.info("Starting the chatbot UI")
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with Your Buddy</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>This is a Prototype</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:red;'>Ask your questions about audits 👇</h2>", unsafe_allow_html=True)

    # Initialize session state for conversation history
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hi PRADA!! How Can I Help You Today :-)"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    # User input
    user_input = st.text_input("", key="input")

    # Process user input and update session state
    if user_input:
        answer = process_answer(user_input)  # Get answer from the model
        st.session_state["past"].append(user_input)  # Save user query
        response = answer
        st.session_state["generated"].append(response)  # Save model response

    # Display the conversation history
    display_conversation({"past": st.session_state["past"], "generated": st.session_state["generated"]})

if __name__ == "__main__":
    main()
