import streamlit as st
import requests

def send_to_server(prompt: str, history: dict):
    url = "http://backend:8002/chat/chat/"
    payload = {
        "text": prompt, 
        'history': history
    }

    print(payload)

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def process_response(prompt, history):
    result = send_to_server(prompt, history)
    result = result['ai_response']

    return result

def main():
    st.write("На этой странице описаны найденные закономерности в данных")

    st.title("Echo Bot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = {}

    # Display chat messages from history on app rerun
    for index, message in st.session_state.messages.items():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages[len(st.session_state.messages)] = {"role": "user", "content": prompt}

        response = process_response(prompt, st.session_state.messages)
        print('====')
        print(st.session_state.messages)

        if response:
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages[len(st.session_state.messages)] = {"role": "assistant", "content": response}

if __name__ == "__main__":
    main()
