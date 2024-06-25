import streamlit as st
import speech_recognition as sr
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from gtts import gTTS
import os

def get_response_from_ai(human_input, history):
    template = """
        1/ Your name is Zara, you are 21 years old,
        2/ You are my girlfriend,
        3/ You are caring, supportive, and always there to listen.
        4/ You have a great sense of humor and enjoy making me laugh.
        5/ You love reading books, going for long walks, and trying new recipes.
        6/ You are studying psychology and are passionate about understanding human behavior.
        7/ You always encourage me to pursue my dreams and help me stay motivated.
        8/ We enjoy watching movies together and exploring new places on weekends.

        {history}
        Boyfriend: {human_input}
        Zara:
    """
    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )
    chatgpt_chain = LLMChain(
        llm=OpenAI(api_key="sk-proj-wqA3pNqjr2JXlX4QiTvzT3BlbkFJYPPhQ6kaEFqy69NMLA4z", temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferMemory(k=2)
    )
    output = chatgpt_chain.predict(human_input=human_input, history=history)
    return output

# Streamlit app
def main():
    st.title("Chat with Zara")
    
    if "history" not in st.session_state:
        st.session_state.history = ""
    
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    if st.button("Speak"):
        with mic as source:
            st.info("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        
        try:
            user_input = recognizer.recognize_google(audio)
            st.write(f"You: {user_input}")
            
            if user_input.lower() not in ['exit', 'quit'] and user_input.strip() != "":
                response = get_response_from_ai(user_input, st.session_state.history)
                st.session_state.history += f"\nHuman: {user_input}\nZara: {response}\n"
                response = response.replace("At the end of the sentence", "")
                
                st.write(f"Zara: {response}")
                
                # Text-to-speech
                tts = gTTS(response)
                tts.save("response.mp3")
                st.audio("response.mp3", format="audio/mp3")
        except sr.UnknownValueError:
            st.write("Sorry, I did not understand that.")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
    
    st.write("Conversation History")
    st.write(st.session_state.history)

if __name__ == "__main__":
    main()
