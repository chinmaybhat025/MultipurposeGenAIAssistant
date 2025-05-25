import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Set up the Streamlit app
st.set_page_config(page_title="Multi Purpose GenAI Assistant", page_icon="🧮")
st.title("Multi Purpose GenAI Assistant")

# Get the API key from sidebar
groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

# Initialize the LLM
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Wikipedia Tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Searches Wikipedia for general information about a given topic."
)

# Custom Math Tool using LLMChain instead of LLMMathChain
math_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful and accurate math expert. Carefully solve the following math problem step-by-step and return the final answer at the end.

Question: {question}

Answer:
"""
)
math_chain = LLMChain(llm=llm, prompt=math_prompt)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Solves math questions step-by-step and provides the final result."
)

# Reasoning Tool
reasoning_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an agent tasked with solving users' logical or reasoning-based questions. Logically arrive at the solution and provide a detailed explanation, step-by-step and point-wise.

Question: {question}

Answer:
"""
)
reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt)
reasoning_tool = Tool(
    name="Reasoning tool",
    func=reasoning_chain.run,
    description="Handles logic-based and reasoning questions with clear explanation."
)

# Initialize the agent with all tools
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a Multi-Purpose chatbot who can answer all your questions."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input question
question = st.text_area("Enter your question:", "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

# Button to get response
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            except Exception as e:
                response = f"An error occurred while processing your question: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("### Response:")
            st.success(response)
    else:
        st.warning("Please enter a question to continue.")
