import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(page_title="LangChain Multi-Tool Chat", page_icon="🤖", layout="centered")

# ------------------ STEP 1: API KEY INPUT ------------------
if "api_key" not in st.session_state:
    st.title("🔑 Enter your Groq API Key")
    api_key_input = st.text_input("Paste your Groq API key:", type="password")

    if st.button("Submit"):
        if api_key_input.strip():
            st.session_state.api_key = api_key_input.strip()
            st.rerun()
        else:
            st.error("⚠️ Please enter a valid API key.")

# ------------------ STEP 2: MAIN CHAT INTERFACE ------------------
else:
    # Define tools
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

    api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

    duckDuckGoSearch = DuckDuckGoSearchRun(name="duckDuckGoSearch")
    tools = [arxiv, wiki, duckDuckGoSearch]

    # LLM with user-provided API key
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b", streaming=True, api_key=st.session_state.api_key)

    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True
    )

    st.title("🤖 LangChain Agent Chatbot with Tools")
    st.markdown("Ask me anything, and I'll use **Wikipedia**, **Arxiv**, and **DuckDuckGoSearch**")

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt_input := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            container = st.container()
            with st.spinner("Thinking..."):
                callback = StreamlitCallbackHandler(container)
                try:
                    response = agent_executor.invoke(
                        {
                            "input": prompt_input,
                            "chat_history": st.session_state.history,
                        },
                        {"callbacks": [callback]}
                    )
                    answer = response.get("output", "Sorry, I couldn't find an answer.")
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
                    answer = "Sorry, something went wrong."

            container.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.history.append(HumanMessage(content=prompt_input))
        st.session_state.history.append(AIMessage(content=answer))
