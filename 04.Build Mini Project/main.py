import streamlit as st
import json
from PIL import Image
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import HumanMessage, AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables.graph import MermaidDrawMethod
import tempfile
import os
import streamlit.components.v1 as components

# --- Define the state for LangGraph ---
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# --- Initialize Gemini LLM once ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# --- Robust JSON parser for Gemini output ---
def try_extract_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"error": "Invalid JSON from Gemini."}

# --- Gemini LLM Schema Generator Node ---
def generate_schema_node(state: State) -> State:
    user_input = state['messages'][-1].content if state['messages'] else ""

    prompt = f"""
            You are a form generator. Given a user's description, output a JSON schema.
            Respond ONLY with a JSON object with: title, fields (name, type, label, required, etc.)
            User Description: "{user_input}"
            Respond with JSON only
    """

    response = llm.invoke(prompt)
    parsed = try_extract_json(response.content)

    if "error" in parsed:
        return {"messages": [HumanMessage(content="Invalid JSON from Gemini.")]}
    else:
        return {"messages": [HumanMessage(content=json.dumps(parsed))]}

# --- Build LangGraph ---
graph_builder = StateGraph(State)
graph_builder.add_node("generate_schema", generate_schema_node)
graph_builder.set_entry_point("generate_schema")
graph_builder.set_finish_point("generate_schema")
graph = graph_builder.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="Gemini JSON Forms Builder", layout="wide")
st.title("🧠 JSON Forms Schema Generator (Gemini + LangGraph)")

# --- Optional Diagram ---
st.subheader("🧭 LangGraph Workflow Diagram")
image = None
# Save LangGraph diagram as PNG locally using Pyppeteer
try:
    image = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,  # Correct method for Streamlit
        max_retries=5,
        retry_delay=2.0
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        st.image(tmp.name, caption="LangGraph Workflow Diagram", use_column_width=True)
except Exception as e:
    pass

# --- LangGraph Visualization ---
st.image(image, caption="LangGraph Workflow Diagram", use_column_width=False)

# --- User Input ---
st.subheader("📝 Describe your form:")
form_description = st.text_area(
    "What fields should the form contain?",
    height=150,
    placeholder="e.g., A form to collect full name, age, and newsletter subscription checkbox."
)

if st.button("Generate Schema"):
    if not form_description.strip():
        st.warning("Please enter a form description.")
    else:
        with st.spinner("Generating schema with Gemini..."):
            final_state = graph.invoke({"messages": [HumanMessage(content=form_description)]})
            result_str = final_state['messages'][-1].content

            try:
                result = json.loads(result_str)
            except json.JSONDecodeError:
                st.error("Gemini returned an unreadable JSON response.")
                st.code(result_str)
                st.stop()

            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader("📦 JSON Schema")
                st.json(result)
                st.download_button("Download JSON Schema", json.dumps(result, indent=2), "json_schema.json")

                # ---- Live Preview of JSON Form ----
                st.subheader("🧪 Live JSON Form Preview")
                json_schema_str = json.dumps(result.get("json_schema", {}))
                ui_schema_str = json.dumps(result.get("ui_schema", {}))

                html_template = f"""
                <!DOCTYPE html>
                <html>
                  <head>
                    <meta charset="UTF-8" />
                    <title>JSON Form Preview</title>
                    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
                    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
                    <script src="https://unpkg.com/@jsonforms/core/bundles/jsonforms-core.umd.js"></script>
                    <script src="https://unpkg.com/@jsonforms/react/bundles/jsonforms-react.umd.js"></script>
                    <script src="https://unpkg.com/@jsonforms/react/vanilla/bundles/jsonforms-react-vanilla.umd.js"></script>
                    <style>
                      body {{ font-family: sans-serif; margin: 10px; }}
                    </style>
                  </head>
                  <body>
                    <div id="form"></div>
                    <script type="text/javascript">
                      const {{ JsonForms }} = window.JSONFormsReactVanilla;
                      const schema = {json_schema_str};
                      const uischema = {ui_schema_str};
                      const render = () => {{
                        ReactDOM.render(
                          React.createElement(JsonForms, {{
                            schema,
                            uischema,
                            data: {{}},
                            renderers: window.JSONFormsReactVanilla.vanillaRenderers
                          }}),
                          document.getElementById('form')
                        );
                      }};
                      render();
                    </script>
                  </body>
                </html>
                """

                components.html(html_template, height=500, scrolling=True)
