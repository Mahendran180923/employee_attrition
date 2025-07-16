import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, r2_score, root_mean_squared_error
import streamlit as st
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import time

# st.button("Click me")


# st.page_link("app.py", label="Home")

st.checkbox("I agree")
st.feedback("thumbs")
st.pills("Tags", ["Sports", "Politics"])
st.radio("Pick one", ["cats", "dogs"])
st.segmented_control("Filter", ["Open", "Closed"])
st.toggle("Enable")
st.selectbox("Pick one", ["cats", "dogs"])
st.multiselect("Buy", ["milk", "apples", "potatoes"])
st.slider("Pick a number", 0, 100)
st.select_slider("Pick a size", ["S", "M", "L"])
st.text_input("First name")
st.number_input("Pick a number", 0, 10)
st.text_area("Text to translate")
st.date_input("Your birthday")
st.time_input("Meeting time")
st.file_uploader("Upload a CSV")
st.audio_input("Record a voice message")

st.color_picker("Pick a color")


st.write("Most objects") # df, err, func, keras!
st.write(["st", "is <", 3])
# st.write_stream(my_generator)
# st.write_stream(my_llm_stream)

st.text("Fixed width text")
st.markdown("_Markdown_")
st.latex(r""" e^{i\pi} + 1 = 0 """)
st.title("My title")
st.header("My header")
st.subheader("My sub")


st.html("<p>Hi!</p>")

# data = [1, 2, 3, 4,5,6,7,8,9,10]
my_dataframe = [1,2,3,4,5]

st.dataframe(my_dataframe)
# st.table(data.iloc[0:10])
st.json({"foo":"bar","fu":"ba"})
st.metric("My metric", 42, 2)



# # Insert a chat message container.
# with st.chat_message("user"):
#     st.write("Hello 👋")
#     st.line_chart(np.random.randn(30, 3))

# # Display a chat input widget at the bottom of the app.
# st.chat_input("Say something")

# # Display a chat input widget inline.
# with st.container():
#     st.chat_input("Say something")



expand = st.expander("My label", icon=":material/info:")
expand.write("Inside the expander.")
pop = st.popover("Button label")
pop.checkbox("Show all")

# You can also use "with" notation:
with expand:
    st.radio("Select one:", [1, 2])


# Show a spinner during a process
with st.spinner(text="In progress"):
    time.sleep(3)
    st.success("Done")

# Show and update progress bar
bar = st.progress(50)
time.sleep(3)
bar.progress(100)

with st.status("Authenticating...") as s:
    time.sleep(2)
    st.write("Some long response.")
    s.update(label="Response")

st.balloons()
st.snow()
st.toast("Warming up...")
st.error("Error message")
st.warning("Warning message")
st.info("Info message")
st.success("Success message")
