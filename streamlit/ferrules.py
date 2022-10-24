from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import seaborn as sns


def ferrule_calibration(folder):
    def read_ferrule_data(path):
        """
        Reads the ferrule data from the specified path (single .csv file).
        """

        path = Path(path)

        # Read the ferrule data
        ferrule_data = pd.read_csv(path, header=18, sep=",")

        # Convert to seconds and microWatts
        ferrule_data["Power (W)"] = ferrule_data["Power (W)"] * 1e6
        ferrule_data["Time (ms)"] = ferrule_data["Time (ms)"] / 1e3

        # rename the columns with the correct units
        ferrule_data.rename(
            columns={"Time (ms)": "Time (s)", "Power (W)": "Power (uW)"}, inplace=True
        )

        # add a column for the ferrule ID
        ferrule_data["Ferrule"] = path.stem.split("-")[0]

        # add a column for the patch-cord identifier (or fiber split)
        ferrule_data["Fiber"] = path.stem.split("-")[1]

        # add a column for the knob value
        ferrule_data["Knob"] = path.stem.split("-")[-1]

        return ferrule_data

    folder = Path(folder)
    files = [str(p) for p in folder.glob("*.csv")]
    all_ferrule_data = []
    for f in files:
        f = Path(f)
        df = read_ferrule_data(f)
        all_ferrule_data.append(df)
    all_ferrule_data = pd.concat(all_ferrule_data)
    return all_ferrule_data.reset_index(drop=True)


user_path = st.text_input(
    "Enter the path to the folder containing the ferrule data:",
    value="/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/ferrules/",
)
df = ferrule_calibration(user_path)


@st.cache()
def ferrule_filter(df, fiber=None, ferrule=None, knob=None):
    df1 = df.copy()
    if fiber is not None:
        df1 = df1[df1["Fiber"] == fiber]
    if ferrule is not None:
        df1 = df1[df1["Ferrule"] == ferrule]
    if knob is not None:
        df1 = df1[df1["Knob"] == knob]
    return df1


"User filters ferrule data"
# user_fiber = st.text_input('Enter the fiber:', value=None)
# user_ferrule = st.text_input('Enter the ferrule:', value=None)
# user_knob = st.text_input('Enter the knob:', value=None)

new_df = ferrule_filter(df, fiber="pc2", ferrule="f4", knob=None)
st.write(new_df)

"User Chart Options"
# color_option = st.selectbox('Color Encodes:', ['Ferrule', 'Fiber', 'Knob', None])
# row_option = st.selectbox('Row Encodes:', ['Ferrule', 'Fiber', 'Knob', None])

f = px.line(
    new_df,
    x="Time (s)",
    y="Power (uW)",
    color="Knob",
    facet_row="Fiber",
    title="Ferrule Calibration",
)
st.plotly_chart(f)
