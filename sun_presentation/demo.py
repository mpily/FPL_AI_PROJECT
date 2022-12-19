import streamlit as st
import pandas as pd
import numpy as np
data_file = 'data_fpl(1).xlsx'
pred_df = pd.read_excel(data_file,sheet_name=3)
st.dataframe(pred_df)