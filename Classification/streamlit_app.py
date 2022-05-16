import torch
import streamlit as st

# st.write(torch.FloatTensor([1., 2.]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)