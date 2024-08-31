import streamlit as st 



def Navbar():

  st.markdown("""
  <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #800000;">
    <a class="navbar-brand" href="http://localhost:8501" target="_blank">Rans Labs</a>
    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav">
        <li class="nav-item active">
          <a class="nav-link disabled" href="http://localhost:8501" target="_self">Home <span class="sr-only">(current)</span></a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="http://localhost:8501/EDA" target="_self">EDA</a>
        </li>
         <li class="nav-item">
          <a class="nav-link" href="http://localhost:8501/FeatureEngineering" target="_self">FeatureEngineering</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="http://localhost:8501/Visualization" target="_self">Visualization</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="http://localhost:8501/Modulbuild" target="_self">Build Model</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="http://localhost:8501/Prediction" target="_self">Make Prediction</a>
        </li>
      </ul>
    </div>
  </nav>
  """, unsafe_allow_html=True)