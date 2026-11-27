## ORIGINAL REFERENCE FOR FRONTEND ##

# Aasli ka rabbit hole yaha sa start ha:
##

import streamlit as st
from detection_logic.url_analysis import validate_url


def show_home():
    st.title("Phishing_detection_system!")
    st.caption("Detect. Verify. Protect.")
    st.markdown("""<hr> </hr>""", unsafe_allow_html=True)
    # st.markdown("""<br> </br> """,unsafe_allow_html=True)

    st.subheader("FOR THE JUST TO BE SERVED!!")
    st.text("""Analyzes suspicious URLs using multiple technical indicators to estimate their phishing risk. It provides an understandable risk score along with the factors that contributed to the result.""")

    st.text("""01 — Enter a URL
The user enters the website link they want to check.

02 — Check the URL & Domain
PhishGuard looks for suspicious patterns in the link and its domain, such as unusual words, long URLs, or strange domain structures.

03 — Check the Website
It can also examine technical details of the website, such as DNS information and webpage elements.

04 — Calculate Risk
All these checks are combined to estimate how suspicious the website is.

05 — Show the Result
The user gets a risk score and a simple explanation of what made the website suspicious.""")

    st.markdown("""<hr> </hr> """, unsafe_allow_html=True)

    st.subheader("IS IT TRUE TO THERE WORD?")
    st.write("Enter a website URL to check for suspicious activity.")

    url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        label_visibility="collapsed"
    )

    # -------------------------
    # ANALYZE URL
    # -------------------------

    if st.button("ANALYZE URL"):

        is_valid, result = validate_url(url)

        if not is_valid:
            st.error(result)

        else:
            st.success(f"Valid URL: {result}")

    st.markdown("## RISK LEVELS")
    st.write("PhishGuard classifies websites based on the number and severity of suspicious indicators.")

    # LOW RISK
    st.markdown("""
    <div style="
        padding:5px;
        border-radius:12px;
        background-color:#102416;
        margin-bottom:15px;
    ">
        <h3>🟢 LOW RISK — 0–24</h3>
        <p>
        Few or no suspicious indicators were detected.
        The URL has a generally normal structure and domain.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # MEDIUM RISK
    st.markdown("""
    <div style="
        padding:5px;
        border-radius:12px;
        background-color:#292510;
        margin-bottom:15px;
    ">
        <h3>🟡 MEDIUM RISK — 25–49</h3>
        <p>
        Some unusual indicators were detected.
        The website should be approached with caution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # HIGH RISK
    st.markdown("""
    <div style="
        padding:5px;
        border-radius:12px;
        background-color:#2b1c0e;
        margin-bottom:15px;
    ">
        <h3>🟠 HIGH RISK — 50–74</h3>
        <p>
        Multiple suspicious indicators were detected.
        Verify the website before entering personal information.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # CRITICAL RISK
    st.markdown("""
    <div style="
        padding:5px;
        border-radius:12px;
        background-color:#2b1010;
        margin-bottom:15px;
    ">
        <h3>🔴 CRITICAL RISK — 75–100</h3>
        <p>
        Strong phishing-related indicators were detected.
        Avoid entering sensitive information unless the website is verified.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        
    </style>
    """, unsafe_allow_html=True)