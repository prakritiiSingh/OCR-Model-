import streamlit as st

def load_custom_css():
    st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }

        .section-heading {
            background-color: #f0f2f6;
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 20px;
            margin-top: 25px;
            font-weight: 600;
            border-left: 5px solid #4A90E2;
        }

        .card-box {
            background: #ffffff;
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
            margin-bottom: 20px;
        }

        .context-box {
            background: #fafafa;
            padding: 14px;
            border-radius: 10px;
            border: 1px solid #e6e6e6;
            font-size: 13px;
            height: 160px;
        }

        .answer-box {
            background: #eef8ff;
            padding: 16px;
            border-radius: 12px;
            border-left: 5px solid #4A90E2;
            margin-top: 10px;
            font-size: 15px;
        }
    </style>
    """, unsafe_allow_html=True)
