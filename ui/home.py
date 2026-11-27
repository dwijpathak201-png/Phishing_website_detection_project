"""PhishGuard V4 Streamlit controller.

The original ui/home.py should first be preserved as ui/home_original.py.
"""
import logging
import streamlit as st
from ui.analysis_service import analyze_url
from ui.v4_bridge import get_home_component, get_report_component


def show_home():
    # Remove Streamlit page chrome without changing V4's own CSS.
    st.markdown("""
    <style>
      .stApp { background: #05090d; }
      .block-container {max-width:100%; padding:0 !important;}
      header[data-testid="stHeader"] {display:none;}
      [data-testid="stToolbar"] {display:none;}
      [data-testid="stBottom"] {display:none;}
      [data-testid="stVerticalBlock"] {gap:0;}
    </style>
    """, unsafe_allow_html=True)

    if 'phishguard_report' in st.session_state:
        report = st.session_state['phishguard_report']
        comp = get_report_component()
        result = comp(
            key='phishguard_report_v4',
            data={'report': report},
            on_back_change=lambda: None,
        )
        if getattr(result, 'back', None):
            st.session_state.pop('phishguard_report', None)
            st.rerun()
        return

    comp = get_home_component()
    error = st.session_state.pop('phishguard_error', None)
    result = comp(
        key='phishguard_home_v4',
        data={'error': error},
        on_analyze_change=lambda: None,
    )

    analyze_target = getattr(result, 'analyze', None)
    if analyze_target:
        try:
            st.session_state['phishguard_report'] = analyze_url(analyze_target)
        except (ValueError, KeyError) as exc:
            st.session_state['phishguard_error'] = str(exc)
        except Exception:
            logging.exception('PhishGuard URL analysis failed')
            st.session_state['phishguard_error'] = (
                'Analysis failed. Check the VS Code terminal for details.'
            )
        st.rerun()
