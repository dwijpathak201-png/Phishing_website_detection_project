"""Mount the unchanged V4 page designs as Streamlit Components V2.

The HTML files remain the source of truth. Only their embedded JavaScript
API/redirect behavior is replaced with Streamlit event handling.
"""
import re
from pathlib import Path
import streamlit as st

FRONTEND = Path(__file__).resolve().parent / "frontend"


def _parts(name):
    source = (FRONTEND / f"{name}.html").read_text(encoding="utf-8")
    css_match = re.search(r"<style[^>]*>(.*?)</style>", source, re.I | re.S)
    body_match = re.search(r"<body[^>]*>(.*?)</body>", source, re.I | re.S)
    script_match = re.search(r"<script[^>]*>(.*?)</script>", source, re.I | re.S)
    if not (css_match and body_match and script_match):
        raise ValueError(f"Missing style, body or script in {name}.html")

    raw_body = body_match.group(1)
    raw_body = re.sub(r"<script\b[^>]*>.*?</script>", "", raw_body, flags=re.I | re.S)

    css = css_match.group(1)
    # The original selectors target a whole HTML document. Inside a V2
    # component, the full design is contained in a shadow root instead.
    css = css.replace(":root{", ":host{")
    for theme in ("white", "cyan", "blue"):
        css = css.replace(
            f'html[data-theme="{theme}"]', f':host([data-theme="{theme}"])'
        )
    css = re.sub(r"\bhtml\{", ":host{", css)
    css = re.sub(r"\bbody\{", ".pg-page{", css)
    css = """@import url("https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&family=Space+Grotesk:wght@400;500;600;700&display=swap");\n""" + css
    css += """
:host { display:block; width:100%; background:#05090d; min-height:100vh; }
.pg-page { position:relative; min-height:100vh; isolation:isolate; }
.pg-page .background-scene { z-index:0; }
.pg-page .background-noise { z-index:0; }
.pg-page .site-header { position:sticky; z-index:40; }
.pg-page .main, .pg-page .container, .pg-page .footer { position:relative; z-index:1; }
.pg-page .overlay-modal { z-index:100; }
.pg-page .video-control { z-index:35; }
"""

    return '<div class="pg-page">' + raw_body + '</div>', css, script_match.group(1)


def _home_assets():
    html, css, original_js = _parts("home")
    # Replace the old local HTTP fetch() call, but leave the other V4
    # interface behaviors (theme, video pause, account preview) unchanged.
    beginning = original_js.index("   document.getElementById('scanForm').addEventListener('submit'")
    ending = original_js.index("   const modal=", beginning)
    scan_handler = r"""
   get('scanForm').addEventListener('submit', e => {
     e.preventDefault();
     const input = get('urlInput').value.trim();
     const notice = get('demoNotice');
     notice.classList.add('visible');
     if (!input) {
       notice.textContent = 'Please enter a website URL to analyze.';
       return;
     }
     notice.textContent = 'Analyzing URL structure with Random Forest V3…';
     component.setTriggerValue('analyze', input);
   });
"""
    original_js = original_js[:beginning] + scan_handler + original_js[ending:]
    original_js = original_js.replace(
        'document.documentElement', 'component.parentElement.host'
    ).replace(
        'document.getElementById(', 'get('
    ).replace(
        'document.querySelectorAll(', 'component.parentElement.querySelectorAll('
    ).replace(
        'document.addEventListener(', 'component.parentElement.addEventListener('
    )
    js = """export default function(component) {
  const get = id => component.parentElement.querySelector('#' + id);
""" + original_js + """
  if (component.data?.error) {
     const notice = get('demoNotice');
     notice.textContent = component.data.error;
     notice.classList.add('visible');
  }
  component.parentElement.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
      const id=link.getAttribute('href').slice(1);
      const target=id && get(id);
      if (target) { e.preventDefault(); target.scrollIntoView({behavior:'smooth'}); }
    });
  });
}
"""
    return html, css, js


def _report_assets():
    html, css, original_js = _parts("report")
    # Reuse the satellite video already embedded in the approved homepage.
    # No additional MP4 download or Streamlit static-file setup is needed.
    home_text = (FRONTEND / 'home.html').read_text(encoding='utf-8')
    src_match = re.search(r'data:video/mp4;base64,[A-Za-z0-9+/=]+', home_text)
    if src_match:
        html = html.replace('src="satellite.mp4"', 'src="' + src_match.group(0) + '"')

    # The report uses a file-based logo. Reuse the original mask artwork
    # already embedded in the homepage CSS so the report is self-contained.
    logo_match = re.search(r'data:image/png;base64,[A-Za-z0-9+/=]+', home_text)
    if logo_match:
        css = css.replace("'phishguard_logo.png'", "'"+logo_match.group(0)+"'")

    original_js = original_js.replace(
        'const root=document.documentElement;', 'const root=component.parentElement.host;'
    ).replace(
        'document.getElementById(', 'get('
    ).replace(
        'document.querySelectorAll(', 'component.parentElement.querySelectorAll('
    )
    original_js = re.sub(
        r"let data=null;\s*try\{data=JSON\.parse\(sessionStorage\.getItem\('phishguard-last-report'\)\|\|'null'\)\}catch\(_\)\{\}",
        'let data=component.data?.report || null;', original_js
    )
    if "sessionStorage.getItem('phishguard-last-report')" in original_js:
        raise RuntimeError('The legacy report sessionStorage script was not replaced')
    js = """export default function(component) {
  const get = id => component.parentElement.querySelector('#' + id);
  component.parentElement.querySelectorAll('a[href^="home.html"]').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      component.setTriggerValue('back', true);
    });
  });
""" + original_js + '\n}\n'
    return html, css, js


@st.cache_resource
def get_home_component():
    html, css, js = _home_assets()
    return st.components.v2.component(
        'phishguard_v4_home', html=html, css=css, js=js,
        isolate_styles=True,
    )


@st.cache_resource
def get_report_component():
    html, css, js = _report_assets()
    return st.components.v2.component(
        'phishguard_v4_report', html=html, css=css, js=js,
        isolate_styles=True,
    )
