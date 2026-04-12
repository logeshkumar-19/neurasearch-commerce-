import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@600;700;800&display=swap');
        :root{
            --primary:#0f766e;
            --primary-mid:#14b8a6;
            --primary-end:#06b6d4;
            --accent:#f59e0b;
            --accent-2:#fb7185;
            --success:#16a34a;
            --bg-1:#fff7ed;
            --bg-2:#fef3c7;
            --text-1:#1f2937;
            --text-2:#4b5563;
            --text-3:#9ca3af;
            --line:rgba(15,118,110,.18);
            --glass:rgba(255,255,255,.75);
            --glass-strong:rgba(255,255,255,.9);
            --shadow-lg:0 28px 70px rgba(15,118,110,.12);
            --shadow-md:0 16px 40px rgba(17,24,39,.08);
            --match-bg:rgba(15,118,110,.12);
            --match-text:#0f766e;
            --match-border:rgba(15,118,110,.25);
        }
        .stApp{
            background:
                radial-gradient(circle at top center, rgba(14,116,144,.12) 0%, rgba(14,116,144,0) 32%),
                linear-gradient(180deg,var(--bg-1) 0%,var(--bg-2) 100%);
            color:var(--text-1);
            font-family:"Manrope","Segoe UI",sans-serif;
            position:relative;
            overflow:hidden;
        }
        .stApp:before,
        .stApp:after{
            content:"";
            position:absolute;
            width:540px;
            height:540px;
            border-radius:50%;
            filter:blur(0px);
            opacity:.25;
            pointer-events:none;
            z-index:0;
        }
        .stApp:before{
            top:-140px;
            right:-140px;
            background:radial-gradient(circle, rgba(15,118,110,.6) 0%, rgba(15,118,110,0) 70%);
        }
        .stApp:after{
            bottom:-220px;
            left:-180px;
            background:radial-gradient(circle, rgba(245,158,11,.55) 0%, rgba(245,158,11,0) 70%);
        }
        .block-container, .stApp > header, .stApp > div{
            position:relative;
            z-index:1;
        }
        [data-testid="stSidebar"]{
            background:rgba(255,255,255,.82);
            border-right:1px solid rgba(15,118,110,.12);
            backdrop-filter:blur(14px);
        }
        [data-testid="stSidebar"] .stMarkdown{
            color:var(--text-1);
        }
        [data-testid="stSidebar"] .block-container{
            padding-top:1.5rem;
        }
        .block-container{padding-top:1.6rem;padding-bottom:4rem;max-width:1180px}
        .hero{text-align:center;padding:2.2rem 0 1.1rem;max-width:900px;margin:0 auto}
        .eyebrow{display:inline-flex;align-items:center;gap:.4rem;padding:.45rem .95rem;border-radius:999px;background:rgba(15,118,110,.12);border:1px solid rgba(15,118,110,.2);color:var(--primary);font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
        .hero-title{font-family:"Space Grotesk","Manrope",sans-serif;font-size:clamp(2.7rem,4vw,4.35rem);line-height:1.02;font-weight:800;letter-spacing:-.045em;margin:1.05rem 0 .7rem}
        .hero-copy{max-width:700px;margin:0 auto;color:var(--text-2);font-size:1.02rem;line-height:1.8}
        .search-shell{max-width:860px;margin:1.6rem auto 0;padding:1.1rem;background:var(--glass);backdrop-filter:blur(18px);border:1px solid rgba(255,255,255,.65);border-radius:28px;box-shadow:var(--shadow-lg)}
        .chip-caption{color:var(--text-3);font-size:.84rem;font-weight:600;margin:.8rem 0 .55rem;text-align:center}
        .panel{background:var(--glass);backdrop-filter:blur(18px);border:1px solid rgba(255,255,255,.65);border-radius:28px;box-shadow:var(--shadow-lg);padding:1.5rem;margin:1.35rem 0}
        .title{font-family:"Space Grotesk","Manrope",sans-serif;font-size:1.35rem;font-weight:760;letter-spacing:-.025em;margin-bottom:.28rem}
        .copy{color:var(--text-2);margin-bottom:1.05rem;line-height:1.65}
        .answer{background:var(--glass-strong);backdrop-filter:blur(14px);border:1px solid var(--line);border-radius:20px;padding:1.05rem 1.1rem;line-height:1.75;color:var(--text-1);box-shadow:0 12px 32px rgba(15,118,110,.08)}
        .card{background:rgba(255,255,255,.72);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);border:1px solid rgba(255,255,255,.7);border-radius:22px;padding:1.25rem;box-shadow:var(--shadow-md);min-height:320px;transition:transform .24s ease,box-shadow .24s ease,border-color .24s ease}
        .card:hover{transform:translateY(-6px);box-shadow:0 24px 50px rgba(15,118,110,.16);border-color:rgba(15,118,110,.25)}
        .name{font-family:"Space Grotesk","Manrope",sans-serif;font-size:1.12rem;font-weight:760;line-height:1.45;letter-spacing:-.02em;margin-bottom:.95rem}
        .meta{display:flex;flex-wrap:wrap;gap:.55rem;margin-bottom:.95rem}
        .pill{display:inline-flex;align-items:center;border-radius:999px;padding:.4rem .82rem;font-size:.78rem;font-weight:700;border:1px solid transparent}
        .price{font-size:1.18rem;font-weight:780;color:var(--success);margin-bottom:.88rem}
        .body{color:var(--text-2);line-height:1.75;font-size:.95rem}
        .match{display:flex;justify-content:space-between;align-items:center;margin:1.05rem 0 .45rem;color:var(--text-2);font-size:.82rem;font-weight:650}
        .bar{width:100%;height:10px;border-radius:999px;overflow:hidden;background:rgba(15,118,110,.12)}
        .fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#0f766e 0%,#06b6d4 100%)}
        .section-heading{font-family:"Space Grotesk","Manrope",sans-serif;font-size:1.75rem;font-weight:780;letter-spacing:-.035em;margin:.25rem 0 1rem}
        .stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]>div{border-radius:18px!important;border:1px solid rgba(15,118,110,.2)!important;min-height:3.5rem;background:rgba(255,255,255,.94)!important;color:var(--text-1)!important;box-shadow:0 8px 24px rgba(17,24,39,.05)!important;transition:border-color .22s ease, box-shadow .22s ease!important}
        .stTextArea textarea{border-radius:22px!important;min-height:7rem!important}
        .stTextInput input:focus,.stTextArea textarea:focus{border-color:rgba(15,118,110,.45)!important;box-shadow:0 0 0 4px rgba(15,118,110,.16)!important}
        .stButton>button{border-radius:999px!important;min-height:3rem;padding:0 1.2rem!important;font-weight:700!important;border:1px solid rgba(15,118,110,.18)!important;color:var(--text-1)!important;background:rgba(255,255,255,.85)!important;transition:transform .22s ease,box-shadow .22s ease,filter .22s ease!important;box-shadow:0 10px 24px rgba(17,24,39,.05)}
        .stButton>button:hover{transform:translateY(-1px);box-shadow:0 14px 28px rgba(17,24,39,.08)!important}
        .stButton>button[kind="primary"]{background:linear-gradient(135deg,#0f766e 0%,#14b8a6 55%,#06b6d4 100%)!important;color:#fff!important;border-color:transparent!important}
        .stButton>button[kind="primary"]:hover{filter:brightness(.96)}
        .stAlert{border-radius:18px!important;border:1px solid var(--line)!important}
        .match-pill{
            background:var(--match-bg);
            color:var(--match-text);
            border-color:var(--match-border);
        }
        @media (max-width: 900px){
            .block-container{padding-left:1rem;padding-right:1rem}
            .panel{padding:1.1rem}
            .search-shell{padding:.9rem}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
