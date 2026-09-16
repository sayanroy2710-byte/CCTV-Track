"""
Streamlit Web Analytics & Administration Portal.
Provides an interactive management dashboard for mall supervisors to monitor
real-time footfall, inspect individual visitor journeys, analyze dwell times,
and review visual ReID snapshots.
"""
import os
import json
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image

import sys
import subprocess
import config

st.set_page_config(
    page_title="Martian CCTV Intelligence Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark modern theme
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .kpi-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid #2e3444;
        margin-bottom: 12px;
    }
    .kpi-title { font-size: 13px; color: #8c9ba5; font-weight: 500; text-transform: uppercase; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #00d2ff; margin-top: 4px; }
    .status-inside { color: #00e676; font-weight: bold; }
    .status-exited { color: #ff9100; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def get_db_data():
    if not os.path.exists(str(config.DATABASE_PATH)):
        return pd.DataFrame(), pd.DataFrame()
        
    conn = sqlite3.connect(str(config.DATABASE_PATH))
    try:
        df_visitors = pd.read_sql_query("SELECT * FROM visitors ORDER BY entry_time DESC", conn)
        df_events = pd.read_sql_query("SELECT * FROM camera_events ORDER BY timestamp DESC LIMIT 50", conn)
    except Exception:
        df_visitors = pd.DataFrame()
        df_events = pd.DataFrame()
    finally:
        conn.close()
        
    return df_visitors, df_events


# Sidebar
st.sidebar.title("🛡️ CCTV Control Center")
st.sidebar.markdown("**Martian Corporation — Mall Surveillance**")
st.sidebar.markdown("---")

auto_refresh = st.sidebar.checkbox("Auto-refresh data (every 5s)", value=True)
if auto_refresh:
    st.empty()

filter_status = st.sidebar.selectbox("Filter Visitor Status", ["ALL", "INSIDE", "EXITED"])
search_id = st.sidebar.text_input("Search Person ID (e.g. Person-001)", "").strip().upper()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Database:** `{config.DATABASE_PATH.name}`")
st.sidebar.markdown(f"**ReID Backbone:** `{config.REID_BACKBONE}`")
st.sidebar.markdown(f"**ReID Threshold:** `{config.REID_SIMILARITY_THRESHOLD}`")

# Video Selection & Processing Section
st.sidebar.markdown("---")
with st.sidebar.expander("📹 Select & Track Video", expanded=False):
    st.markdown("**Run AI Tracking on New Video**")
    uploaded_vid = st.file_uploader("Upload video file (.mp4, .avi, .mov)", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_vid is not None:
        target_path = config.VIDEO_DIR / uploaded_vid.name
        with open(target_path, "wb") as f:
            f.write(uploaded_vid.getbuffer())
        st.success(f"Uploaded: {uploaded_vid.name}")
        if st.button("▶ Run AI Tracking on This Video"):
            with st.spinner("Processing video through YOLO11 & Deep ReID..."):
                cmd = [sys.executable, "main.py", "--sources", str(target_path), "--headless"]
                subprocess.run(cmd, cwd=str(config.BASE_DIR))
            st.success("Tracking complete! Reloading database...")
            st.rerun()

    # Select existing local video
    existing_videos = list(config.VIDEO_DIR.glob("*.mp4")) + list(config.VIDEO_DIR.glob("*.avi"))
    if existing_videos:
        vid_names = [v.name for v in existing_videos]
        chosen_vid = st.selectbox("Or choose existing sample video:", vid_names)
        if st.button("▶ Track Selected Sample Video"):
            with st.spinner("Processing sample video..."):
                cmd = [sys.executable, "main.py", "--sources", str(config.VIDEO_DIR / chosen_vid), "--headless"]
                subprocess.run(cmd, cwd=str(config.BASE_DIR))
            st.success("Tracking complete! Reloading database...")
            st.rerun()

# Header
st.title("🏬 Public CCTV Intelligence & People Tracking")
st.caption("Cross-Camera Re-Identification, Footfall Counting, & Dwell-Time Analytics")

def clean_time_str(val):
    if not val or pd.isna(val) or str(val).strip() in ["None", "", "nat", "NaN"]:
        return "Still Inside"
    try:
        dt = datetime.fromisoformat(str(val))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        s = str(val).replace("T", " ")
        return s[:19] if len(s) >= 19 else s


# Load data
df_visitors, df_events = get_db_data()

if df_visitors.empty:
    st.info("No visitor tracking data recorded yet. Start `python main.py` to begin CCTV inference.")
else:
    # 1. Top KPI Row
    total_entered = len(df_visitors)
    active_visitors = len(df_visitors[df_visitors["status"] == "INSIDE"])
    exited_visitors = len(df_visitors[df_visitors["status"] == "EXITED"])
    
    avg_dwell_sec = df_visitors["dwell_time_seconds"].mean()
    if pd.isna(avg_dwell_sec) or avg_dwell_sec <= 0:
        avg_dwell_str = "0m 00s"
    else:
        m, s = divmod(int(avg_dwell_sec), 60)
        avg_dwell_str = f"{m}m {s}s"
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Footfall Entered</div>
            <div class="kpi-value" style="color: #00d2ff;">{total_entered}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Currently in Premises</div>
            <div class="kpi-value" style="color: #00e676;">{active_visitors}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Exited</div>
            <div class="kpi-value" style="color: #ff9100;">{exited_visitors}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Dwell Time</div>
            <div class="kpi-value" style="color: #ff5252;">{avg_dwell_str}</div>
        </div>
        """, unsafe_allow_html=True)

    # Filter Visitors
    filtered_df = df_visitors.copy()
    if filter_status != "ALL":
        filtered_df = filtered_df[filtered_df["status"] == filter_status]
    if search_id:
        filtered_df = filtered_df[filtered_df["global_id"].str.contains(search_id, case=False, na=False)]

    st.markdown("---")
    
    # 2. Tabs: Live Visitors, Journey Details, Dwell Time Analytics
    tab1, tab2, tab3 = st.tabs(["📋 Visitor Directory & Status", "🔍 Person Journey & ReID Gallery", "📊 Dwell Time Analytics"])
    
    with tab1:
        st.subheader("Real-Time Footfall Registry")
        table_df = filtered_df.copy()
        table_df["Entry Time"] = table_df["entry_time"].apply(clean_time_str)
        table_df["Exit Time"] = table_df["exit_time"].apply(clean_time_str)
        table_df["Status"] = table_df["status"].apply(lambda s: "🟢 INSIDE" if s == "INSIDE" else "🟠 EXITED")
        table_df["Dwell Time"] = table_df["dwell_time_str"]
        table_df["Entry Point"] = table_df["entry_camera"]
        table_df["Last Location"] = table_df["last_camera"]
        table_df["Frames Seen"] = table_df["total_detections"]
        table_df["Global ID"] = table_df["global_id"]
        
        cols = ["Global ID", "Status", "Entry Time", "Exit Time", "Dwell Time", "Entry Point", "Last Location", "Frames Seen"]
        st.dataframe(
            table_df[cols],
            use_container_width=True,
            hide_index=True
        )
        
        # CSV Export
        csv_export_df = table_df[cols]
        csv_data = csv_export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Footfall Report (CSV)",
            data=csv_data,
            file_name=f"mall_footfall_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    with tab2:
        st.subheader("Individual Visitor Tracking & ReID Visual Memory")
        selected_gid = st.selectbox("Select Individual to Inspect", options=df_visitors["global_id"].tolist())
        
        if selected_gid:
            person_data = df_visitors[df_visitors["global_id"] == selected_gid].iloc[0]
            
            c_left, c_right = st.columns([1, 2])
            with c_left:
                st.markdown(f"### `{person_data['global_id']}`")
                status_icon = "🟢" if person_data['status'] == "INSIDE" else "🟠"
                st.write(f"**Status:** {status_icon} {person_data['status']}")
                st.write(f"**Entry Time:** {clean_time_str(person_data['entry_time'])}")
                st.write(f"**Exit Time:** {clean_time_str(person_data['exit_time'])}")
                st.write(f"**Total Dwell Time:** {person_data['dwell_time_str']}")
                st.write(f"**Entry Camera:** {person_data['entry_camera']}")
                st.write(f"**Last Camera:** {person_data['last_camera']}")
                
                # Show snapshot if available
                thumb = person_data.get("thumbnail_path", "")
                if thumb and os.path.exists(thumb):
                    st.image(Image.open(thumb), caption=f"ReID Snapshot: {person_data['global_id']}", width=180)
                else:
                    st.info("No visual crop thumbnail available.")
                    
            with c_right:
                st.markdown("#### Camera Movement Sequence")
                try:
                    seq = json.loads(person_data["camera_sequence"])
                    st.write(" ➔ ".join([f"**`{cam}`**" for cam in seq]))
                except Exception:
                    st.write(person_data["camera_sequence"])
                    
                st.markdown("#### Journey Audit Log")
                person_events = df_events[df_events["global_id"] == selected_gid]
                if not person_events.empty:
                    pe_df = person_events.copy()
                    pe_df["Timestamp"] = pe_df["timestamp"].apply(clean_time_str)
                    st.table(pe_df[["Timestamp", "camera_id", "event_type"]].rename(columns={
                        "camera_id": "Camera",
                        "event_type": "Event"
                    }))
                else:
                    st.write("No granular transition events logged yet.")

    with tab3:
        st.subheader("Visitor Dwell Time Analytics")
        if df_visitors.empty:
            st.info("No visitor data recorded yet.")
        else:
            chart_df = df_visitors.copy()
            chart_df["Dwell (Seconds)"] = chart_df["dwell_time_seconds"].astype(float)
            chart_df["Visitor"] = chart_df["global_id"]
            
            st.markdown("#### Individual Visitor Dwell Durations")
            st.bar_chart(chart_df.set_index("Visitor")["Dwell (Seconds)"], use_container_width=True)
            
            st.markdown("#### Detailed Dwell Breakdown Table")
            summary_cols = chart_df[["global_id", "status", "dwell_time_str", "dwell_time_seconds", "total_detections"]].copy()
            summary_cols["Status"] = summary_cols["status"].apply(lambda s: "🟢 INSIDE" if s == "INSIDE" else "🟠 EXITED")
            st.dataframe(
                summary_cols.rename(columns={
                    "global_id": "Visitor ID",
                    "status": "Status",
                    "dwell_time_str": "Total Dwell Time",
                    "dwell_time_seconds": "Dwell Seconds",
                    "total_detections": "Frames Seen"
                }),
                use_container_width=True,
                hide_index=True
            )