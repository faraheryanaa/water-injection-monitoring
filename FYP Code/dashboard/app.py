import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Water Injection Monitoring System",
    page_icon="💧",
    layout="wide"
)


# ==================================================
# LOAD DATA
# ==================================================

DATA_PATH = "../data/processed/dashboard_data.xlsx"

df = pd.read_excel(DATA_PATH)

# Convert Time column to datetime
df["Time"] = pd.to_datetime(df["Time"])

# Sort chronologically
df = df.sort_values("Time").reset_index(drop=True)


# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================

if "simulation_index" not in st.session_state:
    st.session_state.simulation_index = 0

if "notifications" not in st.session_state:
    st.session_state.notifications = []

if "previous_live_mode" not in st.session_state:
    st.session_state.previous_live_mode = False


# ==================================================
# PAGE TITLE
# ==================================================

st.title("💧 Water Injection Monitoring System")

st.markdown(
    "Integrated monitoring of water injection operations "
    "using Digital Twin and Machine Learning."
)

st.divider()


# ==================================================
# PAGE NAVIGATION
# ==================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overall Monitoring",
        "Digital Twin",
        "Machine Learning"
    ]
)

# ==================================================
# SIDEBAR FILTER
# ==================================================

st.sidebar.header("Dashboard Filter")

min_date = df["Time"].min().date()
max_date = df["Time"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


# ==================================================
# FILTER DATA BY DATE
# ==================================================

if len(date_range) == 2:

    start_date = pd.Timestamp(date_range[0])

    # Add one day so the selected end date
    # is fully included
    end_date = (
        pd.Timestamp(date_range[1])
        + pd.Timedelta(days=1)
    )

    filtered_df = df[
        (df["Time"] >= start_date)
        &
        (df["Time"] < end_date)
    ].copy()

else:

    filtered_df = df.copy()


# ==================================================
# LIVE MONITORING SETTINGS
# ==================================================

st.sidebar.divider()

st.sidebar.header("Live Monitoring")

live_mode = st.sidebar.toggle(
    "Enable Near-Real-Time Simulation",
    value=False
)

update_interval = st.sidebar.slider(
    "Update Interval (seconds)",
    min_value=1,
    max_value=10,
    value=3
)


# ==================================================
# SESSION STATE FOR LIVE MODE
# ==================================================

if "previous_live_mode" not in st.session_state:
    st.session_state.previous_live_mode = False


# ==================================================
# RESET SIMULATION BUTTON
# ==================================================

reset_simulation = st.sidebar.button(
    "🔄 Restart Simulation"
)

if reset_simulation:

    st.session_state.simulation_index = 0

    st.session_state.notifications = []

    st.rerun()


# ==================================================
# RESET WHEN LIVE MODE IS ENABLED
# ==================================================

if (
    live_mode
    and not st.session_state.previous_live_mode
):

    st.session_state.simulation_index = 0

    st.session_state.notifications = []


# Store current Live Mode state
st.session_state.previous_live_mode = live_mode

# =====================================================
# AUTOMATIC LIVE SIMULATION TIMER
# =====================================================

if "last_live_tick" not in st.session_state:
    st.session_state.last_live_tick = time.monotonic()

if live_mode:

    @st.fragment(run_every=update_interval)
    def live_simulation_timer():

        now = time.monotonic()

        if now - st.session_state.last_live_tick >= update_interval:

            if (
                len(filtered_df) > 0
                and st.session_state.simulation_index
                < len(filtered_df) - 1
            ):
                st.session_state.simulation_index += 1
                st.session_state.last_live_tick = now
                st.rerun()

    live_simulation_timer()


# ==================================================
# NEAR-REAL-TIME DATA SIMULATION
# ==================================================

if live_mode:

    # --------------------------------------------------
    # CHECK WHETHER DATA IS AVAILABLE
    # --------------------------------------------------

    if len(filtered_df) == 0:

        live_df = filtered_df.copy()

    else:

        # --------------------------------------------------
        # MAKE SURE SIMULATION INDEX IS VALID
        # --------------------------------------------------

        if (
            st.session_state.simulation_index
            >= len(filtered_df)
        ):

            st.session_state.simulation_index = 0

            st.session_state.notifications = []

        

        # --------------------------------------------------
        # GET CURRENT SIMULATED DATA
        # --------------------------------------------------

        live_df = filtered_df.iloc[
            :st.session_state.simulation_index + 1
        ].copy()

        # --------------------------------------------------
        # CHECK LATEST SIMULATED OBSERVATION
        # --------------------------------------------------

        latest_live = live_df.iloc[-1]


        # --------------------------------------------------
        # CHECK FOR ANOMALY
        # --------------------------------------------------

        if latest_live["Anomaly"] == -1:

            notification = {
                "Time": latest_live["Time"],
                "Injection_Rate": latest_live[
                    "Injection_Rate"
                ],
                "Injection_Pressure": latest_live[
                    "Injection_Pressure"
                ],
                "Anomaly_Score": latest_live[
                    "Anomaly_Score"
                ],
                "Notification": latest_live[
                    "Notification"
                ],
                "Suggested_Action": latest_live[
                    "Suggested_Action"
                ]
            }


            # --------------------------------------------------
            # PREVENT DUPLICATE NOTIFICATIONS
            # --------------------------------------------------

            existing_times = [
                n["Time"]
                for n in st.session_state.notifications
            ]


            if (
                latest_live["Time"]
                not in existing_times
            ):

                st.session_state.notifications.append(
                    notification
                )

else:

    # --------------------------------------------------
    # NORMAL DASHBOARD MODE
    # --------------------------------------------------

    live_df = filtered_df.copy()


# ==================================================
# NOTIFICATION BELL
# ==================================================

notification_count = len(
    st.session_state.notifications
)

bell_col1, bell_col2 = st.columns([10, 1])

with bell_col2:

    if notification_count > 0:

        with st.popover(
            f"🔔 {notification_count}"
        ):

            st.markdown("### 🔔 Active Alerts")

            st.caption(
                f"{notification_count} anomalous "
                "observations detected."
            )

            st.divider()

            # Show latest alerts first
            recent_alerts = list(
                reversed(
                    st.session_state.notifications[-10:]
                )
            )

            for alert in recent_alerts:

                st.markdown(
                    f"**🚨 {alert['Notification']}**"
                )

                st.caption(
                    f"Time: {alert['Time']}"
                )

                st.write(
                    f"**Injection Rate:** "
                    f"{alert['Injection_Rate']:.2f}"
                )

                st.write(
                    f"**Injection Pressure:** "
                    f"{alert['Injection_Pressure']:.2f}"
                )

                st.write(
                    f"**Anomaly Score:** "
                    f"{alert['Anomaly_Score']:.6f}"
                )

                st.info(
                    f"**Suggested Action:** "
                    f"{alert['Suggested_Action']}"
                )

                st.divider()

    else:

        with st.popover("🔔 0"):

            st.markdown("### 🔔 Notifications")

            st.success(
                "✓ No new anomalies detected."
            )

# ==================================================
# MACHINE LEARNING PAGE
# ==================================================



if page == "Machine Learning":

    st.title("🧠 Machine Learning")

    st.markdown(
        "Machine learning-based anomaly detection for "
        "identifying abnormal water injection operating "
        "behaviour."
    )

    st.divider()


    # ==================================================
    # ML DATA SOURCE
    # ==================================================

    if live_mode:

        ml_df = live_df

    else:

        ml_df = filtered_df


    # ==================================================
    # ML OVERVIEW
    # ==================================================

    st.subheader("Anomaly Detection Overview")

    ml_total = len(ml_df)

    ml_anomalies = (
        ml_df["Anomaly"] == -1
    ).sum()

    ml_normal = (
        ml_df["Anomaly"] == 1
    ).sum()

    if ml_total > 0:

        ml_anomaly_rate = (
            ml_anomalies / ml_total
        ) * 100

    else:

        ml_anomaly_rate = 0


    # ==================================================
    # ML KPI CARDS
    # ==================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Observations",
            f"{ml_total:,}"
        )

    with col2:

        st.metric(
            "Normal Observations",
            f"{ml_normal:,}"
        )

    with col3:

        st.metric(
            "Detected Anomalies",
            f"{ml_anomalies:,}"
        )

    with col4:

        st.metric(
            "Anomaly Rate",
            f"{ml_anomaly_rate:.2f}%"
        )


    st.divider()


    # ==================================================
    # LATEST ML ASSESSMENT
    # ==================================================

    st.subheader("Latest ML Assessment")

    if ml_total > 0:

        latest_ml = ml_df.iloc[-1]

        latest_anomaly = latest_ml["Anomaly"]

        latest_score = latest_ml[
            "Anomaly_Score"
        ]

        latest_rate = latest_ml[
            "Injection_Rate"
        ]

        latest_pressure = latest_ml[
            "Injection_Pressure"
        ]


        # --------------------------------------------------
        # STATUS MESSAGE
        # --------------------------------------------------

        if latest_anomaly == -1:

            st.error(
                "⚠️ Anomaly Detected"
            )

        else:

            st.success(
                "✓ Normal Behaviour"
            )


        # --------------------------------------------------
        # LATEST ML METRICS
        # --------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Anomaly Status",
                "Anomaly"
                if latest_anomaly == -1
                else "Normal"
            )

        with col2:

            st.metric(
                "Anomaly Score",
                f"{latest_score:.6f}"
            )

        with col3:

            st.metric(
                "Injection Rate",
                f"{latest_rate:.2f}"
            )

        with col4:

            st.metric(
                "Injection Pressure",
                f"{latest_pressure:.2f}"
            )

    else:

        st.warning(
            "No observations are available "
            "for the selected date range."
        )


    st.divider()


    # ==================================================
    # ANOMALY SCORE OVER TIME
    # ==================================================

    st.subheader("Anomaly Score Over Time")

    fig_score = go.Figure()


    # --------------------------------------------------
    # SCORE LINE
    # --------------------------------------------------

    fig_score.add_trace(
        go.Scatter(
            x=ml_df["Time"],
            y=ml_df["Anomaly_Score"],
            mode="lines",
            name="Anomaly Score"
        )
    )


    # --------------------------------------------------
    # ANOMALY POINTS
    # --------------------------------------------------

    anomaly_score_df = ml_df[
        ml_df["Anomaly"] == -1
    ]

    fig_score.add_trace(
        go.Scatter(
            x=anomaly_score_df["Time"],
            y=anomaly_score_df[
                "Anomaly_Score"
            ],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(
                size=9,
                symbol="circle"
            )
        )
    )


    fig_score.update_layout(
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        hovermode="x unified",
        height=450
    )


    st.plotly_chart(
        fig_score,
        use_container_width=True
    )


    st.divider()


    # ==================================================
    # ANOMALY TYPE SUMMARY
    # ==================================================

    st.subheader("Detected Anomaly Types")

    if ml_anomalies > 0:

        anomaly_types = (
            ml_df[
                ml_df["Anomaly"] == -1
            ]["Notification"]
            .value_counts()
            .reset_index()
        )

        anomaly_types.columns = [
            "Anomaly Type",
            "Occurrences"
        ]

        st.dataframe(
            anomaly_types,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No anomalies detected within "
            "the selected period."
        )


    st.divider()


    # ==================================================
    # ANOMALY RECORDS
    # ==================================================

    st.subheader("Detected Anomaly Records")

    if ml_anomalies > 0:

        anomaly_records = ml_df[
            ml_df["Anomaly"] == -1
        ][
            [
                "Time",
                "Injection_Rate",
                "Injection_Pressure",
                "Anomaly_Score",
                "Notification",
                "Suggested_Action"
            ]
        ].copy()

        st.dataframe(
            anomaly_records,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No anomalous records are available "
            "for the selected period."
        )


    st.divider()


    # ==================================================
    # MACHINE LEARNING INTERPRETATION
    # ==================================================

    st.subheader(
        "Machine Learning Interpretation"
    )

    st.info(
        "The Isolation Forest model analyses historical "
        "operating data to identify observations that "
        "differ from normal operating patterns. Each "
        "observation is assigned an anomaly status and "
        "anomaly score. Detected anomalies are further "
        "classified using the corresponding operating "
        "conditions and notification rules."
    )

    # ==================================================
    # DIGITAL TWIN PAGE
    # ==================================================

if page == "Digital Twin":

        st.title("💧 Digital Twin")

        st.markdown(
            "Comparison between expected and actual water "
            "injection operating conditions."
        )

        st.divider()

    # ==================================================
    # DIGITAL TWIN DATA SOURCE
    # ==================================================

        if live_mode:

            twin_df = live_df

        else:

            twin_df = filtered_df


    # ==================================================
    # DIGITAL TWIN REFERENCE
    # ==================================================

        st.subheader(
            "Expected Operating Condition"
        )

        st.markdown(
            "The Digital Twin represents the expected "
            "behaviour of the water injection system based "
            "on normal historical observations."
    )


    # ==================================================
    # CHECK DATA AVAILABILITY
    # ==================================================

        if len(twin_df) > 0:

        # --------------------------------------------------
        # LATEST TWIN OBSERVATION
        # --------------------------------------------------

            latest_twin = twin_df.iloc[-1]


        # --------------------------------------------------
        # ACTUAL AND EXPECTED RATE
        # --------------------------------------------------

        actual_rate = latest_twin[
            "Injection_Rate"
        ]

        expected_rate = latest_twin[
            "Expected_Rate"
        ]

        rate_deviation = latest_twin[
            "Rate_Deviation"
        ]

        rate_deviation_pct = latest_twin[
            "Rate_Deviation_%"
        ]

        rate_status = latest_twin[
            "Rate_Status"
        ]


        # --------------------------------------------------
        # ACTUAL AND EXPECTED PRESSURE
        # --------------------------------------------------

        actual_pressure = latest_twin[
            "Injection_Pressure"
        ]

        expected_pressure = latest_twin[
            "Expected_Pressure"
        ]

        pressure_deviation = latest_twin[
            "Pressure_Deviation"
        ]

        pressure_deviation_pct = latest_twin[
            "Pressure_Deviation_%"
        ]

        pressure_status = latest_twin[
            "Pressure_Status"
        ]


        # ==================================================
        # CURRENT OPERATING CONDITION
        # ==================================================

        st.subheader(
            "Current Operating Condition"
        )


        # --------------------------------------------------
        # INJECTION RATE KPI
        # --------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Actual Injection Rate",
                f"{actual_rate:.2f}"
            )

        with col2:

            st.metric(
                "Expected Injection Rate",
                f"{expected_rate:.2f}"
            )

        with col3:

            st.metric(
                "Rate Difference",
                f"{rate_deviation:+.2f}"
            )

        with col4:

            st.metric(
                "Rate Deviation (%)",
                f"{rate_deviation_pct:+.2f}%"
            )


        # --------------------------------------------------
        # INJECTION PRESSURE KPI
        # --------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Actual Injection Pressure",
                f"{actual_pressure:.2f}"
            )

        with col2:

            st.metric(
                "Expected Injection Pressure",
                f"{expected_pressure:.2f}"
            )

        with col3:

            st.metric(
                "Pressure Difference",
                f"{pressure_deviation:+.2f}"
            )

        with col4:

            st.metric(
                "Pressure Deviation (%)",
                f"{pressure_deviation_pct:+.2f}%"
            )


        st.divider()


        # ==================================================
        # INJECTION RATE ASSESSMENT
        # ==================================================

        st.subheader(
            "Injection Rate Assessment"
        )


        if rate_status == "Critical Deviation":

            st.error(
                f"⚠️ {rate_status}: "
                f"Actual injection rate is "
                f"{rate_deviation_pct:+.2f}% "
                f"from the expected operating condition."
            )

        elif "Deviation" in str(rate_status):

            st.warning(
                f"⚠️ {rate_status}: "
                f"Actual injection rate is "
                f"{rate_deviation_pct:+.2f}% "
                f"from the expected operating condition."
            )

        else:

            st.success(
                "✓ Injection rate is within the "
                "expected operating condition."
            )

        
        # RECOMMENDED ACTION - INJECTION RATE
        if rate_status == "Critical Deviation":
            rate_action = (
                "Verify the flow-rate reading and compare it with "
                "the expected operating range. Review recent flow "
                "trends and investigate persistent deviations."
            )
        elif "Deviation" in str(rate_status):
            rate_action = (
                "Verify the flow-rate reading and monitor its trend. "
                "If the deviation persists, review recent operating "
                "changes and possible measurement issues."
            )
        else:
            rate_action = (
                "Continue monitoring the injection rate and compare "
                "readings with the expected operating range."
            )

        st.info(f"**Recommended Action:** {rate_action}")


        # ==================================================
        # INJECTION PRESSURE ASSESSMENT
        # ==================================================

        st.subheader(
            "Injection Pressure Assessment"
        )


        if pressure_status == "Critical Deviation":

            st.error(
                f"⚠️ {pressure_status}: "
                f"Actual injection pressure is "
                f"{pressure_deviation_pct:+.2f}% "
                f"from the expected operating condition."
            )

        elif "Deviation" in str(pressure_status):

            st.warning(
                f"⚠️ {pressure_status}: "
                f"Actual injection pressure is "
                f"{pressure_deviation_pct:+.2f}% "
                f"from the expected operating condition."
            )

        else:

            st.success(
                "✓ Injection pressure is within the "
                "expected operating condition."
            )

        
        # RECOMMENDED ACTION - INJECTION PRESSURE
        if pressure_status == "Critical Deviation":
            pressure_action = (
                "Verify the pressure reading and compare it with "
                "the approved operating limit. If the limit is "
                "exceeded, follow site procedures and notify "
                "the responsible engineer."
            )
        elif "Deviation" in str(pressure_status):
            pressure_action = (
                "Verify the pressure reading and monitor its trend. "
                "Compare it with the approved operating limit and "
                "investigate if the deviation persists."
            )
        else:
            pressure_action = (
                "Continue monitoring injection pressure and compare "
                "readings with the approved operating limit."
            )

        st.info(f"**Recommended Action:** {pressure_action}")


        st.divider()


        # ==================================================
        # EXPECTED VS ACTUAL INJECTION RATE
        # ==================================================

        st.subheader(
            "Expected vs Actual Injection Rate"
        )

        fig_rate_twin = go.Figure()


        # --------------------------------------------------
        # ACTUAL RATE
        # --------------------------------------------------

        fig_rate_twin.add_trace(
            go.Scatter(
                x=twin_df["Time"],
                y=twin_df["Injection_Rate"],
                mode="lines",
                name="Actual Injection Rate"
            )
        )


        # --------------------------------------------------
        # EXPECTED RATE
        # --------------------------------------------------

        fig_rate_twin.add_trace(
            go.Scatter(
                x=twin_df["Time"],
                y=twin_df["Expected_Rate"],
                mode="lines",
                name="Expected Injection Rate",
                line=dict(
                    dash="dash"
                )
            )
        )


        fig_rate_twin.update_layout(
            xaxis_title="Time",
            yaxis_title="Injection Rate",
            hovermode="x unified",
            height=450
        )


        st.plotly_chart(
            fig_rate_twin,
            use_container_width=True
        )


        # ==================================================
        # EXPECTED VS ACTUAL INJECTION PRESSURE
        # ==================================================

        st.subheader(
            "Expected vs Actual Injection Pressure"
        )

        fig_pressure_twin = go.Figure()


        # --------------------------------------------------
        # ACTUAL PRESSURE
        # --------------------------------------------------

        fig_pressure_twin.add_trace(
            go.Scatter(
                x=twin_df["Time"],
                y=twin_df["Injection_Pressure"],
                mode="lines",
                name="Actual Injection Pressure"
            )
        )


        # --------------------------------------------------
        # EXPECTED PRESSURE
        # --------------------------------------------------

        fig_pressure_twin.add_trace(
            go.Scatter(
                x=twin_df["Time"],
                y=twin_df["Expected_Pressure"],
                mode="lines",
                name="Expected Injection Pressure",
                line=dict(
                    dash="dash"
                )
            )
        )


        fig_pressure_twin.update_layout(
            xaxis_title="Time",
            yaxis_title="Injection Pressure",
            hovermode="x unified",
            height=450
        )


        st.plotly_chart(
            fig_pressure_twin,
            use_container_width=True
        )


        # ==================================================
        # DIGITAL TWIN INTERPRETATION
        # ==================================================

        st.divider()

        st.subheader(
            "Digital Twin Interpretation"
        )

        st.info(
            "The Digital Twin compares the actual operating "
            "condition with the expected condition derived "
            "from normal historical observations. The "
            "deviation indicates how far the current "
            "operation differs from the expected system "
            "behaviour."
        )

    # ==================================================
    # STOP DIGITAL TWIN PAGE
    # ==================================================



    # ==================================================
    # OVERALL MONITORING PAGE
    # ==================================================

# The Overall Monitoring page is the default page
# when the user selects "Overall Monitoring".

if page == "Overall Monitoring":

    st.title("📊 Overall Monitoring")

    st.markdown(
        "Overview of water injection operating conditions, "
        "anomaly detection, and system status."
    )

    st.divider()


    # ==================================================
    # OVERALL MONITORING DATA SOURCE
    # ==================================================

    # During near-real-time simulation, only the observations
    # that have been released by the simulation are displayed.
    #
    # During normal mode, the complete selected date range
    # is displayed.

    if live_mode:

        overview_df = live_df.copy()

    else:

        overview_df = filtered_df.copy()


    # ==================================================
    # SUMMARY CALCULATIONS
    # ==================================================

    total_observations = len(
        overview_df
    )

    normal_observations = (
        overview_df["Anomaly"] == 1
    ).sum()

    total_anomalies = (
        overview_df["Anomaly"] == -1
    ).sum()


    # --------------------------------------------------
    # ANOMALY RATE
    # --------------------------------------------------

    if total_observations > 0:

        anomaly_percentage = (
            total_anomalies
            / total_observations
        ) * 100

    else:

        anomaly_percentage = 0


    # ==================================================
    # LATEST OPERATING CONDITION
    # ==================================================

    if total_observations > 0:

        latest = overview_df.iloc[-1]

        latest_rate = latest[
            "Injection_Rate"
        ]

        latest_pressure = latest[
            "Injection_Pressure"
        ]

        latest_time = latest[
            "Time"
        ]

        latest_anomaly = latest[
            "Anomaly"
        ]


        # --------------------------------------------------
        # SYSTEM STATUS
        # --------------------------------------------------

        if latest_anomaly == -1:

            system_status = "Anomaly"

        else:

            system_status = "Normal"


    else:

        latest_rate = 0

        latest_pressure = 0

        latest_time = None

        system_status = "No Data"


    # ==================================================
    # SYSTEM OVERVIEW
    # ==================================================

    st.subheader("System Overview")


    col1, col2, col3, col4, col5 = st.columns(5)


    # --------------------------------------------------
    # TOTAL OBSERVATIONS
    # --------------------------------------------------

    with col1:

        st.metric(
            "Total Observations",
            f"{total_observations:,}"
        )


    # --------------------------------------------------
    # NORMAL OBSERVATIONS
    # --------------------------------------------------

    with col2:

        st.metric(
            "Normal",
            f"{normal_observations:,}"
        )


    # --------------------------------------------------
    # ANOMALIES
    # --------------------------------------------------

    with col3:

        st.metric(
            "Anomalies",
            f"{total_anomalies:,}"
        )


    # --------------------------------------------------
    # ANOMALY RATE
    # --------------------------------------------------

    with col4:

        st.metric(
            "Anomaly Rate",
            f"{anomaly_percentage:.2f}%"
        )


    # --------------------------------------------------
    # LATEST SYSTEM STATUS
    # --------------------------------------------------

    with col5:

        st.metric(
            "Latest Status",
            system_status
        )


    st.divider()


    # ==================================================
    # LATEST OPERATING CONDITION
    # ==================================================

    st.subheader(
        "Latest Recorded Operating Condition"
    )


    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------
    # LATEST INJECTION RATE
    # --------------------------------------------------

    with col1:

        st.metric(
            "Injection Rate",
            f"{latest_rate:.2f}"
        )


    # --------------------------------------------------
    # LATEST INJECTION PRESSURE
    # --------------------------------------------------

    with col2:

        st.metric(
            "Injection Pressure",
            f"{latest_pressure:.2f}"
        )


    # --------------------------------------------------
    # LATEST OBSERVATION TIME
    # --------------------------------------------------

    with col3:

        if latest_time is not None:

            st.metric(
                "Latest Observation",
                latest_time.strftime(
                    "%Y-%m-%d %H:%M"
                )
            )

        else:

            st.metric(
                "Latest Observation",
                "No Data"
            )


    st.divider()


    # ==================================================
    # INJECTION RATE OVER TIME
    # ==================================================

    st.subheader(
        "Injection Rate Over Time"
    )


    fig_rate = go.Figure()


    # --------------------------------------------------
    # INJECTION RATE TREND
    # --------------------------------------------------

    fig_rate.add_trace(
        go.Scatter(
            x=overview_df["Time"],
            y=overview_df["Injection_Rate"],
            mode="lines",
            name="Injection Rate"
        )
    )


    # --------------------------------------------------
    # HIGHLIGHT ANOMALIES
    # --------------------------------------------------

    anomaly_df = overview_df[
        overview_df["Anomaly"] == -1
    ]


    fig_rate.add_trace(
        go.Scatter(
            x=anomaly_df["Time"],
            y=anomaly_df["Injection_Rate"],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(
                size=9,
                symbol="circle"
            )
        )
    )


    # --------------------------------------------------
    # RATE CHART LAYOUT
    # --------------------------------------------------

    fig_rate.update_layout(
        xaxis_title="Time",
        yaxis_title="Injection Rate",
        hovermode="x unified",
        height=450
    )


    st.plotly_chart(
        fig_rate,
        use_container_width=True
    )


    st.divider()


    # ==================================================
    # INJECTION PRESSURE OVER TIME
    # ==================================================

    st.subheader(
        "Injection Pressure Over Time"
    )


    fig_pressure = go.Figure()


    # --------------------------------------------------
    # INJECTION PRESSURE TREND
    # --------------------------------------------------

    fig_pressure.add_trace(
        go.Scatter(
            x=overview_df["Time"],
            y=overview_df["Injection_Pressure"],
            mode="lines",
            name="Injection Pressure"
        )
    )


    # --------------------------------------------------
    # HIGHLIGHT PRESSURE ANOMALIES
    # --------------------------------------------------

    anomaly_pressure_df = overview_df[
        overview_df["Anomaly"] == -1
    ]


    fig_pressure.add_trace(
        go.Scatter(
            x=anomaly_pressure_df["Time"],
            y=anomaly_pressure_df[
                "Injection_Pressure"
            ],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(
                size=9,
                symbol="circle"
            )
        )
    )


    # --------------------------------------------------
    # PRESSURE CHART LAYOUT
    # --------------------------------------------------

    fig_pressure.update_layout(
        xaxis_title="Time",
        yaxis_title="Injection Pressure",
        hovermode="x unified",
        height=450
    )


    st.plotly_chart(
        fig_pressure,
        use_container_width=True
    )


    # ==================================================
    # ANOMALY SUMMARY
    # ==================================================

    st.divider()

    st.subheader(
        "Anomaly Summary"
    )


    if total_anomalies > 0:

        st.warning(
            f"{total_anomalies} anomalous observations "
            f"detected within the selected period."
        )


        # --------------------------------------------------
        # MOST RECENT ANOMALY
        # --------------------------------------------------

        latest_anomaly_record = anomaly_df.iloc[-1]


        st.markdown(
            "**Most Recent Anomaly**"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.write(
                f"**Time:** "
                f"{latest_anomaly_record['Time']}"
            )


        with col2:

            st.write(
                f"**Injection Rate:** "
                f"{latest_anomaly_record['Injection_Rate']:.2f}"
            )


        with col3:

            st.write(
                f"**Injection Pressure:** "
                f"{latest_anomaly_record['Injection_Pressure']:.2f}"
            )


        # --------------------------------------------------
        # LATEST ANOMALY NOTIFICATION
        # --------------------------------------------------

        st.markdown(
            f"**Alert:** "
            f"{latest_anomaly_record['Notification']}"
        )


        st.info(
            f"**Suggested Action:** "
            f"{latest_anomaly_record['Suggested_Action']}"
        )


    else:

        st.success(
            "No anomalous observations detected "
            "within the selected period."
        )


    # ==================================================
    # LIVE MODE INFORMATION
    # ==================================================

    if live_mode:

        st.divider()

        st.caption(
            "🔄 Near-real-time simulation is active. "
            "The dashboard is progressively displaying "
            "observations from the selected period."
        )