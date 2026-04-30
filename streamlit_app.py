from pathlib import Path

import pandas as pd
import streamlit as st

from src.api.main import MODELS, SensorInput, _predict_core, health_check


st.set_page_config(page_title="MachineGuard+ GUI", page_icon=":gear:", layout="wide")
st.title("MachineGuard+")
st.caption("Interactive predictive maintenance dashboard")

ROOT_DIR = Path(__file__).resolve().parent

RISK_PRESETS = {
    "Normal operation": {
        "machine_id": "M001",
        "machine_type": "L",
        "machine_age_bin": "Young",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 80,
    },
    "Watch closely": {
        "machine_id": "M052",
        "machine_type": "M",
        "machine_age_bin": "Mid",
        "air_temperature": 300.2,
        "process_temperature": 313.4,
        "rotational_speed": 1480,
        "torque": 49.2,
        "tool_wear": 160,
    },
    "Likely failure": {
        "machine_id": "M911",
        "machine_type": "H",
        "machine_age_bin": "Old",
        "air_temperature": 305.1,
        "process_temperature": 322.8,
        "rotational_speed": 1250,
        "torque": 61.6,
        "tool_wear": 245,
    },
}

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Normal operation"
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


def discover_visualizations() -> list[Path]:
    candidates: list[Path] = []
    search_dirs = [
        ROOT_DIR,
        ROOT_DIR / "outputs",
        ROOT_DIR / "outputs" / "figures",
        ROOT_DIR / "notebooks",
    ]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            candidates.extend(search_dir.rglob(ext))
    unique_sorted = sorted(set(candidates), key=lambda p: p.name.lower())
    return unique_sorted


def run_single_prediction(payload: SensorInput) -> dict:
    (
        fail_prob,
        failure_type,
        health_regime,
        predicted_tool_wear,
        urgency_level,
        triggered_rules,
        recommendations,
    ) = _predict_core(payload)
    return {
        "machine_id": payload.machine_id,
        "failure_probability": float(fail_prob),
        "failure_type": failure_type,
        "health_regime": health_regime,
        "predicted_tool_wear": predicted_tool_wear,
        "urgency_level": urgency_level,
        "triggered_rules": triggered_rules or [],
        "recommendations": recommendations or [],
    }


with st.sidebar:
    st.subheader("Quick Start")
    selected_preset = st.selectbox("Scenario preset", list(RISK_PRESETS.keys()), key="selected_preset")
    preset_values = RISK_PRESETS[selected_preset]
    st.caption("Use presets to simulate normal vs risky machine behavior.")

    if st.button("Show API/Model status", use_container_width=True):
        status = health_check()
        st.success(f"Service: {status['status']}")
        st.write("Model version:", status["model_version"])
        st.write("Loaded artefacts:", ", ".join(status["loaded_artefacts"]) or "None")

    with st.expander("Loaded artefacts", expanded=False):
        for artefact_name in sorted(MODELS.keys()):
            st.write(f"- {artefact_name}")

tabs = st.tabs(["Live Prediction", "Time Series", "Visualizations", "Project Flow"])

with tabs[0]:
    st.markdown("### Sensor Input")
    left_col, right_col = st.columns(2)
    with st.form("predict_form", clear_on_submit=False):
        with left_col:
            machine_id = st.text_input("Machine ID", value=preset_values["machine_id"])
            machine_type = st.selectbox(
                "Machine Type",
                ["L", "M", "H"],
                index=["L", "M", "H"].index(preset_values["machine_type"]),
            )
            machine_age_bin = st.selectbox(
                "Machine Age Bin",
                ["Young", "Mid", "Old"],
                index=["Young", "Mid", "Old"].index(preset_values["machine_age_bin"]),
            )
            air_temperature = st.slider("Air Temperature (K)", 280.0, 340.0, float(preset_values["air_temperature"]), 0.1)
        with right_col:
            process_temperature = st.slider(
                "Process Temperature (K)", 290.0, 360.0, float(preset_values["process_temperature"]), 0.1
            )
            rotational_speed = st.slider("Rotational Speed (rpm)", 200, 3500, int(preset_values["rotational_speed"]), 1)
            torque = st.slider("Torque (Nm)", 1.0, 120.0, float(preset_values["torque"]), 0.1)
            tool_wear = st.slider("Tool Wear (min)", 0, 300, int(preset_values["tool_wear"]), 1)
        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if submitted:
        try:
            payload = SensorInput(
                machine_id=machine_id,
                machine_type=machine_type,
                machine_age_bin=machine_age_bin,
                air_temperature=float(air_temperature),
                process_temperature=float(process_temperature),
                rotational_speed=int(rotational_speed),
                torque=float(torque),
                tool_wear=int(tool_wear),
            )
            result = run_single_prediction(payload)
            st.session_state.prediction_history.append(result)
            st.session_state.prediction_history = st.session_state.prediction_history[-25:]

            st.markdown("### Prediction Dashboard")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Failure Probability", f"{result['failure_probability']:.2%}")
            kpi2.metric("Failure Type", result["failure_type"])
            kpi3.metric("Health Regime", result["health_regime"])
            kpi4.metric("Urgency", result["urgency_level"] or "N/A")

            st.progress(
                min(max(float(result["failure_probability"]), 0.0), 1.0),
                text=f"Risk score: {result['failure_probability']:.2%}",
            )

            chart_df = pd.DataFrame(
                {
                    "sensor": ["Air Temp", "Process Temp", "Rot. Speed", "Torque", "Tool Wear"],
                    "value": [
                        float(air_temperature),
                        float(process_temperature),
                        float(rotational_speed),
                        float(torque),
                        float(tool_wear),
                    ],
                }
            )
            st.bar_chart(chart_df, x="sensor", y="value")

            st.markdown("### Insights")
            if result["predicted_tool_wear"] is not None:
                st.info(f"Predicted future tool wear: **{result['predicted_tool_wear']} min**")
            else:
                st.info("Tool wear projection skipped because failure risk is below threshold.")

            rules_col, recs_col = st.columns(2)
            with rules_col:
                st.markdown("#### Triggered Rules")
                if result["triggered_rules"]:
                    for rule in result["triggered_rules"]:
                        st.write(f"- {rule}")
                else:
                    st.write("No rules triggered.")

            with recs_col:
                st.markdown("#### Recommendations")
                if result["recommendations"]:
                    for idx, rec in enumerate(result["recommendations"], start=1):
                        with st.expander(f"Recommendation #{idx}", expanded=(idx == 1)):
                            st.json(rec)
                else:
                    st.write("No recommendations found.")
        except Exception as exc:  # pragma: no cover
            st.error(f"Prediction failed: {exc}")

    st.markdown("### Recent Predictions")
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(
            history_df[
                ["machine_id", "failure_probability", "failure_type", "health_regime", "predicted_tool_wear", "urgency_level"]
            ],
            use_container_width=True,
        )
        st.line_chart(history_df["failure_probability"], height=220)
    else:
        st.write("No predictions yet. Run one to populate history.")

with tabs[1]:
    st.markdown("### Time Series Step")
    st.caption("Upload a CSV sequence to simulate the `/predict/timeseries` project step.")

    sample_cols = [
        "machine_id",
        "machine_type",
        "machine_age_bin",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ]
    with st.expander("Expected CSV columns"):
        st.code(", ".join(sample_cols))

    uploaded = st.file_uploader("Upload time-series CSV", type=["csv"], key="ts_uploader")
    if uploaded is not None:
        try:
            ts_df = pd.read_csv(uploaded)
            st.write(f"Rows loaded: {len(ts_df)}")
            st.dataframe(ts_df.head(10), use_container_width=True)

            required = [
                "machine_id",
                "machine_type",
                "air_temperature",
                "process_temperature",
                "rotational_speed",
                "torque",
                "tool_wear",
            ]
            missing = [c for c in required if c not in ts_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            elif ts_df.empty:
                st.error("CSV has no rows.")
            else:
                row_results = []
                for _, row in ts_df.iterrows():
                    payload = SensorInput(
                        machine_id=str(row["machine_id"]),
                        machine_type=str(row["machine_type"]),
                        machine_age_bin=str(row.get("machine_age_bin", "Mid")),
                        air_temperature=float(row["air_temperature"]),
                        process_temperature=float(row["process_temperature"]),
                        rotational_speed=int(row["rotational_speed"]),
                        torque=float(row["torque"]),
                        tool_wear=int(row["tool_wear"]),
                    )
                    result = run_single_prediction(payload)
                    row_results.append(result)

                results_df = pd.DataFrame(row_results)
                st.markdown("#### Time Series Output")
                final = results_df.iloc[-1]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows Processed", len(results_df))
                m2.metric("Final Failure Probability", f"{float(final['failure_probability']):.2%}")
                m3.metric("Final Failure Type", str(final["failure_type"]))
                m4.metric("Final Urgency", str(final["urgency_level"]))

                plot_df = pd.DataFrame(
                    {
                        "step": list(range(1, len(results_df) + 1)),
                        "failure_probability": results_df["failure_probability"].astype(float).tolist(),
                        "predicted_tool_wear": pd.to_numeric(results_df["predicted_tool_wear"], errors="coerce").tolist(),
                        "observed_tool_wear": pd.to_numeric(ts_df["tool_wear"], errors="coerce").tolist(),
                    }
                )
                st.line_chart(plot_df.set_index("step")[["failure_probability"]], height=220)
                st.line_chart(plot_df.set_index("step")[["observed_tool_wear", "predicted_tool_wear"]], height=260)
                st.dataframe(results_df, use_container_width=True)
        except Exception as exc:  # pragma: no cover
            st.error(f"Time series processing failed: {exc}")

with tabs[2]:
    st.markdown("### Saved Visualizations Gallery")
    image_paths = discover_visualizations()
    if not image_paths:
        st.info("No saved visualizations found yet. Generate notebook plots first.")
    else:
        st.caption(f"Found {len(image_paths)} visualization files.")
        cols = st.columns(2)
        for idx, img_path in enumerate(image_paths):
            with cols[idx % 2]:
                st.image(str(img_path), caption=str(img_path.relative_to(ROOT_DIR)))

with tabs[3]:
    st.markdown("### Project Time Series Step (Info)")
    st.write(
        """
        This app follows your project pipeline:
        1. **Single Reading Inference**: engineer features, apply PCA/cluster, classify risk/failure type.
        2. **Regression Step**: estimate future tool wear when risk is above threshold.
        3. **Rules + Recommendations**: map alerts to explainable maintenance actions.
        4. **Time Series Step**: process a sequence of machine rows and track trend over steps.
        """
    )
    st.write("Use the **Time Series** tab to test that final step end-to-end with CSV input.")
