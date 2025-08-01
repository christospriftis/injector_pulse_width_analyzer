import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Constants
ENGINE_DISPLACEMENT_L = 1.6
ENGINE_DISPLACEMENT_M3 = ENGINE_DISPLACEMENT_L / 1000
R = 287  # J/kg·K (specific gas constant for air)
NUM_INJECTORS = 4
FUEL_DENSITY_G_PER_ML = 0.745
FUEL_PRESSURE_RATED_BAR = 4.0

def calculate_pulse_width(df, injector_flow_cc_min):
    injector_flow_g_per_s = (injector_flow_cc_min * FUEL_DENSITY_G_PER_ML) / 60
    AFR_stoich = 14.7
    df['AFR_actual'] = df['AFR_specified'] * AFR_stoich
    df['Fuel_mass_flow_g_s'] = df['MAF_gps'] / df['AFR_actual']
    df['Fuel_mass_per_injector_g_s'] = df['Fuel_mass_flow_g_s'] / NUM_INJECTORS
    df['Injections_per_sec'] = df['RPM'] / 120
    df['Fuel_mass_per_injection_g'] = df['Fuel_mass_per_injector_g_s'] / df['Injections_per_sec']
    df['IPW_calc_s'] = df['Fuel_mass_per_injection_g'] / injector_flow_g_per_s
    df['IPW_calc_ms'] = df['IPW_calc_s'] * 1000
    return df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_pred > 1e-6
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_pred[mask])) * 100

st.title("Fuel Injector Pulse Width Calculator")

injector_flow_cc_min = st.number_input(
    "Injector Flow Rate (cc/min)", min_value=50.0, max_value=1000.0, value=155.0, step=5.0
)

uploaded_file = st.file_uploader("Upload OBD2 log CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown("### Upload Field Mapping CSV File")
    mapping_file = st.file_uploader("Upload CSV with field mappings (original,new)", type=["csv"])

    if mapping_file is not None:
        mapping_df = pd.read_csv(mapping_file)
        column_mapping = dict(zip(mapping_df['original'], mapping_df['new']))
        df.rename(columns=column_mapping, inplace=True)

        required_fields = ['RPM', 'MAF_gps', 'AFR_specified', 'IPW_ms']
        missing_fields = [col for col in required_fields if col not in df.columns]
        if missing_fields:
            st.error(f"Missing required fields after mapping: {', '.join(missing_fields)}")
        else:
            df = df.dropna(subset=required_fields)
            df['RPM'] = pd.to_numeric(df['RPM'], errors='coerce')
            df['MAF_gps'] = pd.to_numeric(df['MAF_gps'], errors='coerce')
            df['AFR_specified'] = pd.to_numeric(df['AFR_specified'], errors='coerce')
            df['IPW_ms'] = pd.to_numeric(df['IPW_ms'], errors='coerce')
            df = df[(df['RPM'] > 0) & (df['MAF_gps'] > 0) & (df['AFR_specified'] > 0)]

            exclude_zero_pw = st.checkbox("Exclude rows where read injector pulse width = 0 ms", value=True)
            if exclude_zero_pw:
                df = df[df['IPW_ms'] > 0]

            df = calculate_pulse_width(df, injector_flow_cc_min)
            df['PW_diff_percent'] = ((df['IPW_ms'] - df['IPW_calc_ms']) / df['IPW_calc_ms']) * 100

            trial_rates = np.arange(0.7, 1.31, 0.05)
            results = []
            for mult in trial_rates:
                trial_flow = injector_flow_cc_min * mult
                df_trial = calculate_pulse_width(df.copy(), trial_flow)
                mape = mean_absolute_percentage_error(df_trial['IPW_ms'], df_trial['IPW_calc_ms'])
                results.append({"Injector Flow Rate (cc/min)": trial_flow, "MAPE (%)": mape})

            df_results = pd.DataFrame(results)
            best_row = df_results.loc[df_results['MAPE (%)'].idxmin()]
            best_flow = best_row['Injector Flow Rate (cc/min)']
            best_mape = best_row['MAPE (%)']

            st.markdown(f"### Best Injector Flow Rate Match")
            st.write(f"**{best_flow:.2f} cc/min** gives the lowest mean absolute percentage error of **{best_mape:.2f}%**.")

            st.markdown("### MAPE vs Injector Flow Rate")
            fig_mape = px.bar(
                df_results,
                x="Injector Flow Rate (cc/min)",
                y="MAPE (%)",
                text="MAPE (%)",
                title="MAPE Between Read and Calculated Pulse Width vs Injector Flow Rate"
            )
            fig_mape.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_mape.update_layout(width=1200)
            st.plotly_chart(fig_mape, use_container_width=False)

            st.markdown(f"### Injector Pulse Width: Read vs Calculated (Injector Flow Rate = {injector_flow_cc_min:.2f} cc/min)")
            fig1 = px.line(df, y=['IPW_ms', 'IPW_calc_ms'], title=f"Pulse Width Comparison")
            fig1.update_layout(width=1200, hovermode='x unified')
            st.plotly_chart(fig1, use_container_width=False)

            st.markdown(f"### Percentage Difference with Trendline (Injector Flow Rate = {injector_flow_cc_min:.2f} cc/min)")
            df_diff = df.reset_index()
            fig2 = px.scatter(
                df_diff,
                x='index',
                y='PW_diff_percent',
                trendline="ols",
                opacity=0.6,
                title="Pulse Width % Difference with Trendline"
            )
            fig2.update_traces(marker=dict(color='blue'))
            fig2.data[-1].line.color = 'red'  # set trendline color
            fig2.update_layout(width=1200, hovermode='x unified')
            st.plotly_chart(fig2, use_container_width=False)

            st.markdown("### Box Plot of Pulse Width % Difference by RPM Bins")
            rpm_bins = np.arange(0, df['RPM'].max() + 500, 500)
            df['RPM_bin'] = pd.cut(df['RPM'], bins=rpm_bins)
            df['RPM_bin_upper'] = df['RPM_bin'].apply(lambda x: int(x.right) if pd.notna(x) else np.nan)

            col1, col2 = st.columns([2, 1])
            with col1:
                fig_box_rpm = px.box(df, x='RPM_bin_upper', y='PW_diff_percent', points='outliers',
                                     labels={'RPM_bin_upper': 'RPM (Upper Bin Limit)'})
                fig_box_rpm.update_layout(width=800, title="% Difference Grouped by RPM")
                st.plotly_chart(fig_box_rpm, use_container_width=True)

            with col2:
                rpm_counts = df['RPM_bin_upper'].value_counts().sort_index()
                fig_count_rpm = px.bar(x=rpm_counts.index, y=rpm_counts.values,
                                       labels={'x': 'RPM (Upper Bin Limit)', 'y': 'Count'},
                                       title="Sample Count per RPM Bin")
                fig_count_rpm.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_count_rpm, use_container_width=True)

            if 'MAP' in df.columns:
                df['MAP'] = pd.to_numeric(df['MAP'], errors='coerce')
                df = df[df['MAP'].notna()]
                map_bins = np.arange(0, df['MAP'].max() + 50, 50)
                df['MAP_bin'] = pd.cut(df['MAP'], bins=map_bins)
                df['MAP_bin_upper'] = df['MAP_bin'].apply(lambda x: int(x.right) if pd.notna(x) else np.nan)

                st.markdown("### Box Plot of Pulse Width % Difference by MAP Bins (50 kPa intervals)")
                col3, col4 = st.columns([2, 1])
                with col3:
                    fig_box_map = px.box(df, x='MAP_bin_upper', y='PW_diff_percent', points='outliers',
                                         labels={'MAP_bin_upper': 'MAP (Upper Bin Limit)'})
                    fig_box_map.update_layout(width=800, title="% Difference Grouped by MAP")
                    st.plotly_chart(fig_box_map, use_container_width=True)

                with col4:
                    map_counts = df['MAP_bin_upper'].value_counts().sort_index()
                    fig_count_map = px.bar(x=map_counts.index, y=map_counts.values,
                                           labels={'x': 'MAP (Upper Bin Limit)', 'y': 'Count'},
                                           title="Sample Count per MAP Bin")
                    fig_count_map.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_count_map, use_container_width=True)
    else:
        st.warning("Please upload a field mapping CSV file to proceed.")
else:
    st.info("Please upload a CSV file to proceed.")
