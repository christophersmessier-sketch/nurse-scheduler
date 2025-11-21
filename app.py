import streamlit as st
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nurse Assigner", layout="wide")
st.title("🏥 Nurse Assignment Auto-Scheduler")
st.markdown("""
**Instructions:**
1. Upload your standard **Excel (.xlsx)** file.
2. Select the **Off-Going Charge Nurse** from the dropdown.
3. Click **Generate Assignments**.
""")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Optimization Settings")

# Weights
w_continuity = st.sidebar.slider("Prioritize Continuity", 0, 100, 50)
w_handoffs = st.sidebar.slider("Minimize Handoffs", 0, 100, 40)
w_acuity = st.sidebar.slider("Balance Acuity", 0, 100, 15)
w_charge_pref = st.sidebar.slider("Charge Nurse Protection", 0, 100, 100)

# Hard Limits
st.sidebar.markdown("---")
st.sidebar.header("🛑 Safety Limits")
limit_titratable = st.sidebar.number_input("Max Titratable Drips", 1, 5, 2)
limit_insulin = st.sidebar.number_input("Max Insulin Drips", 1, 2, 1)
limit_isolation = st.sidebar.number_input("Max Isolation Patients", 1, 4, 2)

# --- DATA CLEANING FUNCTION (Replaces your Macro) ---
def preprocess_data(df):
    # 1. Standardize Column Names (Strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # 2. Identify Key Columns (handle potential naming variations)
    # Map your Excel headers to internal names
    col_map = {
        'Nurse Name': 'Nurse Name', 'Role': 'Role', 'Max_Patients': 'Max_Patients',
        'Room': 'Room', 'Acuity_Score': 'Acuity_Score',
        'Current_Nurse': 'Current_Nurse', 'Nurse_24hrs_Ago': 'Nurse_24hrs_Ago',
        'Titratable_Gtt': 'Titratable_Gtt', 'Insulin_Gtt': 'Insulin_Gtt',
        'Isolation': 'Isolation', 'CiWA': 'CiWA', 'Total_Care': 'Total_Care',
        'Discharge_Planned': 'Discharge_Planned', 'Transfer_Planned': 'Transfer_Planned',
        'New_Patient': 'New_Patient', 'Room Empty': 'Room Empty'
    }

    # Create a clean dataframe with standardized names
    clean_df = pd.DataFrame()
    for excel_name, internal_name in col_map.items():
        # Find the column in the uploaded file (case insensitive search)
        match = next((c for c in df.columns if c.lower() == excel_name.lower().strip()), None)
        if match:
            clean_df[internal_name] = df[match]
        else:
            clean_df[internal_name] = 0 # Default if missing

    # 3. CONVERT YES/NO to 1/0
    binary_cols = ['Titratable_Gtt', 'Insulin_Gtt', 'Isolation', 'CiWA', 'Total_Care',
                   'Discharge_Planned', 'Transfer_Planned', 'New_Patient', 'Room Empty']

    for col in binary_cols:
        clean_df[col] = clean_df[col].astype(str).apply(
            lambda x: 1 if x.strip().lower() in ['yes', 'y', 'true', '1'] else 0
        )

    # 4. HANDLE "GHOST DATA" (The Macro Logic)
    # If Room is Empty OR New Patient -> Wipe the clinical data
    mask_reset = (clean_df['Room Empty'] == 1) | (clean_df['New_Patient'] == 1)

    if mask_reset.any():
        st.toast(f"🧹 Automatically cleaned {mask_reset.sum()} Empty/New rooms.", icon="✨")

        # Reset Drips/Iso/CiWA to 0
        cols_to_zero = ['Titratable_Gtt', 'Insulin_Gtt', 'Isolation', 'CiWA', 'Total_Care']
        clean_df.loc[mask_reset, cols_to_zero] = 0

        # Set Acuity to Baseline (3) so they aren't treated as "Zero Work"
        clean_df.loc[mask_reset, 'Acuity_Score'] = 3

    # 5. Ensure Numeric Types for Math
    clean_df['Acuity_Score'] = pd.to_numeric(clean_df['Acuity_Score'], errors='coerce').fillna(1)

    # 6. Fill Missing Text
    clean_df['Nurse Name'] = clean_df['Nurse Name'].fillna('Unknown')
    clean_df['Current_Nurse'] = clean_df['Current_Nurse'].fillna('Unknown')

    return clean_df

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Input File (.xlsx)", type=['xlsx', 'xlsm'])

if uploaded_file is not None:
    try:
        raw_df = pd.read_excel(uploaded_file)
        df = preprocess_data(raw_df)

        st.write("### 📋 Preview (Cleaned Data)")
        st.dataframe(df.head())

        # --- OFF-GOING CHARGE SELECTOR ---
        # Get list of all nurses listed in "Current_Nurse" column
        potential_charges = df['Current_Nurse'].unique().tolist()
        # Filter out garbage values
        potential_charges = [n for n in potential_charges if str(n).lower() not in ['nan', '0', 'unknown', '']]
        potential_charges.sort()

        off_going_charge = st.selectbox(
            "Who was the **Off-Going Charge Nurse**?",
            options=potential_charges,
            index=0 if potential_charges else None,
            help="Select the nurse whose patients should be handed off to the incoming Charge Nurse."
        )

        # --- RUN BUTTON ---
        if st.button("🚀 Generate Assignments"):
            with st.spinner("Optimizing Schedule..."):

                # 1. SETUP NURSES
                nurses = df[['Nurse Name', 'Role', 'Max_Patients']].dropna(subset=['Nurse Name']).drop_duplicates()
                nurses['Max_Patients'] = pd.to_numeric(nurses['Max_Patients'], errors='coerce').fillna(4).astype(int)
                nurses['Role'] = nurses['Role'].fillna('RN').astype(str)

                # 2. SETUP PATIENTS
                # Filter out rows where Room is empty/NaN
                patients = df.dropna(subset=['Room']).copy()
                patients['Room'] = patients['Room'].astype(str)

                # Handoff Targets
                off_going_nurses = [n for n in patients['Current_Nurse'].unique() if str(n).lower() not in ['nan', 'unknown', '0']]

                # 3. MODEL BUILD
                model = cp_model.CpModel()
                assignments = {}

                # Variables
                for n_idx, nurse in nurses.iterrows():
                    for p_idx, patient in patients.iterrows():
                        assignments[(n_idx, p_idx)] = model.NewBoolVar(f'n{n_idx}_p{p_idx}')

                # Constraints
                # One nurse per patient
                for p_idx in patients.index:
                    model.Add(sum(assignments[(n_idx, p_idx)] for n_idx in nurses.index) == 1)

                # Max Patients
                for n_idx, nurse in nurses.iterrows():
                    model.Add(sum(assignments[(n_idx, p_idx)] for p_idx in patients.index) <= int(nurse['Max_Patients']))

                    # Calculate Nurse Totals
                    t_titr = sum(assignments[(n_idx, p_idx)] * patients.loc[p_idx, 'Titratable_Gtt'] for p_idx in patients.index)
                    t_ins = sum(assignments[(n_idx, p_idx)] * patients.loc[p_idx, 'Insulin_Gtt'] for p_idx in patients.index)
                    t_ciwa = sum(assignments[(n_idx, p_idx)] * patients.loc[p_idx, 'CiWA'] for p_idx in patients.index)
                    t_iso = sum(assignments[(n_idx, p_idx)] * patients.loc[p_idx, 'Isolation'] for p_idx in patients.index)
                    t_empty = sum(assignments[(n_idx, p_idx)] * patients.loc[p_idx, 'Room Empty'] for p_idx in patients.index)

                    # Role Logic
                    if str(nurse['Role']).lower() == 'charge':
                        model.Add(t_empty == 0) # No admits for charge
                        model.Add(t_titr == 0)
                        model.Add(t_ins == 0)
                        model.Add(t_ciwa == 0)
                    else:
                        model.Add(t_ins <= limit_insulin)
                        model.Add(t_titr + (10 * t_ins) <= 10) # Mutual exclusion
                        model.Add(t_titr <= limit_titratable)
                        model.Add(t_ciwa <= 1)

                    model.Add(t_iso <= limit_isolation)

                # Objectives
                objective_terms = []

                # A. Minimize Handoffs
                for n_idx, nurse in nurses.iterrows():
                    handoff_count = model.NewIntVar(0, len(off_going_nurses) + 1, f'ho_{n_idx}')
                    interactions = []
                    for off_nurse in off_going_nurses:
                        # Which patients belong to this off-nurse?
                        p_indices = patients[patients['Current_Nurse'] == off_nurse].index
                        has_interaction = model.NewBoolVar(f'int_{n_idx}_{off_nurse}')
                        assigned_count = sum(assignments[(n_idx, p)] for p in p_indices)

                        model.Add(assigned_count > 0).OnlyEnforceIf(has_interaction)
                        model.Add(assigned_count == 0).OnlyEnforceIf(has_interaction.Not())
                        interactions.append(has_interaction)

                    model.Add(handoff_count == sum(interactions))
                    objective_terms.append(handoff_count * -w_handoffs)

                # B. Linear Empty Room Penalty (Fast Logic)
                for n_idx, nurse in nurses.iterrows():
                    expr_empty = sum(assignments[(n_idx, p_idx)] * patients.loc[p_idx, 'Room Empty'] for p_idx in patients.index)
                    expr_acuity = sum(assignments[(n_idx, p_idx)] * patients.loc[p_idx, 'Acuity_Score'] for p_idx in patients.index)
                    expr_drips = sum(assignments[(n_idx, p_idx)] * (patients.loc[p_idx, 'Titratable_Gtt'] + patients.loc[p_idx, 'Insulin_Gtt']) for p_idx in patients.index)

                    has_empty = model.NewBoolVar(f'has_empty_{n_idx}')
                    model.Add(expr_empty > 0).OnlyEnforceIf(has_empty)
                    model.Add(expr_empty == 0).OnlyEnforceIf(has_empty.Not())

                    # If has empty room, Acuity becomes a penalty
                    pen_acuity = model.NewIntVar(0, 200, f'pa_{n_idx}')
                    model.Add(pen_acuity == expr_acuity).OnlyEnforceIf(has_empty)
                    model.Add(pen_acuity == 0).OnlyEnforceIf(has_empty.Not())

                    # If has empty room, Drips become a penalty
                    pen_drips = model.NewIntVar(0, 20, f'pd_{n_idx}')
                    model.Add(pen_drips == expr_drips).OnlyEnforceIf(has_empty)
                    model.Add(pen_drips == 0).OnlyEnforceIf(has_empty.Not())

                    objective_terms.append(pen_acuity * -50) # Hard penalty
                    objective_terms.append(pen_drips * -50)

                    # Patient-Specific Logic
                    for p_idx, patient in patients.iterrows():
                        if str(nurse['Role']).lower() == 'charge':
                            # Charge Logic
                            if patient['Current_Nurse'] == off_going_charge:
                                if patient['Discharge_Planned'] == 1 or patient['Transfer_Planned'] == 1:
                                    objective_terms.append(assignments[(n_idx, p_idx)] * -50)
                                else:
                                    objective_terms.append(assignments[(n_idx, p_idx)] * w_charge_pref)

                            # Penalize Charge Acuity/Iso
                            objective_terms.append(assignments[(n_idx, p_idx)] * int(patient['Acuity_Score']) * -5)
                            objective_terms.append(assignments[(n_idx, p_idx)] * int(patient['Isolation']) * -50)
                        else:
                            # Standard Logic
                            if patient['New_Patient'] == 0 and patient['Room Empty'] == 0:
                                if nurse['Nurse Name'] == patient['Nurse_24hrs_Ago']:
                                    objective_terms.append(assignments[(n_idx, p_idx)] * w_continuity)

                # C. Acuity Balance
                nurse_acuities = []
                for n_idx, nurse in nurses.iterrows():
                    total_acuity = sum(assignments[(n_idx, p_idx)] * int(patients.loc[p_idx, 'Acuity_Score']) for p_idx in patients.index)
                    nurse_acuities.append(total_acuity)

                min_a = model.NewIntVar(0, 200, 'min')
                max_a = model.NewIntVar(0, 200, 'max')
                model.AddMaxEquality(max_a, nurse_acuities)
                model.AddMinEquality(min_a, nurse_acuities)
                objective_terms.append((max_a - min_a) * -w_acuity)

                # Solve
                model.Maximize(sum(objective_terms))
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = 120.0
                status = solver.Solve(model)

                if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                    st.success(f"✅ Solution Found! (Score: {solver.ObjectiveValue()})")

                    # BUILD RESULTS TABLE
                    results = []
                    for n_idx, nurse in nurses.iterrows():
                        p_list = []
                        sources = set()
                        vals = {'acuity':0, 'titr':0, 'ins':0, 'iso':0, 'ciwa':0, 'empty':0}

                        for p_idx, patient in patients.iterrows():
                            if solver.Value(assignments[(n_idx, p_idx)]) == 1:
                                # Label
                                lbl = str(patient['Room'])
                                if patient['Room Empty']:
                                    lbl += " (Empty)"
                                    vals['empty'] += 1
                                p_list.append(lbl)

                                # Sources
                                src = str(patient['Current_Nurse'])
                                if src.lower() not in ['unknown', '0', 'nan']:
                                    sources.add(src)

                                # Stats
                                vals['acuity'] += patient['Acuity_Score']
                                vals['titr'] += patient['Titratable_Gtt']
                                vals['ins'] += patient['Insulin_Gtt']
                                vals['iso'] += patient['Isolation']
                                vals['ciwa'] += patient['CiWA']

                        results.append({
                            'Nurse': nurse['Nurse Name'],
                            'Role': nurse['Role'],
                            'Patients': ", ".join(p_list),
                            'Count': len(p_list),
                            'Total Acuity': vals['acuity'],
                            'Handoffs': len(sources),
                            'Report From': ", ".join(sorted(sources)),
                            'Drips (T/I)': f"{vals['titr']} / {vals['ins']}",
                            'CiWA': vals['ciwa'],
                            'Iso': vals['iso'],
                            'Empty': vals['empty']
                        })

                    final_df = pd.DataFrame(results)
                    st.dataframe(final_df)

                    # Download Button
                    csv = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Schedule (CSV)",
                        data=csv,
                        file_name="Final_Shift_Assignments.csv",
                        mime="text/csv",
                    )

                else:
                    st.error("❌ No solution found. The constraints are too tight (e.g., not enough nurses for the drips/isolation load).")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.warning("Please ensure you uploaded the Excel file, not a CSV.")
