import streamlit as st
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nurse Assigner", layout="wide")
st.title("🏥 Nurse Assignment & Acuity Calculator")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
w_continuity = st.sidebar.slider("Prioritize Continuity", 0, 100, 50)
w_handoffs = st.sidebar.slider("Minimize Handoffs", 0, 100, 40)
w_acuity = st.sidebar.slider("Balance Workload", 0, 100, 25)
w_charge_pref = st.sidebar.slider("Charge Protection", 0, 100, 100)

st.sidebar.markdown("---")
st.sidebar.header("🛑 Limits")
limit_titratable = st.sidebar.number_input("Max Titratable Drips", 1, 5, 2)
limit_insulin = st.sidebar.number_input("Max Insulin", 1, 2, 1)
limit_restraints = st.sidebar.number_input("Max Restraints", 1, 3, 2)

# --- ACUITY CALCULATOR LOGIC (IMCU TOOL) ---
def calculate_acuity_score(row):
    """
    Calculates Acuity (1-4) based on IMCU characteristics.
    Logic: The Patient is assigned the HIGHEST level found across categories.
    """
    if (row['O2_Device'] == 0) and (row['Med_Mgt'] == 0):
        return 3 

    current_max_level = 1
    
    o2_str = str(row['O2_Device']).lower()
    med_str = str(row['Med_Mgt']).lower()
    
    # --- LEVEL 4 (HIGH RISK) ---
    if 'bipap' in o2_str or 'hiflow' in o2_str or 'high' in o2_str: return 4
    if row['Restraints'] == 1: return 4
    if row['Insulin_Gtt'] == 1: return 4
    if row['Titratable_Gtt'] == 1: return 4 
        
    # --- LEVEL 3 (COMPLEX) ---
    if 'mid' in o2_str or 'venti' in o2_str: current_max_level = max(current_max_level, 3)
    if row['Sitter'] == 1: current_max_level = max(current_max_level, 3)
    if row['Heparin_NonTher'] == 1: current_max_level = max(current_max_level, 3)
    if 'high' in med_str: current_max_level = max(current_max_level, 3)
    if row['DI_Score'] > 60: current_max_level = max(current_max_level, 3)

    # --- LEVEL 2 (MODERATE) ---
    if 'nc' in o2_str or 'nasal' in o2_str: current_max_level = max(current_max_level, 2)
    if row['Heparin_Ther'] == 1: current_max_level = max(current_max_level, 2)
    if row['CiWA'] == 1: current_max_level = max(current_max_level, 2)
    if 'med' in med_str: current_max_level = max(current_max_level, 2)
    if row['DI_Score'] > 35: current_max_level = max(current_max_level, 2)

    return current_max_level

# --- DATA PREP ---
def preprocess_data(df):
    df.columns = [str(c).strip() for c in df.columns]
    
    col_map = {
        'Nurse Name': 'Nurse Name', 'Role': 'Role', 'Max_Patients': 'Max_Patients',
        'Room': 'Room', 
        'Current_Nurse': 'Current_Nurse', 'Nurse_24hrs_Ago': 'Nurse_24hrs_Ago',
        'Titratable_Gtt': 'Titratable_Gtt', 'Insulin_Gtt': 'Insulin_Gtt', 
        'Isolation': 'Isolation', 'CiWA': 'CiWA', 'Total_Care': 'Total_Care',
        'Discharge_Planned': 'Discharge_Planned', 'Transfer_Planned': 'Transfer_Planned',
        'New_Patient': 'New_Patient', 'Room Empty': 'Room Empty',
        # MATCHING NEW INPUT FILE HEADERS (Including Typos in Source)
        'Heparin_Gtt (Therapuetic)': 'Heparin_Ther',
        'Heparin_Gtt (Non-therapuetic)': 'Heparin_NonTher',
        'Supplmental_O2': 'O2_Device',
        'Med_Management': 'Med_Mgt',
        'Restraints': 'Restraints',
        'Sitter': 'Sitter',
        'Rapid_Response': 'Rapid_Response',
        'DI_Score': 'DI_Score',
        # OVERRIDES
        'Force_Assign': 'Force_Assign',
        'Avoid_Nurse': 'Avoid_Nurse'
    }
    
    clean = pd.DataFrame()
    for excel, internal in col_map.items():
        match = next((c for c in df.columns if excel.lower() in c.lower()), None)
        if match:
            clean[internal] = df[match]
        else:
            clean[internal] = 0 if internal not in ['O2_Device', 'Med_Mgt', 'Force_Assign', 'Avoid_Nurse', 'Nurse Name', 'Current_Nurse'] else None

    # Binary Cleanup
    binary_cols = ['Titratable_Gtt', 'Insulin_Gtt', 'Isolation', 'CiWA', 'Total_Care', 
                   'Discharge_Planned', 'Transfer_Planned', 'New_Patient', 'Room Empty',
                   'Heparin_Ther', 'Heparin_NonTher', 'Restraints', 'Sitter', 'Rapid_Response']
    
    for c in binary_cols:
        clean[c] = clean[c].astype(str).apply(lambda x: 1 if x.strip().lower() in ['yes', 'y', '1', 'true'] else 0)

    # Numeric Cleanup
    clean['DI_Score'] = pd.to_numeric(clean['DI_Score'], errors='coerce').fillna(0)
    
    # Handle Ghost Data (Empty/New)
    mask_reset = (clean['Room Empty'] == 1) | (clean['New_Patient'] == 1)
    
    # CALCULATE ACUITY
    clean['Calculated_Acuity'] = clean.apply(calculate_acuity_score, axis=1)
    
    if mask_reset.any():
        clean.loc[mask_reset, 'Calculated_Acuity'] = 3
        cols_to_wipe = ['Titratable_Gtt', 'Insulin_Gtt', 'Isolation', 'CiWA', 'Heparin_Ther', 'Restraints', 'DI_Score', 'Sitter', 'Rapid_Response']
        clean.loc[mask_reset, cols_to_wipe] = 0

    # CALCULATE WORKLOAD
    clean['Workload_Score'] = (
        clean['Calculated_Acuity'] + 
        (clean['DI_Score'] / 20.0) +
        (clean['Restraints'] * 0.5) + 
        (clean['Sitter'] * 0.5) +
        (clean['Rapid_Response'] * 2.0)
    )
    
    # Fill Missing Strings
    clean['Current_Nurse'] = clean['Current_Nurse'].fillna('Unknown')
    for c in ['Force_Assign', 'Avoid_Nurse']:
        clean[c] = clean[c].fillna('Unknown').astype(str)
            
    return clean

# --- APP ---
uploaded = st.file_uploader("Upload Input (.xlsx)", type=['xlsx', 'xlsm'])

if uploaded:
    raw = pd.read_excel(uploaded)
    df = preprocess_data(raw)
    
    with st.expander("📋 Data Preview & Overrides"):
        st.dataframe(df[['Room', 'Calculated_Acuity', 'Force_Assign', 'Avoid_Nurse']])

    charges = [n for n in df['Current_Nurse'].unique() if str(n).lower() not in ['nan','0','unknown','']]
    charges.sort()
    off_going_charge = st.selectbox("Off-Going Charge:", options=charges, index=0 if charges else None)

    if st.button("🚀 Run Scheduler"):
        with st.spinner("Optimizing..."):
            # 1. SETUP & VALIDATION (STRICT NURSE FILTER)
            nurses = df[['Nurse Name', 'Role', 'Max_Patients']].copy()
            # Remove NaNs and Empty Strings
            nurses = nurses.dropna(subset=['Nurse Name'])
            nurses = nurses[nurses['Nurse Name'].astype(str).str.strip() != '']
            nurses = nurses[~nurses['Nurse Name'].astype(str).str.lower().isin(['nan', 'unknown', '0'])]
            
            nurses = nurses.drop_duplicates()
            nurses['Max_Patients'] = pd.to_numeric(nurses['Max_Patients'], errors='coerce').fillna(4).astype(int)
            nurses['Role'] = nurses['Role'].fillna('RN').astype(str)
            
            patients = df.dropna(subset=['Room']).copy()
            patients['Room'] = patients['Room'].astype(str)
            off_going_nurses = [n for n in patients['Current_Nurse'].unique() if str(n).lower() not in ['nan', 'unknown', '0']]

            if len(nurses) == 0:
                st.error("❌ No nurses found! Please check columns A-D.")
                st.stop()

            # 2. Model
            model = cp_model.CpModel()
            x = {}
            for n in nurses.index:
                for p in patients.index:
                    x[n, p] = model.NewBoolVar(f'x_{n}_{p}')

            # 3. Constraints
            for p in patients.index:
                model.Add(sum(x[n, p] for n in nurses.index) == 1)

            # *** MANUAL OVERRIDES (FORCE/AVOID) ***
            for p, pat in patients.iterrows():
                # Force Assign
                force = str(pat['Force_Assign']).strip().lower()
                if force not in ['0', 'unknown', 'nan', '', 'none']:
                    target_nurse = next((n for n in nurses.index if force in str(nurses.loc[n, 'Nurse Name']).lower()), None)
                    if target_nurse is not None:
                        model.Add(x[target_nurse, p] == 1)
                    else:
                        st.warning(f"⚠️ Could not force Room {pat['Room']} to '{pat['Force_Assign']}' - Nurse not found in roster.")

                # Avoid Nurse
                avoid = str(pat['Avoid_Nurse']).strip().lower()
                if avoid not in ['0', 'unknown', 'nan', '', 'none']:
                    target_nurse = next((n for n in nurses.index if avoid in str(nurses.loc[n, 'Nurse Name']).lower()), None)
                    if target_nurse is not None:
                        model.Add(x[target_nurse, p] == 0)

            for n, nurse in nurses.iterrows():
                model.Add(sum(x[n, p] for p in patients.index) <= int(nurse['Max_Patients']))
                
                # Clinical Sums
                t_titr = sum(x[n, p] * (patients.loc[p, 'Titratable_Gtt'] + patients.loc[p, 'Heparin_Ther']) for p in patients.index)
                t_ins = sum(x[n, p] * patients.loc[p, 'Insulin_Gtt'] for p in patients.index)
                t_iso = sum(x[n, p] * patients.loc[p, 'Isolation'] for p in patients.index)
                t_ciwa = sum(x[n, p] * patients.loc[p, 'CiWA'] for p in patients.index)
                t_empty = sum(x[n, p] * patients.loc[p, 'Room Empty'] for p in patients.index)
                t_rest = sum(x[n, p] * patients.loc[p, 'Restraints'] for p in patients.index)

                if str(nurse['Role']).lower() == 'charge':
                    model.Add(t_empty == 0)
                    model.Add(t_titr == 0)
                    model.Add(t_ins == 0)
                    model.Add(t_ciwa == 0)
                    model.Add(t_rest == 0)
                else:
                    model.Add(t_ins <= limit_insulin)
                    model.Add(t_titr + (10 * t_ins) <= 10)
                    model.Add(t_titr <= limit_titratable)
                    model.Add(t_ciwa <= 1)
                    model.Add(t_rest <= limit_restraints)
                
                model.Add(t_iso <= 2)

            # 4. Objectives
            objs = []
            
            # Handoffs
            for n in nurses.index:
                ho = model.NewIntVar(0, 10, f'ho_{n}')
                interactions = []
                for off in off_going_nurses:
                    p_idx = patients[patients['Current_Nurse'] == off].index
                    active = model.NewBoolVar(f'act_{n}_{off}')
                    model.Add(sum(x[n, p] for p in p_idx) > 0).OnlyEnforceIf(active)
                    model.Add(sum(x[n, p] for p in p_idx) == 0).OnlyEnforceIf(active.Not())
                    interactions.append(active)
                model.Add(ho == sum(interactions))
                objs.append(ho * -w_handoffs)

            # Workload Balancing
            nurse_loads = []
            for n in nurses.index:
                load = sum(x[n, p] * int(patients.loc[p, 'Workload_Score']*10) for p in patients.index)
                nurse_loads.append(load)
                
                # Empty Room Penalty
                n_empty = sum(x[n, p] * patients.loc[p, 'Room Empty'] for p in patients.index)
                has_empty = model.NewBoolVar(f'he_{n}')
                model.Add(n_empty > 0).OnlyEnforceIf(has_empty)
                model.Add(n_empty == 0).OnlyEnforceIf(has_empty.Not())
                
                pen = model.NewIntVar(0, 5000, f'pen_{n}')
                model.Add(pen == load).OnlyEnforceIf(has_empty)
                model.Add(pen == 0).OnlyEnforceIf(has_empty.Not())
                objs.append(pen * -50)

                # Patient Specifics
                for p, pat in patients.iterrows():
                    if str(nurses.loc[n, 'Role']).lower() == 'charge':
                        if pat['Current_Nurse'] == off_going_charge and pat['Discharge_Planned']==0:
                            objs.append(x[n, p] * w_charge_pref)
                        objs.append(x[n, p] * int(pat['Workload_Score']*10) * -5)
                    else:
                        if pat['New_Patient']==0 and pat['Room Empty']==0:
                            if nurses.loc[n, 'Nurse Name'] == pat['Nurse_24hrs_Ago']:
                                objs.append(x[n, p] * w_continuity)
            
            # Balance
            mn = model.NewIntVar(0, 5000, 'min')
            mx = model.NewIntVar(0, 5000, 'max')
            model.AddMaxEquality(mx, nurse_loads)
            model.AddMinEquality(mn, nurse_loads)
            objs.append((mx - mn) * -w_acuity)

            model.Maximize(sum(objs))
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 120.0
            status = solver.Solve(model)

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                st.success(f"✅ Assignments Generated!")
                
                res = []
                for n in nurses.index:
                    my_p = []
                    sources = set()
                    stats = {'acuity':0, 'work':0, 'iso':0, 'drips':0, 'rest':0, 'hep':0}
                    
                    for p in patients.index:
                        if solver.Value(x[n, p]):
                            lbl = str(patients.loc[p, 'Room'])
                            if patients.loc[p, 'Room Empty']: lbl += " (Empty)"
                            my_p.append(lbl)
                            
                            src = str(patients.loc[p, 'Current_Nurse'])
                            if src.lower() not in ['nan','0','unknown']: sources.add(src)
                            
                            stats['acuity'] += patients.loc[p, 'Calculated_Acuity']
                            stats['work'] += patients.loc[p, 'Workload_Score']
                            stats['iso'] += patients.loc[p, 'Isolation']
                            stats['rest'] += patients.loc[p, 'Restraints']
                            stats['drips'] += (patients.loc[p, 'Titratable_Gtt'] + patients.loc[p, 'Insulin_Gtt'])
                            stats['hep'] += patients.loc[p, 'Heparin_Ther']

                    res.append({
                        'Nurse': nurses.loc[n, 'Nurse Name'],
                        'Role': nurses.loc[n, 'Role'],
                        'Patients': ", ".join(my_p),
                        'Count': len(my_p),
                        'Calculated Acuity': stats['acuity'],
                        'Workload Index': round(stats['work'], 1),
                        'Handoffs': len(sources),
                        'Report From': ", ".join(sorted(sources)),
                        'Drips/Hep': f"{stats['drips']}/{stats['hep']}",
                        'Restr': stats['rest'],
                        'Iso': stats['iso']
                    })
                
                res_df = pd.DataFrame(res)
                
                st.markdown("---")
                st.subheader("📊 Manager Report")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Acuity Points", f"{patients['Calculated_Acuity'].sum()}")
                c2.metric("Restraints / Rapids", f"{patients['Restraints'].sum()} / {patients['Rapid_Response'].sum()}")
                c3.metric("Avg Workload/Nurse", f"{res_df['Workload Index'].mean():.1f}")
                
                st.dataframe(res_df)
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Results", csv, "Assignments.csv", "text/csv")
            else:
                st.error("No solution found. (If you used Force/Avoid, check if you assigned a nurse who is already maxed out!)")
