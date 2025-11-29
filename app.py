import streamlit as st
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
from datetime import date
import io
import smtplib
from email.message import EmailMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nurse Assigner", layout="wide")
st.title("🏥 Nurse Assignment & Acuity Calculator")

# --- SIDEBAR ---
st.sidebar.header("📅 Shift Context")
assignment_date = st.sidebar.date_input("Assignment Date", date.today())
shift_type = st.sidebar.radio("Shift", ["Day", "Night"], horizontal=True)

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

# --- EMAIL FUNCTION ---
def send_email(recipient_input, file_buffer, filename):
    try:
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"]
        smtp_server = st.secrets["email"]["smtp_server"]
        smtp_port = st.secrets["email"]["smtp_port"]

        recipients = [email.strip() for email in recipient_input.split(',')]

        msg = EmailMessage()
        msg['Subject'] = f"Nurse Assignments - {assignment_date} ({shift_type})"
        msg['From'] = sender_email
        msg['To'] = ", ".join(recipients)
        msg.set_content(f"Attached is the Nurse Assignment Workbook for {assignment_date}.\n\nTabs Included:\n1. Nurse Summary\n2. Patient List\n3. Manager/Safety Huddle Report")

        file_data = file_buffer.getvalue()
        msg.add_attachment(file_data, maintype='application', subtype='xlsx', filename=filename)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)

# --- ACUITY CALCULATOR ---
def calculate_acuity_score(row):
    # 1. Get Self-Reported Score (The "Human Judgment")
    try:
        self_reported = int(row['Self_Reported_Acuity'])
    except:
        self_reported = 1 # Default if missing

    # 2. Calculate Data-Driven Score (The "Clinical Reality")
    calculated_level = 1
    
    o2_str = str(row['O2_Device']).lower()
    med_str = str(row['Med_Mgt']).lower()
    rapid_str = str(row['Rapid_Response']).lower()
    line_str = str(row['Central_Line']).lower()
    
    # --- LEVEL 4 (HIGH RISK) ---
    if 'bipap' in o2_str or 'hiflow' in o2_str or 'high' in o2_str: calculated_level = 4
    if 'new' in o2_str and 'trach' in o2_str: calculated_level = 4 
    if row['Restraints'] == 1: calculated_level = 4
    if 'code' in rapid_str: calculated_level = 4 
    if 'rapid' in rapid_str: calculated_level = 4 
    if row['Insulin_Gtt'] == 1: calculated_level = 4
    if row['Titratable_Gtt'] == 1: calculated_level = 4 
        
    # --- LEVEL 3 (COMPLEX) ---
    if calculated_level < 3:
        if 'mid' in o2_str or 'venti' in o2_str: calculated_level = 3
        if 'stable' in o2_str and 'trach' in o2_str: calculated_level = 3
        if row['Drains'] == 1: calculated_level = 3
        if row['Sitter'] == 1: calculated_level = 3
        if row['Heparin_NonTher'] == 1: calculated_level = 3
        if 'high' in med_str: calculated_level = 3
        if row['DI_Score'] > 60: calculated_level = 3
        if line_str not in ['0', 'nan', '', 'none']: calculated_level = 3

    # --- LEVEL 2 (MODERATE) ---
    if calculated_level < 2:
        if 'nc' in o2_str or 'nasal' in o2_str: calculated_level = 2
        if row['Foley'] == 1: calculated_level = 2
        if row['Heparin_Ther'] == 1: calculated_level = 2
        if row['CiWA'] == 1: calculated_level = 2
        if 'med' in med_str: calculated_level = 2
        if row['DI_Score'] > 35: calculated_level = 2

    # 3. Final Score is the HIGHER of the two
    # This prevents the script from downgrading a nurse's concern, 
    # but forces an upgrade if the nurse missed a drip/rapid.
    return max(self_reported, calculated_level)

# --- DATA PREP ---
def preprocess_data(df, target_date, shift):
    df.columns = [str(c).strip() for c in df.columns]
    
    col_map = {
        'Nurse Name': 'Nurse Name', 'Role': 'Role', 'Max_Patients': 'Max_Patients',
        'Room': 'Room', 
        'Current_Nurse': 'Current_Nurse', 'Nurse_24hrs_Ago': 'Nurse_24hrs_Ago',
        'Titratable_Gtt': 'Titratable_Gtt', 'Insulin_Gtt': 'Insulin_Gtt', 
        'Isolation': 'Isolation', 'CiWA': 'CiWA', 'Total_Care': 'Total_Care',
        'Discharge_Planned': 'Discharge_Planned', 'Transfer_Planned': 'Transfer_Planned',
        'New_Patient': 'New_Patient', 'Room Empty': 'Room Empty',
        'Heparin_Gtt': 'Heparin_Gtt', 
        'Heparin_Gtt (Therapuetic)': 'Heparin_Ther_Split', 
        'Heparin_Gtt (Non-therapuetic)': 'Heparin_NonTher_Split', 
        'Supplmental_O2': 'O2_Device',
        'Med_Management': 'Med_Mgt',
        'Restraints': 'Restraints',
        'Sitter': 'Sitter',
        'Rapid_Response/Code': 'Rapid_Response',
        'Foley': 'Foley',
        'Drains/Tubes': 'Drains',
        'Central_Line': 'Central_Line',
        'DI_Score': 'DI_Score',
        'Acuity_Score': 'Self_Reported_Acuity', # NEW: Column G
        'Force_Assign': 'Force_Assign', 'Avoid_Nurse': 'Avoid_Nurse'
    }
    
    clean = pd.DataFrame()
    for excel_header, internal_name in col_map.items():
        if excel_header in df.columns:
            clean[internal_name] = df[excel_header]
        else:
            match = next((c for c in df.columns if c.lower() == excel_header.lower()), None)
            clean[internal_name] = df[match] if match else 0

    # Heparin Logic
    if 'Heparin_Gtt' in clean.columns and not clean['Heparin_Gtt'].astype(str).str.contains('0').all():
        clean['Heparin_NonTher'] = clean['Heparin_Gtt'].astype(str).apply(lambda x: 1 if 'non' in x.lower() else 0)
        clean['Heparin_Ther'] = clean['Heparin_Gtt'].astype(str).apply(lambda x: 1 if 'ther' in x.lower() and 'non' not in x.lower() else 0)
    else:
        if 'Heparin_Ther_Split' in clean.columns:
            clean['Heparin_Ther'] = clean['Heparin_Ther_Split'].astype(str).apply(lambda x: 1 if x.lower() in ['yes','y','1'] else 0)
        else: clean['Heparin_Ther'] = 0
        if 'Heparin_NonTher_Split' in clean.columns:
            clean['Heparin_NonTher'] = clean['Heparin_NonTher_Split'].astype(str).apply(lambda x: 1 if x.lower() in ['yes','y','1'] else 0)
        else: clean['Heparin_NonTher'] = 0

    def clean_bool(val):
        s = str(val).strip().lower()
        if s in ['yes', 'y', '1', 'true', '1:1', 'vsc', 'safety']: return 1
        return 0

    binary_cols = ['Titratable_Gtt', 'Insulin_Gtt', 'Isolation', 'CiWA', 'Total_Care', 
                   'Transfer_Planned', 'New_Patient', 'Room Empty',
                   'Restraints', 'Sitter', 'Foley', 'Drains']
    
    for c in binary_cols:
        clean[c] = clean[c].apply(clean_bool)

    clean['Rapid_Response'] = clean['Rapid_Response'].fillna(0).astype(str)
    clean['Has_Rapid'] = clean['Rapid_Response'].apply(lambda x: 1 if x.lower() in ['yes', '1', 'rapid', 'code'] else 0)

    clean['Central_Line'] = clean['Central_Line'].fillna(0).astype(str)
    clean['Has_Line'] = clean['Central_Line'].apply(lambda x: 1 if x.lower() not in ['0', 'nan', 'none', ''] else 0)

    clean['Discharge_Planned'] = pd.to_datetime(clean['Discharge_Planned'], errors='coerce').dt.date
    clean['Is_Active_DC'] = 0
    if shift == "Day":
        clean.loc[clean['Discharge_Planned'] == target_date, 'Is_Active_DC'] = 1
    
    clean['DI_Score'] = pd.to_numeric(clean['DI_Score'], errors='coerce').fillna(0)
    
    mask_reset = (clean['Room Empty'] == 1) | (clean['New_Patient'] == 1)
    
    # Calculate Acuity (Now uses Column G too)
    clean['Calculated_Acuity'] = clean.apply(calculate_acuity_score, axis=1)
    
    if mask_reset.any():
        clean.loc[mask_reset, 'Calculated_Acuity'] = 3
        cols_to_wipe = ['Titratable_Gtt', 'Insulin_Gtt', 'Isolation', 'CiWA', 'Heparin_Ther', 'Heparin_NonTher', 'Restraints', 'DI_Score', 'Sitter', 'Has_Rapid', 'Foley', 'Drains', 'Has_Line']
        clean.loc[mask_reset, cols_to_wipe] = 0
        clean.loc[mask_reset, 'Rapid_Response'] = 0
        clean.loc[mask_reset, 'Central_Line'] = 0

    clean['Workload_Score'] = (
        clean['Calculated_Acuity'] + 
        (clean['DI_Score'] / 20.0) +
        (clean['Restraints'] * 0.5) + 
        (clean['Sitter'] * 0.5) +
        (clean['Has_Rapid'] * 2.0) +
        (clean['Drains'] * 0.5) + 
        (clean['Foley'] * 0.25)
    )
    
    clean['Current_Nurse'] = clean['Current_Nurse'].fillna('Unknown')
    for c in ['Force_Assign', 'Avoid_Nurse']:
        clean[c] = clean[c].fillna('Unknown').astype(str)
            
    return clean

# --- APP ---
uploaded = st.file_uploader("Upload Input (.xlsx)", type=['xlsx', 'xlsm'])

if uploaded:
    raw = pd.read_excel(uploaded)
    df = preprocess_data(raw, assignment_date, shift_type)
    
    with st.expander("📋 Data Preview"):
        # Show Nurse Score vs Calc Score to verify logic
        st.dataframe(df[['Room', 'Self_Reported_Acuity', 'Calculated_Acuity', 'Workload_Score']])

    charges = [n for n in df['Current_Nurse'].unique() if str(n).lower() not in ['nan','0','unknown','']]
    charges.sort()
    off_going_charge = st.selectbox("Off-Going Charge:", options=charges, index=0 if charges else None)

    if st.button("🚀 Run Scheduler"):
        with st.spinner("Optimizing..."):
            nurses = df[['Nurse Name', 'Role', 'Max_Patients']].copy()
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
                st.error("❌ No nurses found! Check columns A-D.")
                st.stop()

            model = cp_model.CpModel()
            x = {}
            for n in nurses.index:
                for p in patients.index:
                    x[n, p] = model.NewBoolVar(f'x_{n}_{p}')

            for p in patients.index:
                model.Add(sum(x[n, p] for n in nurses.index) == 1)

            for p, pat in patients.iterrows():
                force = str(pat['Force_Assign']).strip().lower()
                if force not in ['0', 'unknown', 'nan', '', 'none']:
                    target_nurse = next((n for n in nurses.index if force in str(nurses.loc[n, 'Nurse Name']).lower()), None)
                    if target_nurse is not None:
                        model.Add(x[target_nurse, p] == 1)
                
                avoid = str(pat['Avoid_Nurse']).strip().lower()
                if avoid not in ['0', 'unknown', 'nan', '', 'none']:
                    target_nurse = next((n for n in nurses.index if avoid in str(nurses.loc[n, 'Nurse Name']).lower()), None)
                    if target_nurse is not None:
                        model.Add(x[target_nurse, p] == 0)

            objs = []
            nurse_dcs = []

            for n, nurse in nurses.iterrows():
                count = sum(x[n, p] for p in patients.index)
                
                t_titr = sum(x[n, p] * patients.loc[p, 'Titratable_Gtt'] for p in patients.index)
                t_ins = sum(x[n, p] * patients.loc[p, 'Insulin_Gtt'] for p in patients.index)
                t_iso = sum(x[n, p] * patients.loc[p, 'Isolation'] for p in patients.index)
                t_ciwa = sum(x[n, p] * patients.loc[p, 'CiWA'] for p in patients.index)
                t_empty = sum(x[n, p] * patients.loc[p, 'Room Empty'] for p in patients.index)
                t_rest = sum(x[n, p] * patients.loc[p, 'Restraints'] for p in patients.index)
                t_dc = sum(x[n, p] * patients.loc[p, 'Is_Active_DC'] for p in patients.index)
                
                nurse_dcs.append(t_dc) 

                model.Add(t_ins <= limit_insulin)
                has_insulin = model.NewBoolVar(f'has_ins_{n}')
                model.Add(t_ins > 0).OnlyEnforceIf(has_insulin)
                model.Add(t_ins == 0).OnlyEnforceIf(has_insulin.Not())
                model.Add(count <= 3).OnlyEnforceIf(has_insulin)

                roster_limit = int(nurse['Max_Patients'])
                model.Add(count <= 6)
                excess = model.NewIntVar(0, 6, f'excess_{n}')
                model.Add(excess >= count - roster_limit)
                model.Add(excess >= 0)
                objs.append(excess * -500)

                if str(nurse['Role']).lower() == 'charge':
                    model.Add(t_empty == 0)
                    model.Add(t_titr == 0)
                    model.Add(t_ins == 0)
                    model.Add(t_ciwa == 0)
                    model.Add(t_rest == 0)
                    model.Add(t_dc <= 1) 
                    objs.append(t_dc * -100)
                else:
                    model.Add(t_titr <= limit_titratable)
                    model.Add(t_ciwa <= 1)
                    model.Add(t_rest <= limit_restraints)
                
                model.Add(t_iso <= 2)

            min_dc = model.NewIntVar(0, 10, 'min_dc')
            max_dc = model.NewIntVar(0, 10, 'max_dc')
            model.AddMaxEquality(max_dc, nurse_dcs)
            model.AddMinEquality(min_dc, nurse_dcs)
            objs.append((max_dc - min_dc) * -50) 

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

                load = sum(x[n, p] * int(patients.loc[p, 'Workload_Score']*10) for p in patients.index)
                n_empty = sum(x[n, p] * patients.loc[p, 'Room Empty'] for p in patients.index)
                has_empty = model.NewBoolVar(f'he_{n}')
                model.Add(n_empty > 0).OnlyEnforceIf(has_empty)
                model.Add(n_empty == 0).OnlyEnforceIf(has_empty.Not())
                pen = model.NewIntVar(0, 5000, f'pen_{n}')
                model.Add(pen == load).OnlyEnforceIf(has_empty)
                model.Add(pen == 0).OnlyEnforceIf(has_empty.Not())
                objs.append(pen * -50)

                for p, pat in patients.iterrows():
                    if str(nurses.loc[n, 'Role']).lower() == 'charge':
                        if pat['Current_Nurse'] == off_going_charge and pat['Is_Active_DC']==0:
                            objs.append(x[n, p] * w_charge_pref)
                        objs.append(x[n, p] * int(pat['Workload_Score']*10) * -5)
                    else:
                        if pat['New_Patient']==0 and pat['Room Empty']==0:
                            if nurses.loc[n, 'Nurse Name'] == pat['Nurse_24hrs_Ago']:
                                objs.append(x[n, p] * w_continuity)
            
            nurse_loads = []
            for n in nurses.index:
                l = sum(x[n, p] * int(patients.loc[p, 'Workload_Score']*10) for p in patients.index)
                nurse_loads.append(l)
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
                st.session_state.results_found = True
                st.session_state.score = solver.ObjectiveValue()
                
                nurse_res = []
                for n in nurses.index:
                    my_p = []
                    handoff_sources = set()
                    stats = {'work':0, 'dc':0}
                    for p in patients.index:
                        if solver.Value(x[n, p]):
                            lbl = str(patients.loc[p, 'Room'])
                            if patients.loc[p, 'Room Empty']: lbl += " (E)"
                            if patients.loc[p, 'Is_Active_DC']: lbl += " (DC)"
                            my_p.append(lbl)
                            stats['work'] += patients.loc[p, 'Workload_Score']
                            stats['dc'] += patients.loc[p, 'Is_Active_DC']
                            src = str(patients.loc[p, 'Current_Nurse'])
                            if src.lower() not in ['nan', 'unknown', '0', '']:
                                handoff_sources.add(src)
                    nurse_res.append({
                        'Nurse': nurses.loc[n, 'Nurse Name'],
                        'Role': nurses.loc[n, 'Role'],
                        'Count': len(my_p),
                        'Rooms': ", ".join(my_p),
                        'Workload': round(stats['work'], 1),
                        'Report From': ", ".join(sorted(handoff_sources))
                    })
                st.session_state.df_nurse = pd.DataFrame(nurse_res)

                pat_res = []
                def is_y(val): return "Yes" if val == 1 else ""
                def get_o2(val): return str(val) if str(val)!='0' else ""
                def get_sitter(val): return "1:1" if val==1 else ""

                for p in patients.index:
                    assigned_n = "Unassigned"
                    for n in nurses.index:
                        if solver.Value(x[n, p]):
                            assigned_n = nurses.loc[n, 'Nurse Name']
                            break
                    pat_res.append({
                        'Room': patients.loc[p, 'Room'],
                        'Oncoming Nurse': assigned_n,
                        'Off-Going Nurse': patients.loc[p, 'Current_Nurse'],
                        'Acuity': patients.loc[p, 'Calculated_Acuity'],
                        'Titratable': is_y(patients.loc[p, 'Titratable_Gtt']),
                        'Insulin': is_y(patients.loc[p, 'Insulin_Gtt']),
                        'Heparin': is_y(patients.loc[p, 'Heparin_Ther']),
                        'Restraints': is_y(patients.loc[p, 'Restraints']),
                        'Sitter': get_sitter(patients.loc[p, 'Sitter']),
                        'Rapid/Code': str(patients.loc[p, 'Rapid_Response']) if patients.loc[p, 'Has_Rapid']==1 else "",
                        'Isolation': is_y(patients.loc[p, 'Isolation']),
                        'O2': get_o2(patients.loc[p, 'O2_Device']),
                        'Foley': is_y(patients.loc[p, 'Foley']),
                        'Drains': is_y(patients.loc[p, 'Drains']),
                        'Central Line': str(patients.loc[p, 'Central_Line']) if patients.loc[p, 'Has_Line']==1 else "",
                        'Active Discharge': is_y(patients.loc[p, 'Is_Active_DC'])
                    })
                st.session_state.df_patient = pd.DataFrame(pat_res)

                # --- MANAGER REPORT (UPDATED) ---
                mgr_res = []
                # Drips
                drip_list = []
                for i, r in patients.iterrows():
                    drips = []
                    if r['Insulin_Gtt']==1: drips.append("Insulin")
                    if r['Titratable_Gtt']==1: drips.append("Titratable")
                    if r['Heparin_Ther']==1: drips.append("Heparin(T)")
                    if r['Heparin_NonTher']==1: drips.append("Heparin(NT)")
                    if drips:
                        drip_list.append(f"{r['Room']} ({', '.join(drips)})")
                if drip_list:
                    mgr_res.append({'Category': 'Active Drips', 'Count': len(drip_list), 'Rooms': ", ".join(drip_list)})

                # Respiratory
                resp_list = []
                for i, r in patients.iterrows():
                    if str(r['O2_Device']) not in ['0', 'nan', '']:
                        resp_list.append(f"{r['Room']} ({r['O2_Device']})")
                if resp_list:
                    mgr_res.append({'Category': 'Respiratory', 'Count': len(resp_list), 'Rooms': ", ".join(resp_list)})

                # Rapids/Codes
                rapid_list = []
                for i, r in patients.iterrows():
                    if r['Has_Rapid'] == 1:
                        rapid_list.append(f"{r['Room']} ({r['Rapid_Response']})")
                if rapid_list:
                    mgr_res.append({'Category': 'Events (Rapid/Code)', 'Count': len(rapid_list), 'Rooms': ", ".join(rapid_list)})

                # Central Lines
                line_list = []
                for i, r in patients.iterrows():
                    if r['Has_Line'] == 1:
                        line_list.append(f"{r['Room']} ({r['Central_Line']})")
                if line_list:
                    mgr_res.append({'Category': 'Central Lines', 'Count': len(line_list), 'Rooms': ", ".join(line_list)})

                # Restraints
                rest_mask = (patients['Restraints']==1)
                if rest_mask.any():
                    rooms = patients.loc[rest_mask, 'Room'].tolist()
                    mgr_res.append({'Category': 'Restraints', 'Count': len(rooms), 'Rooms': ", ".join(rooms)})

                # Sitters
                sit_list = []
                for i, r in patients.iterrows():
                    if r['Sitter'] == 1:
                        sit_list.append(f"{r['Room']} (1:1)")
                if sit_list:
                    mgr_res.append({'Category': 'Sitters', 'Count': len(sit_list), 'Rooms': ", ".join(sit_list)})

                # Foleys
                foley_mask = (patients['Foley']==1)
                if foley_mask.any():
                    rooms = patients.loc[foley_mask, 'Room'].tolist()
                    mgr_res.append({'Category': 'Foleys', 'Count': len(rooms), 'Rooms': ", ".join(rooms)})

                # Discharges
                dc_mask = (patients['Is_Active_DC']==1)
                if dc_mask.any():
                    rooms = patients.loc[dc_mask, 'Room'].tolist()
                    mgr_res.append({'Category': 'Active Discharges', 'Count': len(rooms), 'Rooms': ", ".join(rooms)})

                st.session_state.df_manager = pd.DataFrame(mgr_res)
                st.session_state.total_acuity = patients['Calculated_Acuity'].sum()
                st.session_state.total_restr = patients['Restraints'].sum()
                st.session_state.total_rapid = patients['Has_Rapid'].sum()

            else:
                st.error("No solution found. Check constraints.")

    # --- DISPLAY ---
    if 'results_found' in st.session_state and st.session_state.results_found:
        st.success(f"✅ Assignments Generated!")
        
        st.subheader("📊 Manager Report")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Acuity Points", f"{st.session_state.total_acuity}")
        c2.metric("Restraints / Rapids", f"{st.session_state.total_restr} / {st.session_state.total_rapid}")
        c3.metric("Avg Workload/Nurse", f"{st.session_state.df_nurse['Workload'].mean():.1f}")

        st.table(st.session_state.df_manager)

        st.subheader("Nurse Workload Summary")
        st.dataframe(st.session_state.df_nurse)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_n = st.session_state.df_nurse.sort_values(by='Role', key=lambda col: col.str.lower() != 'charge')
            df_n.to_excel(writer, sheet_name='Nurse Summary', index=False)
            st.session_state.df_patient.to_excel(writer, sheet_name='Patient List', index=False)
            st.session_state.df_manager.to_excel(writer, sheet_name='Manager Huddle', index=False)
        
        st.download_button(
            label="💾 Download Workbook (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"Assignments_{assignment_date}_{shift_type}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("---")
        st.subheader("📧 Email Report")
        recipient = st.text_input("Recipient Email(s) (comma-separated):")
        if st.button("Send to Email"):
            if recipient:
                buffer.seek(0)
                success, msg = send_email(recipient, buffer, f"Assignments_{assignment_date}_{shift_type}.xlsx")
                if success: st.success(msg)
                else: st.error(f"Failed: {msg}")
            else:
                st.warning("Please enter an email address.")
