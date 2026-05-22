import streamlit as st
import subprocess
import os
import urllib.request
import json
import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import streamlit.components.v1 as components
import base64

# --- SET UP ENVIRONMENT ---
if not os.path.exists("./vina"):
    url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
    urllib.request.urlretrieve(url, "./vina")
    os.chmod("./vina", 0o755)

# --- HELPER FUNCTIONS ---
def get_pubchem_data(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{urllib.parse.quote(smiles)}/property/Title,MolecularWeight,MolecularFormula/JSON"
        with urllib.request.urlopen(url, timeout=5) as res:
            data = json.loads(res.read().decode())
            p = data["PropertyTable"]["Properties"][0]
            return f"Name: {p.get('Title')} | MW: {p.get('MolecularWeight')} | Formula: {p.get('MolecularFormula')}"
    except: return "Metadata fetch failed."

def run_docking(cx, cy, cz, sx, sy, sz, ex):
    cmd = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt", 
           "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz), 
           "--exhaustiveness", str(ex), "--out", "docking_poses.pdbqt"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

# --- UI LOGIC ---
st.set_page_config(page_title="InSilico BioSphere", layout="wide")
st.title("🔬 InSilico BioSphere: Automated Docking Studio")

if st.button("🔄 Reset Entire Environment"):
    for key in st.session_state.keys(): del st.session_state[key]
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt"]:
        if os.path.exists(f): os.remove(f)
    st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.header("Step 1: Input Setup")
    prot_id = st.text_input("Enter PDB ID (e.g., 2AMB)", "2AMB")
    if st.button("Load Protein"):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{prot_id}.pdb", "protein.pdb")
        st.session_state.prot_loaded = True
        st.success("Protein Loaded")

    smiles = st.text_input("Ligand SMILES", "CC(=O)NC1=CC=C(O)C=C1")
    if st.button("Load Ligand"):
        st.session_state.lig_meta = get_pubchem_data(smiles)
        st.session_state.lig_loaded = True
        st.success("Ligand Loaded")

    if st.session_state.get("lig_loaded"): st.write(st.session_state.lig_meta)

    st.header("Step 2: Docking Parameters")
    cx = st.number_input("Center X", 0.0)
    cy = st.number_input("Center Y", 0.0)
    cz = st.number_input("Center Z", 0.0)
    sx = st.slider("Size X", 10, 40, 20)
    sy = st.slider("Size Y", 10, 40, 20)
    sz = st.slider("Size Z", 10, 40, 20)
    ex = st.slider("Exhaustiveness", 4, 32, 8)

    if st.button("🚀 Initialize Docking"):
        with st.spinner("Running Vina..."):
            st.session_state.results = run_docking(cx, cy, cz, sx, sy, sz, ex)
            st.rerun()

with col2:
    st.header("Step 3: Results")
    if st.session_state.get("results"):
        st.subheader("Docking Output")
        st.text(st.session_state.results)
        
        # Simple pose parser
        st.subheader("Pose Visualization")
        if os.path.exists("docking_poses.pdbqt"):
            # For brevity, visualization logic is simplified here
            st.info("Docking complete. Use the above logs to analyze results.")
            
            # Automated Report
            st.subheader("📋 Screening Report")
            report = f"Docking Report for {prot_id}\n\n{st.session_state.lig_meta}\n\nParameters: Center({cx},{cy},{cz}) Size({sx},{sy},{sz})"
            st.text_area("Summary:", value=report, height=200)
    else:
        st.info("Docking results will appear here after execution.")
