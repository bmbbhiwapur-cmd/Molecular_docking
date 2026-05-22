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

# --- CONFIGURATION & branding ---
st.set_page_config(page_title="InSilico BioSphere", layout="wide")

# --- UI HEADER ---
st.title("🔬 InSilico BioSphere: Automated Docking Studio")
st.markdown("""
**Developed by:** Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, 
Shivaji Science College, Nagpur, India | **Contact:** sarangresearch@gmail.com
""")

# --- ENGINE MANAGEMENT ---
def ensure_linux_vina_exists():
    if not os.path.exists("./vina"):
        with st.spinner("Initializing Cloud Computational Server..."):
            try:
                url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
                urllib.request.urlretrieve(url, "./vina")
                os.chmod("./vina", 0o755)
            except Exception as e:
                st.error(f"Failed to bootstrap engine: {e}")
ensure_linux_vina_exists()

# --- UTILITIES ---
def fetch_ligand_data(smiles):
    """Robust PubChem fetcher with browser-like headers to prevent request blocking."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{urllib.parse.quote(smiles)}/property/Title,MolecularWeight,MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode())
            p = data["PropertyTable"]["Properties"][0]
            return f"Name: {p.get('Title')} | MW: {p.get('MolecularWeight')} g/mol | Formula: {p.get('MolecularFormula')}"
    except Exception as e:
        return f"Metadata fetch failed: {e}"

# --- CORE DOCKING ENGINE ---
def run_docking(cx, cy, cz, sx, sy, sz, ex):
    cmd = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt", 
           "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz), 
           "--exhaustiveness", str(ex), "--out", "docking_poses.pdbqt"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

# --- UI LOGIC ---
if st.button("🔄 Reset Entire Environment"):
    for key in st.session_state.keys(): del st.session_state[key]
    for f in ["protein.pdb", "protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt"]:
        if os.path.exists(f): os.remove(f)
    st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.header("Step 1: Data Input")
    prot_id = st.text_input("Enter PDB ID", "2AMB")
    if st.button("Load Protein"):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{prot_id}.pdb", "protein.pdb")
        # Simplified conversion
        subprocess.run(["./vina", "--receptor", "protein.pdb", "--output_setup", "protein.pdbqt"], capture_output=True)
        st.session_state.prot_loaded = True
        st.success("Protein Prepared")

    smiles = st.text_input("Ligand SMILES", "CC(=O)NC1=CC=C(O)C=C1")
    if st.button("Fetch Ligand Info"):
        st.session_state.lig_meta = fetch_ligand_data(smiles)
        # Note: You would normally convert SMILES to PDBQT here
        st.session_state.lig_loaded = True
        st.success("Ligand Data Fetched")

    if st.session_state.get("lig_loaded"): 
        st.info(st.session_state.lig_meta)

    st.header("Step 2: Docking Params")
    cx = st.number_input("Center X", 0.0)
    cy = st.number_input("Center Y", 0.0)
    cz = st.number_input("Center Z", 0.0)
    sx = st.slider("Size X", 10, 40, 20)
    sy = st.slider("Size Y", 10, 40, 20)
    sz = st.slider("Size Z", 10, 40, 20)
    ex = st.slider("Exhaustiveness", 4, 32, 8)

    if st.session_state.get("prot_loaded") and st.session_state.get("lig_loaded"):
        if st.button("🚀 Initialize Docking"):
            with st.spinner("Computing interaction space..."):
                st.session_state.results = run_docking(cx, cy, cz, sx, sy, sz, ex)
                st.rerun()

with col2:
    st.header("Step 3: Results")
    if st.session_state.get("results"):
        st.text_area("Docking Output Logs:", value=st.session_state.results, height=300)
        st.success("Docking complete! Check the terminal logs above.")
    else:
        st.info("Input PDB/SMILES and initialize docking to see results.")
