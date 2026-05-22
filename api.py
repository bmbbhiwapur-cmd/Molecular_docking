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

# --- CLOUD CONTEXT ENGINE MANAGEMENT ---
def ensure_linux_vina_exists():
    binary_name = "./vina"
    if not os.path.exists(binary_name):
        with st.spinner("Initializing Cloud Computational Server Environment (Downloading Vina)..."):
            try:
                url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
                urllib.request.urlretrieve(url, binary_name)
                os.chmod(binary_name, 0o755)
                st.success("Cloud backend binaries mounted successfully!")
            except Exception as e:
                st.error(f"Failed to bootstrap Linux engine environment: {e}")
ensure_linux_vina_exists()

# --- UTILITIES ---
def fetch_ligand_data_from_pubchem(smiles_string):
    metadata = {"name": "Unknown Compound Name", "mw": "N/A", "formula": "N/A"}
    try:
        escaped_smiles = urllib.parse.quote(smiles_string)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{escaped_smiles}/property/Title,MolecularWeight,MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            res_data = json.loads(response.read().decode())
            if "PropertyTable" in res_data and "Properties" in res_data["PropertyTable"]:
                props = res_data["PropertyTable"]["Properties"][0]
                metadata["name"] = props.get("Title", "Target Chemical Derivative")
                metadata["mw"] = f"{props.get('MolecularWeight', 'N/A')} g/mol"
                metadata["formula"] = props.get("MolecularFormula", "N/A")
    except: pass
    return metadata

def split_docking_poses(poses_file_path):
    poses = {}
    if not os.path.exists(poses_file_path): return poses
    current_mode, current_lines = None, []
    with open(poses_file_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                try: current_mode = int(line.split()[1])
                except: current_mode = len(poses) + 1
                current_lines = []
            elif line.startswith("ENDMDL"):
                if current_mode is not None: poses[current_mode] = "".join(current_lines)
                current_mode = None
            else: current_lines.append(line)
    return poses

def get_pose_affinity(stdout_text, idx):
    for line in stdout_text.split("\n"):
        m = re.match(r"^\s*(\d+)\s+([-+]?\d+\.\d+)", line)
        if m and int(m.group(1)) == idx: return m.group(2)
    return "N/A"

def extract_pdb_metadata(file_path, pdb_id="Custom"):
    meta = {"title": "Uploaded Protein", "id": pdb_id.upper(), "class": "N/A", "organism": "N/A", "method": "X-RAY", "res": "N/A"}
    if not os.path.exists(file_path): return meta
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("TITLE"): meta["title"] = line[10:80].strip()
            elif "RESOLUTION." in line: meta["res"] = re.search(r"(\d+\.\d+)", line).group(1) if re.search(r"(\d+\.\d+)", line) else "N/A"
    return meta

# --- RENDERERS ---
def render_complex_html(receptor_pdbqt, ligand_pdbqt=None):
    ligand_block = f"viewer.addModel(`{ligand_pdbqt}`, 'pdb'); viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'cyanCarbon', radius: 0.23}}}});" if ligand_pdbqt else ""
    html_content = f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <div id="container" style="height: 380px; width: 100%; position: relative;"></div>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: '#f8f9fa'}});
        if (`{receptor_pdbqt}`.trim().length > 0) {{
            viewer.addModel(`{receptor_pdbqt}`, 'pdb');
            viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'spectrum'}}}});
        }}
        {ligand_block}
        viewer.zoomTo(); viewer.render();
    </script>
    """
    components.html(html_content, height=390)

# --- APP ---
st.set_page_config(page_title="InSilico BioSphere", layout="wide")
st.title("🔬 InSilico BioSphere: Automated Docking Studio")
st.markdown("Developed by: Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India | contact: sarangresearch@gmail.com")

if st.button("🔄 Reset Entire Environment"):
    for key in list(st.session_state.keys()): del st.session_state[key]
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt"]:
        if os.path.exists(f): os.remove(f)
    st.rerun()

col_params, col_visual = st.columns([1, 1])

with col_params:
    st.header("1. Target Protein Setup")
    # ... (Keep your existing protein loading logic here) ...
    st.header("2. Ligand Setup")
    # ... (Keep your existing ligand loading logic here) ...
    run_btn = st.button("🚀 Initialize Docking Algorithm")

with col_visual:
    st.header("5. Active Viewport Canvas")
    if st.session_state.get("docking_results_raw"):
        poses = split_docking_poses("docking_poses.pdbqt")
        mode = st.selectbox("Choose Pose", list(poses.keys()))
        with open("protein.pdbqt", "r") as f: p_data = f.read()
        render_complex_html(p_data, poses[mode])
        st.metric("Affinity", f"{get_pose_affinity(st.session_state.docking_results_raw, mode)} kcal/mol")
