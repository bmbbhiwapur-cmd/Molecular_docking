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
from datetime import datetime

# --- CLOUD CONTEXT ENGINE MANAGEMENT ---
def ensure_linux_vina_exists():
    binary_name = "./vina"
    if not os.path.exists(binary_name):
        with st.spinner("Initializing Cloud Computational Server Environment..."):
            try:
                url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
                urllib.request.urlretrieve(url, binary_name)
                os.chmod(binary_name, 0o755)
            except Exception as e:
                st.error(f"Failed to bootstrap engine: {e}")
ensure_linux_vina_exists()

# --- UTILITIES ---
def fetch_ligand_data(smiles):
    meta = {"name": "Unknown", "mw": "N/A", "formula": "N/A"}
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{urllib.parse.quote(smiles)}/property/Title,MolecularWeight,MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as res:
            data = json.loads(res.read().decode())
            p = data["PropertyTable"]["Properties"][0]
            meta = {"name": p.get("Title", "Ligand"), "mw": f"{p.get('MolecularWeight')} g/mol", "formula": p.get("MolecularFormula")}
    except: pass
    return meta

def extract_pdb_metadata(path, pdb_id):
    meta = {"id": pdb_id.upper(), "title": "N/A", "organism": "N/A", "method": "N/A", "res": "N/A"}
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if line.startswith("TITLE"): meta["title"] = line[10:80].strip()
                elif "EXPDTA" in line: meta["method"] = line[10:80].strip()
                elif "RESOLUTION." in line: meta["res"] = re.search(r"(\d+\.\d+)", line).group(1) if re.search(r"(\d+\.\d+)", line) else "N/A"
    return meta

def parse_bound_ligands(path):
    ligands = {}
    if not os.path.exists(path): return []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                res = line[17:20].strip()
                if res in ["HOH", "WAT", "DOD"]: continue
                key = f"{res}-{line[21].strip() or 'A'}-{line[22:26].strip()}"
                if key not in ligands: ligands[key] = {"res": res, "coords": []}
                ligands[key]["coords"].append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return [{"ID": v["res"], "cx": round(np.mean([c[0] for c in v["coords"]]), 2), "cy": round(np.mean([c[1] for c in v["coords"]]), 2), "cz": round(np.mean([c[2] for c in v["coords"]]), 2), "bx": 20, "by": 20, "bz": 20} for k, v in ligands.items() if len(v["coords"]) > 4]

def compute_spatial_interactions(rec_file, lig_pdbqt):
    def get_atoms(data):
        return [{"coord": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]), "res": f"{line[17:20].strip()}{line[22:26].strip()}"} for line in data.split('\n') if line.startswith(("ATOM", "HETATM"))]
    
    ints = []
    r_atoms = get_atoms(open(rec_file).read())
    l_atoms = get_atoms(lig_pdbqt)
    seen = set()
    for l in l_atoms:
        for r in r_atoms:
            dist = np.linalg.norm(l["coord"] - r["coord"])
            if dist < 3.8 and r["res"] not in seen:
                seen.add(r["res"])
                ints.append({"Residue Contact": r["res"], "Interaction Type": "Contact", "Distance (Å)": round(dist, 2), "r_coord": r["coord"].tolist(), "l_coord": l["coord"].tolist()})
    return ints

def split_poses(file):
    if not os.path.exists(file): return {}
    content = open(file, "r").read()
    parts = re.split(r"MODEL", content)
    return {i: p for i, p in enumerate(parts) if p.strip()}

# --- VIEWPORT ---
def render_3d(receptor_data, ligand_data, interactions=[]):
    lines = "".join([f"viewer.addCylinder({{start:{{x:{i['r_coord'][0]},y:{i['r_coord'][1]},z:{i['r_coord'][2]}}}, end:{{x:{i['l_coord'][0]},y:{i['l_coord'][1]},z:{i['l_coord'][2]}}}, radius:0.07, color:'yellow', dashed:true}});" for i in interactions])
    html = f"""<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <div id="container" style="height: 480px; width: 100%; border: 1px solid #ccc;"></div>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: 'white'}});
        viewer.addModel(`{receptor_data}`, 'pdb');
        viewer.setStyle({{cartoon: {{colorscheme: 'chain'}} }});
        viewer.addModel(`{ligand_data}`, 'pdb');
        viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon'}} }});
        {lines}
        viewer.zoomTo(); viewer.render();
    </script>"""
    components.html(html, height=500)

# --- UI WORKSPACE ---
st.set_page_config(page_title="InSilico BioSphere", layout="wide")
st.title("🔬 InSilico BioSphere: Automated Docking Studio")

if st.button("🔄 Reset Entire Environment for Fresh Docking"):
    for key in list(st.session_state.keys()): del st.session_state[key]
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt", "temp_lig_state.pdb"]:
        if os.path.exists(f): os.remove(f)
    st.rerun()

# [ ... (Protein/Ligand loading code here) ... ]

if st.session_state.get("docking_results_raw"):
    st.header("📊 Results Analysis")
    poses = split_poses("docking_poses.pdbqt")
    mode = st.selectbox("Select Pose", list(poses.keys()))
    
    # Analysis & Report
    active_ints = compute_spatial_interactions("protein.pdbqt", poses[mode])
    
    # Report Section
    st.subheader("📋 Comprehensive In Silico Screening Report")
    cats = {"Acidic": [], "Basic": [], "Polar": [], "Hydrophobic": []}
    for i in active_ints:
        res = "".join([c for c in i["Residue Contact"] if c.isalpha()])
        if res in ["ASP", "GLU"]: cats["Acidic"].append(i["Residue Contact"])
        elif res in ["LYS", "ARG", "HIS"]: cats["Basic"].append(i["Residue Contact"])
        elif res in ["SER", "THR", "ASN", "GLN", "CYS", "TYR"]: cats["Polar"].append(i["Residue Contact"])
        else: cats["Hydrophobic"].append(i["Residue Contact"])
    
    report_text = f"Report: Docking {st.session_state.pdb_id_display} vs Ligand\n\nInteractions:\n" + "\n".join([f"{k}: {', '.join(set(v))}" for k,v in cats.items() if v])
    st.text_area("Copy Report:", value=report_text, height=300)
    
    render_3d(open("protein.pdbqt").read(), poses[mode], interactions_list=active_ints)
