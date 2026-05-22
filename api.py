import streamlit as st
import subprocess
import os
import urllib.request
import json
import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
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


# --- PUBCHEM AUTOMATED DATA CONVERTER ---

def fetch_ligand_data_from_pubchem(smiles_string):
    """Queries NCBI PubChem REST API to dynamically fetch validated small molecule attributes."""
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
    except Exception:
        pass 
    return metadata


# --- PDB METADATA & HETATM CO-CRYSTAL PARSER ---

def extract_pdb_metadata(file_path, pdb_id="Custom"):
    meta = {
        "title": "Uploaded Protein Structure Matrix", "id": pdb_id.upper(),
        "class": "Unknown Classification", "organism": "Unknown",
        "system": "Unknown Expression System", "method": "X-RAY DIFFRACTION", "res": "N/A"
    }
    if not os.path.exists(file_path): return meta
    
    with open(file_path, "r") as f:
        title_parts = []
        for line in f:
            if line.startswith("TITLE"): title_parts.append(line[10:80].strip())
            elif line.startswith("HEADER"): meta["class"] = line[10:50].strip().title()
            elif "ORGANISM_SCIENTIFIC" in line: meta["organism"] = line.split(":")[-1].replace(";","").strip()
            elif "EXPRESSION_SYSTEM" in line: meta["system"] = line.split(":")[-1].replace(";","").strip()
            elif line.startswith("EXPDTA"): meta["method"] = line[10:80].strip()
            elif "RESOLUTION." in line and "ANGSTROMS." in line:
                match = re.search(r"(\d+\.\d+)", line)
                if match: meta["res"] = f"{match.group(1)} Å"
    if title_parts: meta["title"] = " ".join(title_parts).title()
    return meta

def parse_bound_ligands(file_path):
    ligands = {}
    if not os.path.exists(file_path): return ligands
    
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                chain_id = line[21].strip() if line[21].strip() else "A"
                try: res_seq = int(line[22:26].strip())
                except ValueError: continue
                if res_name in ["HOH", "WAT", "DOD"]: continue
                
                key = f"{res_name}-{chain_id}-{res_seq}"
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError: continue
                
                if key not in ligands:
                    ligands[key] = {"res": res_name, "chain": chain_id, "seq": res_seq, "coords": []}
                ligands[key]["coords"].append((x, y, z))
                
    processed_ligands = []
    for key, info in ligands.items():
        pts = info["coords"]
        n_atoms = len(pts)
        if n_atoms < 4: continue
        
        cx, cy, cz = sum([p[0] for p in pts])/n_atoms, sum([p[1] for p in pts])/n_atoms, sum([p[2] for p in pts])/n_atoms
        bx = max([p[0] for p in pts]) - min([p[0] for p in pts]) + 10.0
        by = max([p[1] for p in pts]) - min([p[1] for p in pts]) + 10.0
        bz = max([p[2] for p in pts]) - min([p[2] for p in pts]) + 10.0
        
        processed_ligands.append({
            "ID": info["res"], "Chain": info["chain"], "ResSeq": info["seq"], "Atoms": n_atoms,
            "cx": round(cx, 2), "cy": round(cy, 2), "cz": round(cz, 2),
            "bx": round(bx, 1), "by": round(by, 1), "bz": round(bz, 1)
        })
    return processed_ligands


# --- ADVANCED BIOPHYSICAL INTERACTION PARSER ENGINE ---

def parse_pdbqt_coordinates(pdbqt_string):
    atoms = []
    for line in pdbqt_string.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = line[76:78].strip().upper()
                res_name = line[17:20].strip()
                res_seq = line[22:26].strip()
                atoms.append({"coord": np.array([x, y, z]), "element": element, "res": f"{res_name}{res_seq}"})
            except ValueError: continue
    return atoms

def compute_spatial_interactions(receptor_file, ligand_pdbqt_str):
    interactions = []
    if not os.path.exists(receptor_file): return interactions
    
    with open(receptor_file, "r") as f:
         receptor_atoms = parse_pdbqt_coordinates(f.read())
    ligand_atoms = parse_pdbqt_coordinates(ligand_pdbqt_str)
    
    seen = set()
    for l_at in ligand_atoms:
        for r_at in receptor_atoms:
            dist = np.linalg.norm(l_at["coord"] - r_at["coord"])
            if dist < 3.8: 
                res_id = r_at["res"]
                if res_id in seen: continue
                
                if l_at["element"] in ["N", "O", "F", "S"] and r_at["element"] in ["N", "O", "F", "S"]:
                    b_type = "Hydrogen Bond"
                elif "A" in r_at["element"] or (l_at["element"] == "C" and r_at["element"] == "C" and any(aro in r_at["res"] for aro in ["PHE", "TYR", "TRP"])):
                    b_type = "pi-Stacking / Hydrophobic"
                else:
                    b_type = "van der Waals Contact"
                    
                seen.add(res_id)
                interactions.append({
                    "Residue Contact": res_id,
                    "Interaction Type": b_type,
                    "Distance (Å)": round(dist, 2),
                    "r_coord": r_at["coord"].tolist(),
                    "l_coord": l_at["coord"].tolist()
                })
    return interactions


# --- BIOINFORMATICS STRUCTURAL CONVERTERS ---

def fetch_pdb_from_rcsb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_pdb)
        return True, local_pdb
    except Exception:
        return False, f"Could not find or download PDB ID '{pdb_id.upper()}'."

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False):
    autodock_type_map = {
        "H": "H", "HD": "HD", "HS": "HS", "C": "C", "A": "A", "N": "N", "NA": "NA", 
        "NS": "NS", "O": "O", "OA": "OA", "S": "S", "SA": "SA", "P": "P", "F": "F", 
        "CL": "Cl", "BR": "Br", "I": "I", "ZN": "Zn", "MG": "Mg"
    }
    torsions = 0
    if is_ligand:
        try:
            mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
            if mol: torsions = AllChem.CalcNumRotatableBonds(mol)
        except Exception: torsions = 4
    try:
        with open(input_pdb, "r") as pdb, open(output_pdbqt, "w") as pdbqt:
            if is_ligand: pdbqt.write("ROOT\n")
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    record_type = line[:6].strip()
                    try: atom_id = int(line[6:11].strip())
                    except ValueError: atom_id = 1
                    atom_name = line[12:16]
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip() if line[21].strip() else "A"
                    try: res_seq = int(line[22:26].strip())
                    except ValueError: res_seq = 1
                    try: x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                    except ValueError: continue
                    element = line[76:78].strip()
                    if not element: element = ''.join([c for c in atom_name if c.isalpha()])[0]
                    element = ''.join([c for c in element if c.isalpha()]).upper()
                    vina_type = autodock_type_map.get(element, element.title())
                    if element == "C" and "AR" in atom_name.upper(): vina_type = "A"
                    pdbqt.write(f"{record_type:<6}{atom_id:>5} {atom_name:<4} {res_name:>3} {chain_id}{res_seq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}    +0.000 {vina_type:<2}\n")
            if is_ligand:
                pdbqt.write("ENDROOT\n")
                pdbqt.write(f"TORSDOF {torsions}\n")
            else: pdbqt.write("ENDMDL\n")
        return True, output_pdbqt
    except Exception as e: return False, str(e)

def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return False, "Invalid SMILES."
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        temp_pdb = "temp_ligand.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
        convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        return True, output_filename
    except Exception as e: return False, str(e)


# --- LOG FILE PARSERS & UTILITY CONFIGURATIONS ---

def split_docking_poses(poses_file_path):
    poses = {}
    if not os.path.exists(poses_file_path): return poses
    current_mode, current_lines = None, []
    with open(poses_file_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                try: current_mode = int(line.split()[1])
                except Exception: current_mode = len(poses) + 1
                current_lines = []
            elif line.startswith("ENDMDL"):
                if current_mode is not None: poses[current_mode] = "".join(current_lines)
                current_mode = None
            else: current_lines.append(line)
    return poses

def get_pose_affinity(stdout_text, idx):
    """Parses standard Vina output terminal logs to trace specific pose metrics row values."""
    for line in stdout_text.split("\n"):
        m = re.match(r"^\s*(\d+)\s+([-+]?\d+\.\d+)", line)
        if m and int(m.group(1)) == idx: return m.group(2)
    return "N/A"


# --- HIGH PERFORMANCE VISUALIZATION CONSTRUCTS ---

def generate_2d_ligand_img(mol):
    if mol is None: return None
    try:
        mol_flat = Chem.Mol(mol)
        Chem.SanitizeMol(mol_flat)
        AllChem.Compute2DCoords(mol_flat)
        img = Draw.MolToImage(mol_flat, size=(340, 260))
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception: return None

def render_advanced_modeling_blueprint(receptor_data, ligand_data, mode="cartoon", show_surface=False, interactions_list=[]):
    surface_js = "viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if show_surface else ""
    int_lines_js = ""
    for interact in interactions_list:
        rc = interact["r_coord"]
        lc = interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js += f"""
        viewer.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        viewer.addLabel("{interact['Residue Contact']} ({interact['Distance (Å)']}A)", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:11}});
        """

    html_content = f"""
    <div id="wrapper_div" style="position:relative; width:100%;">
        <button onclick="toggleFullScreen()" style="position:absolute; top:12px; right:12px; z-index:9999; padding:6px 12px; background:#007bff; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; font-family:sans-serif; box-shadow:0 2px 4px rgba(0,0,0,0.15);">🖥 Fullscreen View</button>
        <div id="container" style="height: 480px; width: 100%; position: relative; border-radius:10px; border:1px solid #eaeaea; background:#ffffff;"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: '#ffffff'}});
        if (`{receptor_data}`.trim().length > 0) {{
            viewer.addModel(`{receptor_data}`, 'pdb');
            if ('{mode}' === 'cartoon') {{
                viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}});
            }} else if ('{mode}' === 'spacefill') {{
                viewer.setStyle({{model: 0}}, {{sphere: {{colorscheme: 'chain', radius:1.1}}}});
            }} else {{
                viewer.setStyle({{model: 0}}, {{stick: {{colorscheme: 'chain', radius:0.25}}}});
            }}
        }}
        {surface_js}
        if (`{ligand_data}`.trim().length > 0) {{
            viewer.addModel(`{ligand_data}`, 'pdb');
            viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
        }}
        {int_lines_js}
        viewer.zoomTo(); viewer.render();
        function toggleFullScreen() {{
            let elem = document.getElementById("wrapper_div");
            if (!document.fullscreenElement) {{ elem.requestFullscreen(); document.getElementById("container").style.height = "90vh"; }}
            else {{ document.exitFullscreen(); document.getElementById("container").style.height = "480px"; }}
        }}
        document.addEventListener('fullscreenchange', () => {{ if (!document.fullscreenElement) document.getElementById("container").style.height = "480px"; }});
    </script>
    """
    components.html(html_content, height=510)


# --- ADAPTIVE LAB-TIER PDF EXECUTABLE REPORT LAYER ---

def generate_pdf_report_bytes(protein_meta, ligand_summary, grid_box, pose_no, energy, interactions, full_df):
    from datetime import datetime
    try:
        from weasyprint import HTML
    except ImportError:
        return None
        
    table_rows_html = ""
    for idx, row in full_df.iterrows():
        table_rows_html += f"""
        <tr>
            <td>Mode {row['Binding Mode']}</td>
            <td><b>{row['Affinity (kcal/mol)']}</b></td>
            <td>{row['RMSD l.b.']}</td>
            <td>{row['RMSD u.b.']}</td>
            <td style="font-size: 10px; color: #4a5568;">{row['Interacting Residues']}</td>
        </tr>
        """
        
    interactions_html = ""
    for item in interactions:
        interactions_html += f"<li><b>{item['Residue Contact']}:</b> {item['Interaction Type']} ({item['Distance (Å)']} Å)</li>"
    if not interactions_html:
        interactions_html = "<li>No connections tracked under 3.8 Å radius.</li>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 20mm 15mm 22mm 15mm;
                @bottom-center {{
                    content: "Report is generated by InSilico BioSphere, Developed by Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India | Contact: sarangreserach@gmail.com";
                    font-family: Arial, sans-serif;
                    font-size: 7pt;
                    color: #718096;
                    border-top: 1px solid #e2e8f0;
                    padding-top: 5px;
                }}
            }}
            body {{ font-family: Arial, sans-serif; color: #2d3748; line-height: 1.4; }}
            .header-banner {{ background-color: #1a365d; color: white; padding: 20px; text-align: center; margin-bottom: 20px; border-radius: 4px; }}
            .header-banner h1 {{ margin: 0; font-size: 24pt; }}
            .header-banner p {{ margin: 5px 0 0 0; font-size: 11pt; opacity: 0.9; }}
            h2 {{ font-size: 13pt; color: #1a365d; border-bottom: 2px solid #2b6cb0; padding-bottom: 3px; margin-top: 20px; }}
            .kpi-container {{ background-color: #f0fff4; border: 1px solid #c6f6d5; border-left: 5px solid #38a169; padding: 12px; margin: 15px 0; border-radius: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th {{ background-color: #2b6cb0; color: white; padding: 6px; text-align: left; font-size: 9pt; }}
            td {{ padding: 6px; border-bottom: 1px solid #e2e8f0; font-size: 9pt; }}
        </style>
    </head>
    <body>
        <div class="header-banner">
            <h1>InSilico BioSphere</h1>
            <p>Report on Docking Between PDB ID: {protein_meta['id']} vs Ligand SMILES</p>
        </div>
        
        <h2>1. Target ID Protein Specification Information</h2>
        <table>
            <tr><td style="width:30%;"><b>PDB Entry Accession:</b></td><td><code>{protein_meta['id']}</code></td></tr>
            <tr><td><b>Macromolecular Title:</b></td><td>{protein_meta['title']}</td></tr>
            <tr><td><b>Source Organism:</b></td><td><i>{protein_meta['organism']}</i></td></tr>
            <tr><td><b>Method / Resolution:</b></td><td>{protein_meta['method']} / {protein_meta['res']}</td></tr>
        </table>

        <h2>2. Small Molecule Ligand Specification Information</h2>
        <table>
            <tr><td style="width:30%;"><b>Chemical Coordinates:</b></td><td>{ligand_summary.replace('**','')}</td></tr>
        </table>

        <h2>3. Docking Grid Box Search space Configurations</h2>
        <table>
            <tr><td style="width:30%;"><b>Center Matrix (X, Y, Z):</b></td><td>({grid_box['cx']}, {grid_box['cy']}, {grid_box['cz']})</td></tr>
            <tr><td><b>Pocket Dimensions size:</b></td><td>{grid_box['sx']} Å × {grid_box['sy']} Å × {grid_box['sz']} Å</td></tr>
            <tr><td><b>Exhaustiveness Level:</b></td><td>{grid_box['ex']}</td></tr>
        </table>

        <h2>4. Selected Pose Fit Structural Metrics Profile</h2>
        <div class="kpi-container">
            <b>Active Mode Configuration Fit:</b> Mode {pose_no} Pose Matrix Alignment Alignment<br>
            <span style="font-size: 18pt; font-weight: bold; color: #22543d;">Binding Free Energy Energy: {energy} kcal/mol</span>
        </div>

        <h2>5. Binding Site Target Residues & Bonding Types Breakdown</h2>
        <ul>{interactions_html}</ul>

        <div style="page-break-before: always;"></div>
        <h2>6. Comprehensive Screening Rankings Matrix (All Compiled Poses)</h2>
        <table>
            <thead>
                <tr>
                    <th>Pose Mode</th>
                    <th>Binding Energy (kcal/mol)</th>
                    <th>RMSD l.b.</th>
                    <th>RMSD u.b.</th>
                    <th>Interacting Target Amino Acid Residues</th>
                </tr>
            </thead>
            <tbody>{table_rows_html}</tbody>
        </table>
    </body>
    </html>
    """
    import io
    pdf_buffer = io.BytesIO()
    HTML(string=html_content).write_pdf(pdf_buffer)
    return pdf_buffer.getvalue()


# --- APPLICATION DASHBOARD WORKSPACE ---

st.set_page_config(page_title="In Silico Docking Hub", layout="wide")
st.title("🔬 Automated Molecular Docking Studio")

if "cx" not in st.session_state: st.session_state.cx = 0.0
if "cy" not in st.session_state: st.session_state.cy = 0.0
if "cz" not in st.session_state: st.session_state.cz = 0.0
if "sx" not in st.session_state: st.session_state.sx = 20
if "sy" not in st.session_state: st.session_state.sy = 20
if "sz" not in st.session_state: st.session_state.sz = 20
if "target_ready" not in st.session_state: st.session_state.target_ready = False
if "ligand_ready" not in st.session_state: st.session_state.ligand_ready = False
if "local_target_path" not in st.session_state: st.session_state.local_target_path = None
if "pdb_id_display" not in st.session_state: st.session_state.pdb_id_display = "Custom"
if "docking_results_raw" not in st.session_state: st.session_state.docking_results_raw = None
if "serialized_ligand_block" not in st.session_state: st.session_state.serialized_ligand_block = None
if "ligand_summary_text" not in st.session_state: st.session_state.ligand_summary_text = ""
if "smiles_cache" not in st.session_state: st.session_state.smiles_cache = ""
if "selected_pose_idx" not in st.session_state: st.session_state.selected_pose_idx = 1

if st.button("🔄 Reset Entire Environment for Fresh Docking", type="secondary", use_container_width=True):
    for key in list(st.session_state.keys()): del st.session_state[key]
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt", "temp_lig_state.pdb"]:
        if os.path.exists(f): os.remove(f)
    st.success("Dashboard cache and runtime structures completely cleared!")
    st.rerun()

col_params, col_visual = st.columns([1, 1])

with col_params:
    st.header("1. Target Protein Setup")
    protein_source = st.radio("Choose Protein Input Method:", ["Type 4-Letter PDB ID", "Upload File (.pdb or .pdbqt)"])
    
    if protein_source == "Type 4-Letter PDB ID":
        pdb_id_input = st.text_input("Enter RCSB PDB ID", value="2AMB").strip()
        if st.button("📥 Load Target Structure"):
            if pdb_id_input:
                success, path = fetch_pdb_from_rcsb(pdb_id_input)
                if success:
                    st.session_state.local_target_path = path
                    st.session_state.pdb_id_display = pdb_id_input.upper()
                    conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                    st.session_state.target_ready = conv_ok
                    st.success(f"Protein {pdb_id_input.upper()} successfully loaded!")
                    st.rerun()
                else: st.error(path)
    else:
        uploaded_file = st.file_uploader("Upload Target Protein File", type=["pdb", "pdbqt"])
        if uploaded_file:
            path = f"uploaded_{uploaded_file.name}"
            if st.session_state.local_target_path != path:
                with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
                st.session_state.local_target_path = path
                st.session_state.pdb_id_display = "Uploaded File"
                if uploaded_file.name.endswith(".pdb"):
                    conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                    st.session_state.target_ready = conv_ok
                else:
                    os.replace(path, "protein.pdbqt")
                    st.session_state.target_ready = True
                    st.session_state.local_target_path = None
                st.rerun()

    if st.session_state.target_ready and st.session_state.local_target_path:
        meta = extract_pdb_metadata(st.session_state.local_target_path, st.session_state.pdb_id_display)
        st.markdown(f"""
        > **Protein Summary Profile:** \n> * **Title:** {meta['title']}  
        > * **PDB ID:** `{meta['id']}` | **Classification:** {meta['class']}  
        > * **Organism(s):** *{meta['organism']}* | **Expression System:** {meta['system']}  
        > * **Experimental Method:** {meta['method']} | **Resolution:** **{meta['res']}**
        """)

    st.header("2. Small Molecule Ligand Setup")
    ligand_source = st.radio("Choose Ligand Input Method:", ["SMILES String Input", "Upload Structural File (.pdb, .sdf)"])
    
    smiles_input_val = ""
    uploaded_lig_buffer = None
    uploaded_lig_name = ""

    if ligand_source == "SMILES String Input":
        smiles_input_val = st.text_input("Enter Ligand SMILES String", "CC(=O)NC1=CC=C(O)C=C1").strip()
    else:
        uploaded_lig_file = st.file_uploader("Upload Small Molecule File", type=["pdb", "sdf"])
        if uploaded_lig_file:
            uploaded_lig_buffer = uploaded_lig_file
            uploaded_lig_name = uploaded_lig_file.name

    if st.button("📥 Load Ligand Structure", key="load_ligand_btn"):
        if ligand_source == "SMILES String Input" and smiles_input_val:
            with st.spinner("Querying PubChem Repositories..."):
                pub_data = fetch_ligand_data_from_pubchem(smiles_input_val)
                try:
                    mol = Chem.MolFromSmiles(smiles_input_val)
                    if mol:
                        ok, _ = convert_smiles_to_pdbqt(smiles_input_val, "ligand.pdbqt")
                        if ok:
                            st.session_state.ligand_ready = True
                            st.session_state.smiles_cache = smiles_input_val
                            with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                            st.session_state.ligand_summary_text = f"**Name:** {pub_data['name']} | **Formula:** {pub_data['formula']} | **Molecular Weight:** {pub_data['mw']}"
                            st.success("Ligand metadata mapped from PubChem!")
                            st.rerun()
                except Exception as e: st.error(f"
