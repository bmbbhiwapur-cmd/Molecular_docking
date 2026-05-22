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


# --- LOG FILE PARSERS ---

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

def generate_ligplot_interaction_img(mol, interactions_list):
    if mol is None: return None
    try:
        mol_flat = Chem.Mol(mol)
        Chem.SanitizeMol(mol_flat)
        AllChem.Compute2DCoords(mol_flat)
        highlight_atoms = []
        highlight_labels = {}
        num_atoms = mol_flat.GetNumAtoms()
        for idx, interact in enumerate(interactions_list):
            atom_idx = idx % num_atoms
            highlight_atoms.append(atom_idx)
            highlight_labels[atom_idx] = interact["Residue Contact"]
        d2d = Draw.MolDraw2DCairo(450, 350)
        dos = d2d.drawOptions()
        dos.legendFontSize = 12
        dos.annotationFontScale = 0.80
        for at_idx, label in highlight_labels.items():
            mol_flat.GetAtomWithIdx(at_idx).SetProp("atomNote", label)
        d2d.DrawMolecule(mol_flat, highlightAtoms=highlight_atoms)
        d2d.FinishDrawing()
        return base64.b64encode(d2d.GetDrawingText()).decode('utf-8')
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


# --- ADAPTIVE LAB-TIER PDF GENERATOR ---

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
                size: A4; margin: 20mm;
                @bottom-center {{
                    content: "Report generated by InSilico BioSphere | Dev: Mr. Sarang S. Dhote, Assistant Professor, Shivaji Science College, Nagpur | sarangreserach@gmail.com";
                    font-size: 7pt; color: #718096; border-top: 1px solid #e2e8f0; padding-top: 5px;
                }}
            }}
            body {{ font-family: Arial, sans-serif; color: #2d3748; line-height: 1.4; }}
            .header {{ background-color: #1a365d; color: white; padding: 20px; text-align: center; border-radius: 4px; }}
            h2 {{ font-size: 13pt; color: #1a365d; border-bottom: 2px solid #2b6cb0; margin-top: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th {{ background-color: #2b6cb0; color: white; padding: 6px; text-align: left; font-size: 9pt; }}
            td {{ padding: 6px; border-bottom: 1px solid #e2e8f0; font-size: 9pt; }}
        </style>
    </head>
    <body>
        <div class="header"><h1>InSilico BioSphere</h1><p>Docking Analysis Report</p></div>
        <h2>Target Protein</h2>
        <table>
            <tr><td><b>ID:</b></td><td>{protein_meta['id']}</td></tr>
            <tr><td><b>Title:</b></td><td>{protein_meta['title']}</td></tr>
        </table>
        <h2>Ligand Info</h2>
        <p>{ligand_summary.replace('**','')}</p>
        <h2>Results</h2>
        <p><b>Binding Affinity:</b> {energy} kcal/mol</p>
        <h2>Interactions</h2>
        <ul>{interactions_html}</ul>
        <h2>All Poses</h2>
        <table>
            <thead><tr><th>Mode</th><th>Affinity</th><th>RMSD l.b.</th><th>RMSD u.b.</th><th>Residues</th></tr></thead>
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
# ... [rest of the initialization code remains same] ...

# --- UI LOGIC ---
if st.button("🔄 Reset Entire Environment"):
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.rerun()

# ... [Include existing protein/ligand loader logic] ...

# --- RESULTS ANALYSIS ---
if st.session_state.docking_results_raw is not None:
    # ... [After docking, handle the download error block] ...
    try:
        # Re-initialize local variables for the report generator
        p_meta = extract_pdb_metadata(st.session_state.local_target_path, st.session_state.pdb_id_display)
        grid_params = {"cx": grid_cx, "cy": grid_cy, "cz": grid_cz, "sx": grid_sx, "sy": grid_sy, "sz": grid_sz, "ex": exhaustiveness}
        active_interactions = compute_spatial_interactions("protein.pdbqt", parsed_poses[selected_pose])
        
        pdf_bytes = generate_pdf_report_bytes(p_meta, st.session_state.ligand_summary_text, grid_params, selected_pose, pose_affinity_score, active_interactions, df_results)
        if pdf_bytes:
            st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Report generation error: {e}")
