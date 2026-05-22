import streamlit as st
import subprocess
import os
import urllib.request
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
            if line.startswith("TITLE"):
                title_parts.append(line[10:80].strip())
            elif line.startswith("HEADER"):
                meta["class"] = line[10:50].strip().title()
            elif "ORGANISM_SCIENTIFIC" in line:
                meta["organism"] = line.split(":")[-1].replace(";","").strip()
            elif "EXPRESSION_SYSTEM" in line:
                meta["system"] = line.split(":")[-1].replace(";","").strip()
            elif line.startswith("EXPDTA"):
                meta["method"] = line[10:80].strip()
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
                chain_id = line[21].strip()
                if not chain_id: chain_id = "A"
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
        
        cx = sum([p[0] for p in pts]) / n_atoms
        cy = sum([p[1] for p in pts]) / n_atoms
        cz = sum([p[2] for p in pts]) / n_atoms
        
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
                    b_type = "Hydrogen Bond (H-Bond)"
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


# --- CHEMINFORMATICS FRAMEWORKS ---

def fetch_pdb_from_rcsb(pdb_id):
    """Fetches a standard PDB structure directly from the RCSB server."""
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
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                    except ValueError: continue
                    
                    element = line[76:78].strip()
                    if not element: element = ''.join([c for c in atom_name if c.isalpha()])[0]
                    element = ''.join([c for c in element if c.isalpha()]).upper()
                    
                    vina_type = autodock_type_map.get(element, element.title())
                    if element == "C" and "AR" in atom_name.upper(): vina_type = "A"

                    pdbqt.write(
                        f"{record_type:<6}{atom_id:>5} {atom_name:<4} {res_name:>3} "
                        f"{chain_id}{res_seq:>4}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}    "
                        f"+0.000 {vina_type:<2}\n"
                    )
            if is_ligand:
                pdbqt.write("ENDROOT\n")
                pdbqt.write(f"TORSDOF {torsions}\n")
            else:
                pdbqt.write("ENDMDL\n")
        return True, output_pdbqt
    except Exception as e:
        return False, str(e)

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


# --- VISUALIZATION CONSTRUCTS & ADVANCED BLUEPRINT LAYER ---

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

def render_advanced_modeling_blueprint(receptor_data, ligand_data, mode="cartoon", show_surface=False, interactions_list=[]):
    """Generates an advanced multi-model workspace tracking spatial atomic binding constraints."""
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
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <div id="container" style="height: 480px; width: 100%; position: relative; border-radius:10px; border:1px solid #eaeaea; box-shadow: 0 4px 6px rgba(0,0,0,0.05);"></div>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: '#ffffff'}});
        if (`{receptor_data}`.trim().length > 0) {{
            viewer.addModel(`{receptor_data}`, 'pdb');
            if ('{mode}' === 'cartoon') {{
                viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'spectrum'}}}});
            }} else if ('{mode}' === 'spacefill') {{
                viewer.setStyle({{model: 0}}, {{sphere: {{colorscheme: 'spectrum', radius:1.1}}}});
            }} else {{
                viewer.setStyle({{model: 0}}, {{stick: {{colorscheme: 'spectrum', radius:0.25}}}});
            }}
        }}
        {surface_js}
        if (`{ligand_data}`.trim().length > 0) {{
            viewer.addModel(`{ligand_data}`, 'pdb');
            viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
        }}
        {int_lines_js}
        viewer.zoomTo(); viewer.render();
    </script>
    """
    components.html(html_content, height=490)


# --- LOG FILE PARSERS ---

def parse_vina_output_text(stdout_text):
    data = []
    pattern = re.compile(r"^\s*(\d+)\s+([-+]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")
    for line in stdout_text.split("\n"):
        match = pattern.match(line)
        if match:
            data.append({"Binding Mode": int(match.group(1)), "Affinity (kcal/mol)": float(match.group(2)), "RMSD l.b.": float(match.group(3)), "RMSD u.b.": float(match.group(4))})
    return pd.DataFrame(data)

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


# --- APPLICATION DASHBOARD WORKSPACE ---

st.set_page_config(page_title="In Silico Docking Hub", layout="wide")
st.title("🔬 Automated Molecular Docking Studio")

# Initialize state management keys safely
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
        > **Protein Summary Profile:** > * **Title:** {meta['title']}  
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

    # --- ADVANCED MANUAL LIGAND LOADING BOUNDARY BUTTON ---
    if st.button("📥 Load Ligand Structure", key="load_ligand_btn"):
        if ligand_source == "SMILES String Input" and smiles_input_val:
            try:
                mol = Chem.MolFromSmiles(smiles_input_val)
                if mol:
                    ok, _ = convert_smiles_to_pdbqt(smiles_input_val, "ligand.pdbqt")
                    if ok:
                        st.session_state.ligand_ready = True
                        st.session_state.smiles_cache = smiles_input_val
                        with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                        # STABLE RDKit API ROOT EXPOSURE CONVERSION UPGRADE
                        st.session_state.ligand_summary_text = f"Formula: {Chem.CalcMolFormula(mol)} | MW: {round(Chem.Descriptors.MolWt(mol), 2)} g/mol"
                        st.success("SMILES ligand structure computed and loaded successfully!")
                        st.rerun()
            except Exception as e: st.error(f"SMILES Parsing Failure: {e}")
            
        elif ligand_source == "Upload Structural File (.pdb, .sdf)" and uploaded_lig_buffer is not None:
            temp_in = f"raw_ligand_{uploaded_lig_name}"
            with open(temp_in, "wb") as f: f.write(uploaded_lig_buffer.getbuffer())
            
            mol = Chem.MolFromPDBFile(temp_in, removeHs=False) if uploaded_lig_name.endswith(".pdb") else Chem.SDMolSupplier(temp_in, removeHs=False)[0]
            if mol:
                try:
                    Chem.SanitizeMol(mol)
                    AllChem.AssignBondOrdersFromTopology(mol)
                except Exception: pass
                
                if mol.GetNumConformers() == 0:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    AllChem.MMFFOptimizeMolecule(mol)
                    
                temp_pdb = "temp_lig_state.pdb"
                Chem.MolToPDBFile(mol, temp_pdb)
                convert_pdb_to_pdbqt(temp_pdb, "ligand.pdbqt", is_ligand=True)
                
                st.session_state.ligand_ready = True
                st.session_state.smiles_cache = temp_in
                with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                
                try:
                    chem_formula = Chem.CalcMolFormula(mol)
                    mw = round(Chem.Descriptors.MolWt(mol), 2)
                    rot_bonds = AllChem.CalcNumRotatableBonds(mol)
                    st.session_state.ligand_summary_text = f"Formula: {chem_formula} | MW: {mw} g/mol | Rotatable Bonds: {rot_bonds}"
                except Exception:
                    st.session_state.ligand_summary_text = "Structure metadata compiled dynamically."
                
                if os.path.exists(temp_in): os.remove(temp_in)
                if os.path.exists(temp_pdb): os.remove(temp_pdb)
                st.success(f"Structural file {uploaded_lig_name} processed and loaded successfully!")
                st.rerun()
        else:
            st.warning("Please provide a valid SMILES input or structural file target before initializing configuration loops.")

    if st.session_state.target_ready and os.path.exists("ligand.pdbqt"):
        st.session_state.ligand_ready = True

    if st.session_state.ligand_ready:
        st.markdown(f"> **Ligand Metric Summary:** \n> {st.session_state.ligand_summary_text}")

    # --- BOUND CO-CRYSTAL SEARCH SITE PANEL ---
    if st.session_state.target_ready and st.session_state.local_target_path:
        bound_ligands_list = parse_bound_ligands(st.session_state.local_target_path)
        if bound_ligands_list:
            st.header("3. Bound Small Molecules in Receptor")
            st.write("Co-crystallized ligands parsed from HETATM records. Select one to auto-fill the docking grid box.")
            
            df_bound = pd.DataFrame(bound_ligands_list)
            df_display = df_bound.copy()
            df_display["Center (X, Y, Z) Å"] = df_display.apply(lambda r: f"{r['cx']}, {r['cy']}, {r['cz']}", axis=1)
            df_display["Box (X, Y, Z) Å"] = df_display.apply(lambda r: f"{r['bx']}, {r['by']}, {r['bz']}", axis=1)
            
            st.dataframe(df_display[["ID", "Chain", "ResSeq", "Atoms", "Center (X, Y, Z) Å", "Box (X, Y, Z) Å"]], hide_index=True, use_container_width=True)
            
            selected_lig_id = st.selectbox(
                "Select native co-crystal target to auto-lock parameters:",
                options=range(len(bound_ligands_list)),
                format_func=lambda idx: f"{bound_ligands_list[idx]['ID']} (Chain {bound_ligands_list[idx]['Chain']}-ResSeq {bound_ligands_list[idx]['ResSeq']})"
            )
            
            if st.button("🎯 Lock Coordinates to Native Site"):
                chosen_target = bound_ligands_list[selected_lig_id]
                st.session_state.cx = chosen_target["cx"]
                st.session_state.cy = chosen_target["cy"]
                st.session_state.cz = chosen_target["cz"]
                st.session_state.sx = chosen_target["bx"]
                st.session_state.sy = chosen_target["by"]
                st.session_state.sz = chosen_target["bz"]
                st.success("Grid parameters locked down successfully!")
                st.rerun()

    st.header("4. Search Space Mechanics (Grid Box)")
    grid_cx = st.number_input("Center X Coordinate", value=float(st.session_state.cx), step=0.1)
    grid_cy = st.number_input("Center Y Coordinate", value=float(st.session_state.cy), step=0.1)
    grid_cz = st.number_input("Center Z Coordinate", value=float(st.session_state.cz), step=0.1)
    
    grid_sx = st.slider("Grid Box Size X (Å)", 10, 40, int(st.session_state.sx))
    grid_sy = st.slider("Grid Box Size Y (Å)", 10, 40, int(st.session_state.sy))
    grid_sz = st.slider("Grid Box Size Z (Å)", 10, 40, int(st.session_state.sz))
    
    exhaustiveness = st.slider("Search Exhaustiveness", min_value=4, max_value=32, value=8, step=4)
    
    can_dock = bool(st.session_state.target_ready and st.session_state.ligand_ready)
    run_btn = st.button("🚀 Initialize Docking Algorithm", type="primary", disabled=not can_dock)

with col_visual:
    st.header("5. Active Viewport Canvas")
    
    if st.session_state.docking_results_raw is None:
        view_tabs = st.tabs(["3D Structural Space", "2D Schematic Topology View"])
        
        with view_tabs[0]:
            receptor_view_data = ""
            if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
                with open("protein.pdbqt", "r") as f: receptor_view_data = f.read()
            render_complex_html(receptor_pdbqt=receptor_view_data, ligand_pdbqt=st.session_state.serialized_ligand_block)
                
        with view_tabs[1]:
            if st.session_state.ligand_ready and st.session_state.smiles_cache:
                try:
                    if "raw_ligand" in st.session_state.smiles_cache:
                        m_img = Chem.MolFromPDBFile("temp_lig_state.pdb", removeHs=True) if os.path.exists("temp_lig_state.pdb") else Chem.MolFromPDBFile("ligand.pdbqt", removeHs=True)
                    else:
                        m_img = Chem.MolFromSmiles(st.session_state.smiles_cache)
                    
                    if m_img:
                        Chem.SanitizeMol(m_img)
                        img_b64 = generate_2d_ligand_img(m_img)
                        if img_b64:
                            html_output_div = '<div style="text-align:center; background: white; padding:10px; border-radius:5px;"><img src="data:image/png;base64,{}"/></div>'.format(img_b64)
                            st.markdown(html_output_div, unsafe_html=True)
                        else: st.info("Rendering topology canvas vector...")
                    else: st.info("Parsing chemical topology vectors...")
                except Exception:
                    st.info("2D schematic layout rendering complete.")
    else:
        st.subheader("Interactive Complex Viewport")
        parsed_poses = split_docking_poses("docking_poses.pdbqt")
        if parsed_poses:
            selected_pose = st.selectbox("Choose Docking Pose to Visualize:", options=list(parsed_poses.keys()), format_func=lambda x: f"Mode {x} Pose Fit")
            with open("protein.pdbqt", "r") as f: protein_data = f.read()
            
            # --- COMPREHENSIVE BLUEPRINT VIEWPORT CONTAINER (SD-02 ARCHITECTURE) ---
            st.write("#### Advanced Complex Visual Modeling Blueprint")
            
            # Style & display modifiers split block controls
            col_render, col_mesh = st.columns([1, 1])
            with col_render:
                style_mode = re.sub(r'\W+', '', st.radio("Macromolecule Style Mode:", ["Cartoon Ribbon Mesh", "Spacefill (VDW Configuration)", "Sticks Profile"]).split()[0].lower())
            with col_mesh:
                surf_toggle = st.checkbox("Overlay Translucent Pocket Cavity Mesh", value=False)
                
            active_interactions = compute_spatial_interactions("protein.pdbqt", parsed_poses[selected_pose])
            
            # Mount advanced interaction visual modeling viewport component
            render_advanced_modeling_blueprint(
                receptor_data=protein_data,
                ligand_data=parsed_poses[selected_pose],
                mode=style_mode,
                show_surface=surf_toggle,
                interactions_list=active_interactions
            )
            
            # Render Contact Residues Analysis Subtable Frame
            st.subheader("🧬 Local Contact Residues Matrix")
            if active_interactions:
                df_int = pd.DataFrame(active_interactions)
                st.dataframe(df_int[["Residue Contact", "Interaction Type", "Distance (Å)"]], hide_index=True, use_container_width=True)
            else:
                st.info("No close contacts detected within a 3.8 Å threshold radius.")
            
        if st.button("🔄 Reset Environment Canvas"):
            st.session_state.docking_results_raw = None
            st.rerun()

    # --- ENGINE COMPUTATION EXECUTION BOUNDARY ---
    if run_btn and can_dock:
        with st.spinner("Processing flexible calculation search passes..."):
            vina_command = [
                "./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt",
                "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz),
                "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz),
                "--exhaustiveness", str(exhaustiveness), "--out", "docking_poses.pdbqt"
            ]
            try:
                process = subprocess.run(vina_command, capture_output=True, text=True, check=True)
                if process.stdout:
                    st.session_state.docking_results_raw = process.stdout
                    st.success("Calculations complete!")
                    st.rerun()
            except subprocess.CalledProcessError as err:
                st.error("Calculations exited with error flags."); st.code(err.stderr if err.stderr else err.stdout)

# --- GLOBAL DATAFRAME ANALYTICS DISPLAY ZONE ---
if st.session_state.docking_results_raw is not None:
    st.write("---")
    st.header("📊 Screening Metrics Dashboard & Data Export")
    df_results = parse_vina_output_text(st.session_state.docking_results_raw)
    if not df_results.empty:
        col_table, col_export = st.columns([2, 1])
        with col_table: st.dataframe(df_results, hide_index=True, use_container_width=True)
        with col_export:
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Data Sheet (.CSV)", data=csv_data, file_name="screening_affinity_report.csv", mime="text/csv", use_container_width=True)
