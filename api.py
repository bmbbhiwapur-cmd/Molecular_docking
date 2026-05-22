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


# --- PDB METADATA & BOUND LIGANDS PARSERS ---

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
                    x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                except ValueError: continue
                if key not in ligands: ligands[key] = {"res": res_name, "chain": chain_id, "seq": res_seq, "coords": []}
                ligands[key]["coords"].append((x, y, z))
                
    processed = []
    for key, info in ligands.items():
        pts = info["coords"]
        n_atoms = len(pts)
        if n_atoms < 4: continue
        cx, cy, cz = sum([p[0] for p in pts])/n_atoms, sum([p[1] for p in pts])/n_atoms, sum([p[2] for p in pts])/n_atoms
        bx, by, bz = max([p[0] for p in pts])-min([p[0] for p in pts])+10.0, max([p[1] for p in pts])-min([p[1] for p in pts])+10.0, max([p[2] for p in pts])-min([p[2] for p in pts])+10.0
        processed.append({"ID": info["res"], "Chain": info["chain"], "ResSeq": info["seq"], "Atoms": n_atoms, "cx": round(cx,2), "cy": round(cy,2), "cz": round(cz,2), "bx": round(bx,1), "by": round(by,1), "bz": round(bz,1)})
    return processed


# --- ADVANCED CHEMICAL INTERACTION PARSER ENGINE ---

def parse_pdbqt_coordinates(pdbqt_string):
    """Extracts atom entries and 3D coordinates from a PDBQT string data block."""
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
    """Calculates distances between ligand and protein atoms to determine binding types."""
    interactions = []
    if not os.path.exists(receptor_file): return interactions
    
    with open(receptor_file, "r") as f:
         receptor_atoms = parse_pdbqt_coordinates(f.read())
    ligand_atoms = parse_pdbqt_coordinates(ligand_pdbqt_str)
    
    seen = set()
    for l_at in ligand_atoms:
        for r_at in receptor_atoms:
            dist = np.linalg.norm(l_at["coord"] - r_at["coord"])
            if dist < 3.8: # Threshold under 3.8 Å indicates an active non-covalent bond
                res_id = r_at["res"]
                if res_id in seen: continue
                
                # Determine interaction type based on elemental properties
                if l_at["element"] in ["N", "O", "F"] and r_at["element"] in ["N", "O", "F"]:
                    b_type = "Hydrogen Bond (H-Bond)"
                elif "C" in l_at["element"] and ("C" in r_at["element"] or "A" in r_at["element"]):
                    b_type = "pi-Stacking / Hydrophobic"
                else:
                    b_type = "Electrostatic / van der Waals"
                    
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

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False):
    autodock_type_map = {"H": "H", "HD": "HD", "HS": "HS", "C": "C", "A": "A", "N": "N", "NA": "NA", "NS": "NS", "O": "O", "OA": "OA", "S": "S", "SA": "SA", "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I", "ZN": "Zn", "MG": "Mg"}
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


# --- HIGH PERFORMANCE PY3DMOL LAYOUT VISUALIZATIONS ---

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

def render_dynamic_docking_html(receptor_data, ligand_data, mode="stick", show_surface=False, interactions_list=[]):
    """Generates an online interactive 3D viewport canvas supporting advanced style toggles."""
    surface_js = "viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.65, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if show_surface else ""
    
    # Generate JavaScript vectors to draw non-covalent interactions as dashed lines
    int_lines_js = ""
    for idx, interact in enumerate(interactions_list):
        rc = interact["r_coord"]
        lc = interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js += f"""
        viewer.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.06, color:'{color}', dashed:true}});
        viewer.addLabel("{interact['Residue Contact']}", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.7, fontSize:10}});
        """

    html_content = f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <div id="container" style="height: 440px; width: 100%; position: relative; border-radius:8px; border:1px solid #ddd;"></div>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: '#ffffff'}});
        
        // Model 0: Receptor Protein
        viewer.addModel(`{receptor_data}`, 'pdb');
        if ('{mode}' === 'cartoon') {{
            viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'spectrum'}}}});
        }} else if ('{mode}' === 'spacefill') {{
            viewer.setStyle({{model: 0}}, {{sphere: {{radius: 1.0, colorscheme: 'spectrum'}}}});
        }} else {{
            viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'spectrum'}}}});
        }}
        
        {surface_js}
        
        // Model 1: Dynamic Bound Ligand
        if (`{ligand_data}`.trim().length > 0) {{
            viewer.addModel(`{ligand_data}`, 'pdb');
            viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'cyanCarbon', radius: 0.25}}}});
        }}
        
        {int_lines_js}
        
        viewer.zoomTo(); viewer.render();
    </script>
    """
    components.html(html_content, height=450)


# --- APPLICATION DASHBOARD WORKSPACE ---

st.set_page_config(page_title="In Silico Docking Hub", layout="wide")
st.title("🔬 Automated Molecular Docking Studio (CB-Dock2 Architecture)")

# Persistent memory state setups
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

# --- EXPERIMENTAL TAB WINDOW ARRAYS ---
window_tabs = st.tabs(["🏛 Target Protein Profile", "💊 Small Molecule Ligand Profiler", "⚙ Computational Grid Box Settings"])

# --- WINDOW 1: PROTEIN CONTROL FRAMEWORK ---
with window_tabs[0]:
    st.subheader("Target Macromolecule Workspace Configuration")
    protein_source = st.radio("Choose Protein Input Method:", ["Type 4-Letter PDB ID", "Upload File (.pdb or .pdbqt)"], key="prot_src_radio")
    
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
                    st.success(f"Protein {pdb_id_input.upper()} successfully initialized!")
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
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.markdown(f"""
            ### 📊 Structural Header Summary Card
            *   **Experimental Title:** {meta['title']}  
            *   **Target Repository ID:** `{meta['id']}`  
            *   **Classification Entry:** *{meta['class']}*  
            *   **Organism Profile:** **{meta['organism']}**  
            *   **Expression Host System:** `{meta['system']}`  
            *   **Resolution Parameter Value:** `{meta['res']}`
            """)
        with col_m2:
            st.write("### Isolated 3D Protein Mesh Visualization")
            with open("protein.pdbqt", "r") as f: p_data = f.read()
            render_dynamic_docking_html(receptor_data=p_data, ligand_data="", mode="cartoon")

# --- WINDOW 2: LIGAND CONTROL FRAMEWORK ---
with window_tabs[1]:
    st.subheader("Small Molecule Chemical Profiler")
    ligand_source = st.radio("Choose Ligand Input Method:", ["SMILES String Input", "Upload Structural File (.pdb, .sdf)"], key="lig_src_radio")
    
    if ligand_source == "SMILES String Input":
        smiles_input = st.text_input("Enter Ligand SMILES String", "CC(=O)NC1=CC=C(O)C=C1")
        if smiles_input and smiles_input != st.session_state.smiles_cache:
            try:
                mol = Chem.MolFromSmiles(smiles_input)
                if mol:
                    ok, _ = convert_smiles_to_pdbqt(smiles_input, "ligand.pdbqt")
                    if ok:
                        st.session_state.ligand_ready = True
                        st.session_state.smiles_cache = smiles_input
                        with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                        st.session_state.ligand_summary_text = f"Formula: {Chem.rdmolops.CalcMolFormula(mol)} | MW: {round(Chem.Descriptors.MolWt(mol), 2)} g/mol"
                        st.rerun()
            except Exception: pass
    else:
        uploaded_lig_file = st.file_uploader("Upload Small Molecule File", type=["pdb", "sdf"])
        if uploaded_lig_file:
            temp_in = f"raw_ligand_{uploaded_lig_file.name}"
            if st.session_state.smiles_cache != temp_in:
                with open(temp_in, "wb") as f: f.write(uploaded_lig_file.getbuffer())
                mol = Chem.MolFromPDBFile(temp_in, removeHs=False) if uploaded_lig_file.name.endswith(".pdb") else Chem.SDMolSupplier(temp_in, removeHs=False)[0]
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
                    st.session_state.ligand_summary_text = f"Formula: {Chem.rdmolops.CalcMolFormula(mol)} | MW: {round(Chem.Descriptors.MolWt(mol), 2)} g/mol | Rotatable Bonds: {AllChem.CalcNumRotatableBonds(mol)}"
                    if os.path.exists(temp_in): os.remove(temp_in)
                    if os.path.exists(temp_pdb): os.remove(temp_pdb)
                    st.rerun()

    if os.path.exists("ligand.pdbqt") and st.session_state.target_ready: st.session_state.ligand_ready = True

    if st.session_state.ligand_ready:
        st.info(st.session_state.ligand_summary_text)
        col_l2d, col_l3d = st.columns([1, 1])
        with col_l2d:
            st.write("#### 2D Chemical Blueprint Structure")
            m_img = Chem.MolFromPDBFile("ligand.pdbqt", removeHs=True) if "raw_ligand" in st.session_state.smiles_cache else Chem.MolFromSmiles(st.session_state.smiles_cache)
            img_b64 = generate_2d_ligand_img(m_img)
            if img_b64: st.markdown(f'<div style="text-align:center; background: white; padding:10px; border-radius:8px;"><img src="data:image/png;base64,{img_b64}"/></div>', unsafe_html=True)
        with col_l3d:
            st.write("#### Isolated 3D Chemical Sticks")
            render_dynamic_docking_html(receptor_data="", ligand_data=st.session_state.serialized_ligand_block, mode="stick")

# --- WINDOW 3: AUTOMATED BOUND POCKET GRID AUTO-LOCK SYSTEM ---
with window_tabs[2]:
    st.subheader("Grid Parameter Search Workspace Settings")
    if st.session_state.target_ready and st.session_state.local_target_path:
        bound_ligands_list = parse_bound_ligands(st.session_state.local_target_path)
        if bound_ligands_list:
            st.write("### 🎯 Co-crystallized Receptor Active Sites Detected")
            df_bound = pd.DataFrame(bound_ligands_list)
            df_display = df_bound.copy()
            df_display["Center (X,Y,Z)"] = df_display.apply(lambda r: f"{r['cx']}, {r['cy']}, {r['cz']}", axis=1)
            df_display["Box Bounds (X,Y,Z)"] = df_display.apply(lambda r: f"{r['bx']}, {r['by']}, {r['bz']}", axis=1)
            st.dataframe(df_display[["ID", "Chain", "ResSeq", "Atoms", "Center (X,Y,Z)", "Box Bounds (X,Y,Z)"]], hide_index=True, use_container_width=True)
            
            selected_lig_id = st.selectbox("Select site cavity target to auto-lock parameters:", options=range(len(bound_ligands_list)), format_func=lambda idx: f"Pocket site: {bound_ligands_list[idx]['ID']} (Chain {bound_ligands_list[idx]['Chain']}-ResSeq {bound_ligands_list[idx]['ResSeq']})")
            if st.button("🎯 Auto-Lock Parameters To Selection"):
                chosen = bound_ligands_list[selected_lig_id]
                st.session_state.cx, st.session_state.cy, st.session_state.cz = chosen["cx"], chosen["cy"], chosen["cz"]
                st.session_state.sx, st.session_state.sy, st.session_state.sz = chosen["bx"], chosen["by"], chosen["bz"]
                st.success("Grid inputs dynamically aligned over target bounding box pocket cavity matrix.")
                st.rerun()

    # Core Parameter Layout adjustments
    col_g1, col_g2 = st.columns([1, 1])
    with col_g1:
        grid_cx = st.number_input("Center X Coordinate", value=float(st.session_state.cx), step=0.1)
        grid_cy = st.number_input("Center Y Coordinate", value=float(st.session_state.cy), step=0.1)
        grid_cz = st.number_input("Center Z Coordinate", value=float(st.session_state.cz), step=0.1)
    with col_g2:
        grid_sx = st.slider("Grid Box Size X (Å)", 10, 40, int(st.session_state.sx))
        grid_sy = st.slider("Grid Box Size Y (Å)", 10, 40, int(st.session_state.sy))
        grid_sz = st.slider("Grid Box Size Z (Å)", 10, 40, int(st.session_state.sz))
        
    exhaustiveness = st.slider("Global Computational Search Exhaustiveness", min_value=4, max_value=32, value=8, step=4)
    can_dock = bool(st.session_state.target_ready and st.session_state.ligand_ready)
    run_btn = st.button("🚀 Run Molecular Docking Job", type="primary", disabled=not can_dock, use_container_width=True)


# --- WINDOW 4: POST-DOCKING SCREENING ANALYSIS VIEWPORT HUB ---
if st.session_state.docking_results_raw is not None:
    st.write("---")
    st.header("🏁 Screening Metrics Workspace Dashboard & 3D Interactive Viewer")
    
    col_view3d, col_table_metrics = st.columns([1, 1])
    parsed_poses = split_docking_poses("docking_poses.pdbqt")
    
    with col_table_metrics:
        st.subheader("📊 Vina Binding Modes Scoring Matrix")
        df_results = parse_vina_output_text(st.session_state.docking_results_raw)
        st.dataframe(df_results, hide_index=True, use_container_width=True)
        
        # Free Download Data actions
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Affinity Excel Report Sheet (.CSV)", data=csv_data, file_name="screening_affinity_report.csv", mime="text/csv", use_container_width=True)
        
        # Setup active toggle switch parameters for the 3D visual engine controls
        selected_pose = st.selectbox("Select Target Mode Pose to Profile:", options=list(parsed_poses.keys()), format_func=lambda x: f"Mode {x} Pose Binding Affinity Alignment")
        
        # Real-time Display Modifiers
        st.subheader("🎨 Viewport Rendering Controls")
        style_mode = re.sub(r'\W+', '', st.radio("Macromolecule Display Representation Mode:", ["Cartoon Backbone Mesh", "Spacefill (VDW Surface Profile)"]).split()[0].lower())
        surf_toggle = st.checkbox("Superimpose Translucent Binding Pocket Cavity Surface Mesh", value=False)
        
        # --- DYNAMIC INTERACTION TABLE PARSER ---
        if selected_pose in parsed_poses:
            st.subheader("🧬 Local Pocket Contact Residues Matrix")
            interactions = compute_spatial_interactions("protein.pdbqt", parsed_poses[selected_pose])
            if interactions:
                df_int = pd.DataFrame(interactions)
                st.dataframe(df_int[["Residue Contact", "Interaction Type", "Distance (Å)"]], hide_index=True, use_container_width=True)
            else:
                st.info("No active residue contacts detected within a 3.8 Å spatial radius bounding box threshold.")
                
        if st.button("🔄 Clear System State and Reset Canvas", type="secondary", use_container_width=True):
            st.session_state.docking_results_raw = None
            st.rerun()

    with col_view3d:
        st.subheader("3D Binding Pocket Structural Workspace")
        if selected_pose in parsed_poses:
            with open("protein.pdbqt", "r") as f: receptor_data_string = f.read()
            active_interactions = compute_spatial_interactions("protein.pdbqt", parsed_poses[selected_pose])
            
            # Send values over to the javascript script execution layer
            render_dynamic_docking_html(
                receptor_data=receptor_data_string, 
                ligand_data=parsed_poses[selected_pose], 
                mode=style_mode, 
                show_surface=surf_toggle,
                interactions_list=active_interactions
            )

# --- BACKEND COMPILATION TRIGGER LOOP THREADS ---
if run_btn and can_dock:
    with st.spinner("Processing flexible cloud computing binding search passes..."):
        vina_command = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt", "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), "--exhaustiveness", str(exhaustiveness), "--out", "docking_poses.pdbqt"]
        try:
            process = subprocess.run(vina_command, capture_output=True, text=True, check=True)
            if process.stdout:
                st.session_state.docking_results_raw = process.stdout
                st.success("Calculations complete!")
                st.rerun()
        except subprocess.CalledProcessError as err:
            st.error("Calculations exited with error flags."); st.code(err.stderr if err.stderr else err.stdout)
