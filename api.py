import streamlit as st
import subprocess
import os
import urllib.request
import re
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
    """Parses structural header records to fetch experimental metadata."""
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
    """Scans HETATM records to isolate native bound ligands and map bounding coordinates."""
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
                
                # Exclude standard crystallographic solvent water molecules
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
        if n_atoms < 4: continue  # Filter out standalone buffer ions (e.g. Cl, Mg)
        
        # Calculate geometric bounding center
        cx = sum([p[0] for p in pts]) / n_atoms
        cy = sum([p[1] for p in pts]) / n_atoms
        cz = sum([p[2] for p in pts]) / n_atoms
        
        # Determine appropriate search space boundaries based on atom distances
        bx = max([p[0] for p in pts]) - min([p[0] for p in pts]) + 10.0
        by = max([p[1] for p in pts]) - min([p[1] for p in pts]) + 10.0
        bz = max([p[2] for p in pts]) - min([p[2] for p in pts]) + 10.0
        
        processed_ligands.append({
            "ID": info["res"], "Chain": info["chain"], "ResSeq": info["seq"], "Atoms": n_atoms,
            "cx": round(cx, 2), "cy": round(cy, 2), "cz": round(cz, 2),
            "bx": round(bx, 1), "by": round(by, 1), "bz": round(bz, 1)
        })
    return processed_ligands


# --- CHEMINFORMATICS COORDINATE PARSERS ---

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

def process_uploaded_ligand(file_buffer, filename):
    """Processes uploaded small-molecule files across PDB/SDF formats into 3D environments."""
    try:
        temp_in = f"raw_ligand_{filename}"
        with open(temp_in, "wb") as f: f.write(file_buffer.getbuffer())
        
        if filename.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(temp_in, removeHs=False)
        elif filename.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(temp_in, removeHs=False)
            mol = suppl[0] if suppl else None
        else: return False, None, "Unsupported structural file layout format."
        
        if mol is None: return False, None, "RDKit structure parsing execution trace failed."
        
        # Embed and optimize 3D conformations if source coordinates are flat
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            
        temp_pdb = "temp_lig_out.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
        convert_pdb_to_pdbqt(temp_pdb, "ligand.pdbqt", is_ligand=True)
        
        # Extract metadata metrics
        chem_formula = Chem.rdmolops.CalcMolFormula(mol)
        mw = round(Chem.Descriptors.MolWt(mol), 2)
        summary = f"Formula: {chem_formula} | Molecular Weight: {mw} g/mol | Rotatable Bonds: {AllChem.CalcNumRotatableBonds(mol)}"
        
        if os.path.exists(temp_in): os.remove(temp_in)
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        return True, mol, summary
    except Exception as e: return False, None, str(e)


# --- RENDER ENGRAVING IMAGES BLOCK ---

def generate_2d_ligand_img(mol):
    """Generates a base64 encoded chemical image to embed directly inline into data layouts."""
    try:
        img = Draw.MolToImage(mol, size=(320, 240))
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
        viewer.addModel(`{receptor_pdbqt}`, 'pdb');
        viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'spectrum'}}}});
        {ligand_block}
        viewer.zoomTo(); viewer.render();
    </script>
    """
    components.html(html_content, height=390)


# --- PARSING PARAMS LOGS ---

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


# --- WEB RUNTIME INTERFACE ---

st.set_page_config(page_title="In Silico Docking Hub", layout="wide")
st.title("🔬 Automated Molecular Docking Studio")

# Session State initializations
if "cx" not in st.session_state: st.session_state.cx = 0.0
if "cy" not in st.session_state: st.session_state.cy = 0.0
if "cz" not in st.session_state: st.session_state.cz = 0.0
if "sx" not in st.session_state: st.session_state.sx = 20
if "sy" not in st.session_state: st.session_state.sy = 20
if "sz" not in st.session_state: st.session_state.sz = 20
if "docking_results_raw" not in st.session_state: st.session_state.docking_results_raw = None

col_params, col_visual = st.columns([1, 1])
target_ready, ligand_ready = False, False
prepared_receptor_path = "protein.pdbqt"
active_ligand_mol = None
ligand_summary_text = ""

with col_params:
    st.header("1. Target Protein Setup")
    protein_source = st.radio("Choose Protein Input Method:", ["Type 4-Letter PDB ID", "Upload File (.pdb or .pdbqt)"])
    
    pdb_file_target = None
    if protein_source == "Type 4-Letter PDB ID":
        pdb_id_input = st.text_input("Enter RCSB PDB ID", value="2AMB").strip()
        if pdb_id_input:
            fetch_success, pdb_file_target = fetch_pdb_from_rcsb(pdb_id_input)
            if fetch_success: target_ready, _ = convert_pdb_to_pdbqt(pdb_file_target, prepared_receptor_path)
    else:
        uploaded_file = st.file_uploader("Upload Target Protein File", type=["pdb", "pdbqt"])
        if uploaded_file:
            pdb_file_target = f"uploaded_{uploaded_file.name}"
            with open(pdb_file_target, "wb") as f: f.write(uploaded_file.getbuffer())
            if uploaded_file.name.endswith(".pdb"): target_ready, _ = convert_pdb_to_pdbqt(pdb_file_target, prepared_receptor_path)
            else: os.replace(pdb_file_target, prepared_receptor_path); target_ready = True; pdb_file_target = None

    # Render Protein Metadata Card
    if target_ready and pdb_file_target:
        meta = extract_pdb_metadata(pdb_file_target, pdb_id_input if protein_source == "Type 4-Letter PDB ID" else "Upload")
        st.markdown(f"""
        > **Protein Summary Profile:**  
        > *   **Title:** {meta['title']}  
        > *   **PDB ID:** `{meta['id']}` | **Classification:** {meta['class']}  
        > *   **Organism(s):** *{meta['organism']}* | **Expression System:** {meta['system']}  
        > *   **Experimental Method:** {meta['method']} | **Resolution:** **{meta['res']}**
        """)

    st.header("2. Small Molecule Ligand Setup")
    ligand_source = st.radio("Choose Ligand Input Method:", ["SMILES String Input", "Upload Structural File (.pdb, .sdf)"])
    
    if ligand_source == "SMILES String Input":
        smiles_input = st.text_input("Enter Ligand SMILES String", "CC(=O)NC1=CC=C(O)C=C1")
        if smiles_input:
            try:
                active_ligand_mol = Chem.MolFromSmiles(smiles_input)
                if active_ligand_mol:
                    ligand_ready, _ = convert_smiles_to_pdbqt(smiles_input)
                    ligand_summary_text = f"Formula: {Chem.rdmolops.CalcMolFormula(active_ligand_mol)} | Molecular Weight: {round(Chem.Descriptors.MolWt(active_ligand_mol), 2)} g/mol"
            except Exception: pass
    else:
        uploaded_lig_file = st.file_uploader("Upload Small Molecule File", type=["pdb", "sdf"])
        if uploaded_lig_file:
            ligand_ready, active_ligand_mol, ligand_summary_text = process_uploaded_ligand(uploaded_lig_file, uploaded_lig_file.name)

    if ligand_ready and active_ligand_mol:
        st.markdown(f"> **Ligand Metric Summary:**  \n> {ligand_summary_text}")

    # --- BOUND SMALL MOLECULES INTERACTIVE PARSER PANEL ---
    if target_ready and pdb_file_target:
        bound_ligands_list = parse_bound_ligands(pdb_file_target)
        if bound_ligands_list:
            st.header("3. Bound Small Molecules in Receptor")
            st.write("Co-crystallized ligands parsed from HETATM records. Select one to auto-fill the docking grid box at its native binding site.")
            
            # Construct display table rows matching requested layout
            df_bound = pd.DataFrame(bound_ligands_list)
            df_display = df_bound.copy()
            df_display["Center (X, Y, Z) Å"] = df_display.apply(lambda r: f"{r['cx']}, {r['cy']}, {r['cz']}", axis=1)
            df_display["Box (X, Y, Z) Å"] = df_display.apply(lambda r: f"{r['bx']}, {r['by'], r['bz']}", axis=1)
            
            st.dataframe(df_display[["ID", "Chain", "ResSeq", "Atoms", "Center (X, Y, Z) Å", "Box (X, Y, Z) Å"]], hide_index=True, use_container_width=True)
            
            # Dropdown choice selection to trigger grid autofill parameters
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
                st.success("Grid parameters aligned to match pocket bounding box dimensions!")

    st.header("4. Search Space Mechanics (Grid Box)")
    grid_cx = st.number_input("Center X Coordinate", value=st.session_state.cx, step=0.1)
    grid_cy = st.number_input("Center Y Coordinate", value=st.session_state.cy, step=0.1)
    grid_cz = st.number_input("Center Z Coordinate", value=st.session_state.cz, step=0.1)
    
    grid_sx = st.slider("Grid Box Size X (Å)", 10, 40, int(st.session_state.sx))
    grid_sy = st.slider("Grid Box Size Y (Å)", 10, 40, int(st.session_state.sy))
    grid_sz = st.slider("Grid Box Size Z (Å)", 10, 40, int(st.session_state.sz))
    
    exhaustiveness = st.slider("Search Exhaustiveness", min_value=4, max_value=32, value=8, step=4)
    run_btn = st.button("🚀 Initialize Docking Algorithm", type="primary", disabled=not (target_ready and ligand_ready))

with col_visual:
    st.header("5. Active Viewport Canvas")
    
    if st.session_state.docking_results_raw is None:
        view_tabs = st.tabs(["3D Target Molecular Space", "2D Chemical Structure Matrix"])
        
        with view_tabs[0]:
            if target_ready:
                with open(prepared_receptor_path, "r") as f: receptor_data = f.read()
                ligand_data_str = None
                if ligand_ready:
                    with open("ligand.pdbqt", "r") as f: ligand_data_str = f.read()
                render_complex_html(receptor_pdbqt=receptor_data, ligand_pdbqt=ligand_data_str)
                
        with view_tabs[1]:
            if ligand_ready and active_ligand_mol:
                img_b64 = generate_2d_ligand_img(active_ligand_mol)
                if img_b64: st.markdown(f'<div style="text-align:center;"><img src="data:image/png;base64,{img_b64}"/></div>', unsafe_html=True)
                else: st.info("2D schematic view rendering unavailable for this asset style topology.")
    else:
        st.subheader("Interactive Complex Viewport")
        parsed_poses = split_docking_poses("docking_poses.pdbqt")
        if parsed_poses:
            selected_pose = st.selectbox("Choose Docking Pose to Visualize:", options=list(parsed_poses.keys()), format_func=lambda x: f"Mode {x} Pose Fit")
            with open(prepared_receptor_path, "r") as f: protein_data = f.read()
            render_complex_html(receptor_pdbqt=protein_data, ligand_pdbqt=parsed_poses[selected_pose])
            
        if st.button("🔄 Reset Environment Canvas"):
            st.session_state.docking_results_raw = None
            st.rerun()

    # --- ENGINE COMPACTION THREAD RUNNER ---
    if run_btn and target_ready and ligand_ready:
        with st.spinner("Processing flexible computational binding space matching passes..."):
            vina_command = [
                "./vina", "--receptor", prepared_receptor_path, "--ligand", "ligand.pdbqt",
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

# --- INLINE SCREENING SUB-TABLE RENDER MATRIX PANEL ---
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
