import streamlit as st
import subprocess
import os
import urllib.request
import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import streamlit.components.v1 as components

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


# --- PARSING UTILITIES FOR RESULTS & POSES ---

def parse_vina_output_text(stdout_text):
    """
    Parses the text output table directly from Vina's stdout 
    and converts it into a clean, structured Pandas DataFrame.
    """
    data = []
    # Regular expression pattern to capture rows like: "   1         -7.4      0.000      0.000"
    pattern = re.compile(r"^\s*(\d+)\s+([-+]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")
    
    for line in stdout_text.split("\n"):
        match = pattern.match(line)
        if match:
            mode = int(match.group(1))
            affinity = float(match.group(2))
            rmsd_lb = float(match.group(3))
            rmsd_ub = float(match.group(4))
            data.append({
                "Binding Mode": mode,
                "Affinity (kcal/mol)": affinity,
                "RMSD Lower Bound": rmsd_lb,
                "RMSD Upper Bound": rmsd_ub
            })
    return pd.DataFrame(data)

def split_docking_poses(poses_file_path):
    """
    Splits the multi-model docking_poses.pdbqt output file into 
    separate individual text frames for pose-by-pose 3D visualization.
    """
    poses = {}
    if not os.path.exists(poses_file_path):
        return poses
        
    current_mode = None
    current_lines = []
    
    with open(poses_file_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                # Extract the index number out of the model tag
                try:
                    current_mode = int(line.split()[1])
                except (IndexError, ValueError):
                    current_mode = len(poses) + 1
                current_lines = []
            elif line.startswith("ENDMDL"):
                if current_mode is not None:
                    poses[current_mode] = "".join(current_lines)
                current_mode = None
            else:
                current_lines.append(line)
    return poses


# --- REAL-TIME PROTEIN STRUCTURE FETCHING & CONVERSION ---

def fetch_pdb_from_rcsb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_pdb)
        return True, local_pdb
    except Exception as e:
        return False, f"Could not find or download PDB ID '{pdb_id.upper()}'."

def calculate_protein_center(input_pdb):
    x_coords, y_coords, z_coords = [], [], []
    try:
        with open(input_pdb, "r") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x_coords.append(float(line[30:38].strip()))
                        y_coords.append(float(line[38:46].strip()))
                        z_coords.append(float(line[46:54].strip()))
                    except ValueError:
                        continue
        if x_coords:
            return (sum(x_coords)/len(x_coords), 
                    sum(y_coords)/len(y_coords), 
                    sum(z_coords)/len(z_coords))
    except Exception:
        pass
    return 0.0, 0.0, 0.0

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False):
    autodock_type_map = {
        "H": "H", "HD": "HD", "HS": "HS", "C": "C", "A": "A", 
        "N": "N", "NA": "NA", "NS": "NS", "O": "O", "OA": "OA", 
        "S": "S", "SA": "SA", "P": "P", "F": "F", 
        "CL": "Cl", "BR": "Br", "I": "I", "ZN": "Zn", "MG": "Mg"
    }
    torsions = 0
    if is_ligand:
        try:
            mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
            if mol:
                torsions = AllChem.CalcNumRotatableBonds(mol)
        except Exception:
            torsions = 4

    try:
        with open(input_pdb, "r") as pdb, open(output_pdbqt, "w") as pdbqt:
            if is_ligand:
                pdbqt.write("ROOT\n")

            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    record_type = line[:6].strip()
                    try:
                        atom_id = int(line[6:11].strip())
                    except ValueError:
                        atom_id = 1
                        
                    atom_name = line[12:16]
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    if not chain_id:
                        chain_id = "A"
                    try:
                        res_seq = int(line[22:26].strip())
                    except ValueError:
                        res_seq = 1
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                    except ValueError:
                        continue
                        
                    element = line[76:78].strip()
                    if not element:
                        element = ''.join([c for c in atom_name if c.isalpha()])[0]
                    element = ''.join([c for c in element if c.isalpha()]).upper()
                    
                    vina_type = autodock_type_map.get(element, element.title())
                    if element == "C" and "AR" in atom_name.upper():
                        vina_type = "A"

                    pdbqt_line = (
                        f"{record_type:<6}{atom_id:>5} {atom_name:<4} {res_name:>3} "
                        f"{chain_id}{res_seq:>4}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}    "
                        f"+0.000 {vina_type:<2}\n"
                    )
                    pdbqt.write(pdbqt_line)
                    
            if is_ligand:
                pdbqt.write("ENDROOT\n")
                pdbqt.write(f"TORSDOF {torsions}\n")
            else:
                pdbqt.write("ENDMDL\n")
        return True, output_pdbqt
    except Exception as e:
        return False, str(e)


# --- LIGAND MOLECULAR GENERATION ---

def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return False, "Invalid SMILES string structure."
        
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1:
            return False, "3D coordinate embedding step failed."
        
        AllChem.MMFFOptimizeMolecule(mol)
        
        temp_pdb = "temp_ligand.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
        convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)
        
        if os.path.exists(temp_pdb):
            os.remove(temp_pdb)
        return True, output_filename
    except Exception as e:
        return False, str(e)


# --- ADVANCED PY3DMOL VIEWPORT (COMPLEX INTERACTION RENDERER) ---

def render_complex_html(receptor_pdbqt, ligand_pdbqt=None):
    """Generates an iframe script rendering both target ribbon structures and sticks."""
    ligand_block = f"viewer.addModel(`{ligand_pdbqt}`, 'pdb'); viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'cyanCarbon', radius: 0.25}}}});" if ligand_pdbqt else ""
    
    html_content = f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <div id="container" style="height: 400px; width: 100%; position: relative;"></div>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: '#f8f9fa'}});
        
        // Load target structure model (Index 0)
        viewer.addModel(`{receptor_pdbqt}`, 'pdb');
        viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'spectrum'}}}});
        
        // Load target ligand model (Index 1) if available
        {ligand_block}
        
        viewer.zoomTo();
        viewer.render();
    </script>
    """
    components.html(html_content, height=410)


# --- WEB RUNTIME INTERFACE ---

st.set_page_config(page_title="In Silico Docking Hub", layout="wide")
st.title("🔬 Automated Molecular Docking Studio")

# Session state initializations to track cross-interaction values
if "center_x" not in st.session_state: st.session_state.center_x = 0.0
if "center_y" not in st.session_state: st.session_state.center_y = 0.0
if "center_z" not in st.session_state: st.session_state.center_z = 0.0
if "docking_results_raw" not in st.session_state: st.session_state.docking_results_raw = None

col_params, col_visual = st.columns([1, 1])
target_ready = False
prepared_receptor_path = "protein.pdbqt"

with col_params:
    st.header("1. Target Protein Setup")
    protein_source = st.radio("Choose Protein Input Method:", ["Type 4-Letter PDB ID", "Upload File (.pdb or .pdbqt)"])
    
    if protein_source == "Type 4-Letter PDB ID":
        pdb_id_input = st.text_input("Enter RCSB PDB ID", value="1IEP").strip()
        if pdb_id_input:
            fetch_success, pdb_file_path = fetch_pdb_from_rcsb(pdb_id_input)
            if fetch_success:
                cx, cy, cz = calculate_protein_center(pdb_file_path)
                st.session_state.center_x, st.session_state.center_y, st.session_state.center_z = cx, cy, cz
                st.success(f"Protein structural center mapped automatically: X={cx:.1f}, Y={cy:.1f}, Z={cz:.1f}")
                conv_success, err_msg = convert_pdb_to_pdbqt(pdb_file_path, prepared_receptor_path, is_ligand=False)
                target_ready = conv_success
                
    else:
        uploaded_file = st.file_uploader("Upload Target Protein File", type=["pdb", "pdbqt"])
        if uploaded_file:
            temp_upload_path = f"uploaded_{uploaded_file.name}"
            with open(temp_upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            if uploaded_file.name.endswith(".pdb"):
                cx, cy, cz = calculate_protein_center(temp_upload_path)
                st.session_state.center_x, st.session_state.center_y, st.session_state.center_z = cx, cy, cz
                conv_success, err_msg = convert_pdb_to_pdbqt(temp_upload_path, prepared_receptor_path, is_ligand=False)
                target_ready = conv_success
            else:
                os.replace(temp_upload_path, prepared_receptor_path)
                target_ready = True

    st.header("2. Small Molecule Ligand Setup")
    smiles_input = st.text_input("Enter Ligand SMILES String", "CC(=O)NC1=CC=C(O)C=C1")
    
    st.header("3. Grid Box Coordinates")
    grid_cx = st.number_input("Center X Coordinate", value=st.session_state.center_x)
    grid_cy = st.number_input("Center Y Coordinate", value=st.session_state.center_y)
    grid_cz = st.number_input("Center Z Coordinate", value=st.session_state.center_z)
    
    grid_sx = st.slider("Grid Box Size X (Å)", 15, 40, 22)
    grid_sy = st.slider("Grid Box Size Y (Å)", 15, 40, 22)
    grid_sz = st.slider("Grid Box Size Z (Å)", 15, 40, 22)
    
    exhaustiveness = st.slider("Search Exhaustiveness", min_value=4, max_value=32, value=8, step=4)
    run_btn = st.button("🚀 Initialize Docking Algorithm", type="primary", disabled=not target_ready)

with col_visual:
    st.header("4. Active Viewport & Analytics")
    
    # Toggle interface layout between active setup states or evaluation results dashboards
    if st.session_state.docking_results_raw is None:
        view_mode = st.radio("Select Viewport Target Matrix:", ["View Input Ligand", "View Target Protein Structure"])
        
        if view_mode == "View Input Ligand" and smiles_input:
            success, res = convert_smiles_to_pdbqt(smiles_input)
            if success:
                with open(res, "r") as f: ligand_data = f.read()
                render_complex_html(receptor_pdbqt="", ligand_pdbqt=ligand_data)
                
        elif view_mode == "View Target Protein Structure" and target_ready:
            if os.path.exists(prepared_receptor_path):
                with open(prepared_receptor_path, "r") as f: protein_data = f.read()
                render_complex_html(receptor_pdbqt=protein_data, ligand_pdbqt=None)
    else:
        st.subheader("Interactive Complex Viewport")
        
        # Pull separate calculated poses arrays
        parsed_poses = split_docking_poses("docking_poses.pdbqt")
        
        if parsed_poses:
            # Dropdown menu to let users cycle through modes
            selected_pose = st.selectbox(
                "Choose Docking Pose to Visualize:", 
                options=list(parsed_poses.keys()),
                format_func=lambda x: f"Mode {x} Pose Fit"
            )
            
            # Read both elements back to project a combined image frame
            with open(prepared_receptor_path, "r") as f: protein_data = f.read()
            render_complex_html(receptor_pdbqt=protein_data, ligand_pdbqt=parsed_poses[selected_pose])
        else:
            st.warning("No structural model models found in output variables.")
            
        # Reset toggle to go back to target adjustments setup window
        if st.button("🔄 Reset Environment Canvas"):
            st.session_state.docking_results_raw = None
            st.rerun()

    # --- ACTION EXECUTION BOUNDARY ---
    if run_btn and target_ready:
        with st.spinner("Processing structural search calculations using flexible ligand geometries..."):
            vina_command = [
                "./vina",
                "--receptor", prepared_receptor_path,
                "--ligand", "ligand.pdbqt",
                "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz),
                "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz),
                "--exhaustiveness", str(exhaustiveness),
                "--out", "docking_poses.pdbqt"
            ]
            try:
                process = subprocess.run(vina_command, capture_output=True, text=True, check=True)
                if process.stdout:
                    # Capture the data in our session state to prevent refresh losses
                    st.session_state.docking_results_raw = process.stdout
                    st.success("Calculations complete! Redirecting viewport to dashboard views...")
                    st.rerun()
            except subprocess.CalledProcessError as err:
                st.error("Calculations exited with error flags.")
                st.code(err.stderr if err.stderr else err.stdout)

# --- GLOBAL DATAFRAME ANALYTICS DISPLAY ZONE ---
if st.session_state.docking_results_raw is not None:
    st.write("---")
    st.header("📊 Screening Metrics Dashboard & Data Export")
    
    # Run text conversion routine to generate clear tabular formats
    df_results = parse_vina_output_text(st.session_state.docking_results_raw)
    
    if not df_results.empty:
        col_table, col_export = st.columns([2, 1])
        
        with col_table:
            st.subheader("Sortable Affinities Log Matrix")
            # Render interactive data matrix component
            st.dataframe(df_results, hide_index=True, use_container_width=True)
            
        with col_export:
            st.subheader("Export Evaluation Sheet")
            st.write("Extract calculated parameters directly for use in formal research publications, charting, or records.")
            
            # Convert tabular structures directly to clean csv streams
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Data Sheet (.CSV)",
                data=csv_data,
                file_name="screening_affinity_report.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning("Data logger output was unparseable. Raw log dump matches below:")
        st.text(st.session_state.docking_results_raw)
