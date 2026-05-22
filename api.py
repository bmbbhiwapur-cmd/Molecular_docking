import streamlit as st
import subprocess
import os
import urllib.request
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


# --- REAL-TIME PROTEIN STRUCTURE FETCHING & CONVERSION ---

def fetch_pdb_from_rcsb(pdb_id):
    """Fetches a standard PDB structure directly from the RCSB server."""
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_pdb)
        return True, local_pdb
    except Exception as e:
        return False, f"Could not find or download PDB ID '{pdb_id.upper()}'. Check your internet link or ID format."

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt"):
    """
    Parses a standard PDB structural file and builds a compliant PDBQT format
    by assigning default partial gas-charges and Gasteiger-type atom descriptors.
    """
    try:
        with open(input_pdb, "r") as pdb, open(output_pdbqt, "w") as pdbqt:
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    # Extract standard PDB coordinates element tag
                    element = line[76:78].strip()
                    if not element:
                        element = line[12:14].strip() # Fallback mapping element from atom name string
                    
                    # Sanitize element mapping digits or symbols
                    element = ''.join([i for i in element if i.isalpha()]).upper()
                    if not element:
                        element = "C" # Safe generic fallback assignment
                        
                    # Write out structured line adding generic charge space (+0.000) and element flags
                    pdbqt.write(f"{line[:70]}    +0.000 {element:<2}\n")
                elif line.startswith("TER"):
                    pdbqt.write("TER\n")
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
        
        with open(temp_pdb, "r") as pdb_file, open(output_filename, "w") as pdbqt_file:
            for line in pdb_file:
                if line.startswith(("ATOM", "HETATM")):
                    atom_type = line[76:78].strip()
                    pdbqt_file.write(f"{line[:70]}    +0.000 {atom_type}\n")
        
        if os.path.exists(temp_pdb):
            os.remove(temp_pdb)
            
        return True, output_filename
    except Exception as e:
        return False, str(e)


# --- ISOLATED IFRAME PY3DMOL VIEWPORT RENDERING ---

def render_molecule_html(pdb_string, style_type="stick", scheme="cyanCarbon"):
    """Generates an interactive, containerized 3D rendering iframe viewport."""
    html_content = f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <div id="container" style="height: 380px; width: 100%; position: relative;"></div>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: '#f8f9fa'}});
        viewer.addModel(`{pdb_string}`, 'pdb');
        viewer.setStyle({{}}, {{{style_type}: {{colorscheme: '{scheme}'}}}});
        viewer.zoomTo();
        viewer.render();
    </script>
    """
    components.html(html_content, height=390)


# --- WEB RUNTIME INTERFACE ---

st.set_page_config(page_title="In Silico Docking Hub", layout="wide")
st.title("🔬 Automated Molecular Docking Studio")
st.write("Streamline protein preparation via direct PDB lookup or standard file uploads, then map binding affinities instantly.")

col_params, col_visual = st.columns([1, 1])

# Global state tracker variables for prepared target structure pathing
target_ready = False
prepared_receptor_path = "protein.pdbqt"

with col_params:
    st.header("1. Target Protein Setup")
    
    # Toggle interface logic between remote network streams or manual disk uploads
    protein_source = st.radio("Choose Protein Input Method:", ["Type 4-Letter PDB ID", "Upload File (.pdb or .pdbqt)"])
    
    if protein_source == "Type 4-Letter PDB ID":
        pdb_id_input = st.text_input("Enter RCSB PDB ID (e.g., 1IEP, 6LU7)", value="1IEP").strip()
        if pdb_id_input:
            fetch_success, pdb_file_path = fetch_pdb_from_rcsb(pdb_id_input)
            if fetch_success:
                st.success(f"Successfully downloaded structural file: {pdb_file_path}")
                conv_success, err_msg = convert_pdb_to_pdbqt(pdb_file_path, prepared_receptor_path)
                if conv_success:
                    st.info("Structure auto-formatted into valid docking coordinates.")
                    target_ready = True
                else:
                    st.error(f"Format Conversion Error: {err_msg}")
            else:
                st.error(pdb_file_path)
                
    else:
        uploaded_file = st.file_uploader("Upload Target Protein File", type=["pdb", "pdbqt"])
        if uploaded_file:
            temp_upload_path = f"uploaded_{uploaded_file.name}"
            with open(temp_upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            if uploaded_file.name.endswith(".pdb"):
                conv_success, err_msg = convert_pdb_to_pdbqt(temp_upload_path, prepared_receptor_path)
                if conv_success:
                    st.success("Uploaded standard PDB compiled down to operational format.")
                    target_ready = True
                else:
                    st.error(f"Format Conversion Error: {err_msg}")
            else:
                # Direct assignment route if file uploaded is already formatted correctly as .pdbqt
                os.replace(temp_upload_path, prepared_receptor_path)
                st.success("Valid coordinate format loaded directly.")
                target_ready = True

    st.header("2. Small Molecule Ligand Setup")
    smiles_input = st.text_input("Enter Ligand SMILES String", "CC(=O)NC1=CC=C(O)C=C1")
    
    st.header("3. Grid Box Grid Coordinates")
    grid_cx = st.number_input("Center X Coordinate", value=15.0, step=0.1)
    grid_cy = st.number_input("Center Y Coordinate", value=50.0, step=0.1)
    grid_cz = st.number_input("Center Z Coordinate", value=15.0, step=0.1)
    
    grid_sx = st.slider("Grid Box Size X (Å)", 10, 40, 20)
    grid_sy = st.slider("Grid Box Size Y (Å)", 10, 40, 20)
    grid_sz = st.slider("Grid Box Size Z (Å)", 10, 40, 20)
    
    exhaustiveness = st.slider("Search Exhaustiveness", min_value=4, max_value=32, value=8, step=4)
    run_btn = st.button("🚀 Initialize Docking Algorithm", type="primary", disabled=not target_ready)

with col_visual:
    st.header("4. Active Viewport Canvas")
    
    view_mode = st.radio("Select Viewport Target Matrix:", ["View Ligand Geometry", "View Target Protein Structure"])
    
    if view_mode == "View Ligand Geometry" and smiles_input:
        st.subheader("Optimized Ligand 3D Topology")
        success, res = convert_smiles_to_pdbqt(smiles_input)
        if success:
            with open(res, "r") as f:
                ligand_data = f.read()
            render_molecule_html(ligand_data, style_type="stick", scheme="cyanCarbon")
        else:
            st.error(f"Structure Building Failed: {res}")
            
    elif view_mode == "View Target Protein Structure" and target_ready:
        st.subheader("Prepared Target Biopolymer Mesh")
        if os.path.exists(prepared_receptor_path):
            with open(prepared_receptor_path, "r") as f:
                protein_data = f.read()
            # Render larger proteins using ribbon layouts for clarity over simple raw sticks
            render_molecule_html(protein_data, style_type="cartoon", scheme="spectrum")

    # --- ACTION EXECUTION BOUNDARY ---
    if run_btn and target_ready:
        with st.spinner("Processing cloud-based structural search calculations..."):
            vina_command = [
                "./vina",
                "--receptor", prepared_receptor_path,
                "--ligand", "ligand.pdbqt",
                "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz),
                "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz),
                "--exhaustiveness", str(exhaustiveness),
                "--out", "docking_poses.pdbqt",
                "--log", "docking_log.txt"
            ]
            
            try:
                process = subprocess.run(vina_command, capture_output=True, text=True, check=True)
                st.success("Docking processing calculations completed successfully!")
                
                if os.path.exists("docking_log.txt"):
                    with open("docking_log.txt", "r") as log_file:
                        st.text_area("Engine Scoring Log Information Output", log_file.read(), height=250)
            except subprocess.CalledProcessError as err:
                st.error("Calculations exited with error flags.")
                st.code(err.stderr)
