import streamlit as st
import subprocess
import os
import urllib.request
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
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


# --- NATIVE PY3DMOL VISUALIZATION COUPLING ---

def render_molecule_html(pdb_string):
    """
    Bypasses stmol dependency by generating an isolated 
    HTML/JS iframe string to execute py3Dmol smoothly.
    """
    html_content = f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <div id="container" style="height: 350px; width: 100%; position: relative;"></div>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: 'white'}});
        viewer.addModel(`{pdb_string}`, 'pdb');
        viewer.setStyle({{}}, {{stick: {{colorscheme: 'cyanCarbon'}}}});
        viewer.zoomTo();
        viewer.render();
    </script>
    """
    # Embed the HTML component safely inside the Streamlit viewport frame
    components.html(html_content, height=360)


# --- CHEMINFORMATICS TOPO CONVERSION ---

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


# --- WEB RUNTIME INTERFACE ---

st.set_page_config(page_title="In Silico Docking Hub", layout="wide")
st.title("🔬 Automated Molecular Docking Studio")
st.write("Generate ligand configurations directly via SMILES and process binding space mechanics using AutoDock Vina.")

col_params, col_visual = st.columns([1, 1])

with col_params:
    st.header("1. Input Configuration")
    uploaded_receptor = st.file_uploader("Upload Target Protein (PDBQT format)", type=["pdbqt"])
    smiles_input = st.text_input("Enter Ligand SMILES String", "CC(=O)NC1=CC=C(O)C=C1")
    
    st.header("2. Search Space Mechanics")
    grid_cx = st.number_input("Center X Coordinate", value=0.0, step=0.1)
    grid_cy = st.number_input("Center Y Coordinate", value=0.0, step=0.1)
    grid_cz = st.number_input("Center Z Coordinate", value=0.0, step=0.1)
    
    grid_sx = st.slider("Grid Box Size X (Å)", 10, 40, 20)
    grid_sy = st.slider("Grid Box Size Y (Å)", 10, 40, 20)
    grid_sz = st.slider("Grid Box Size Z (Å)", 10, 40, 20)
    
    exhaustiveness = st.slider("Search Exhaustiveness", min_value=4, max_value=32, value=8, step=4)
    run_btn = st.button("🚀 Initialize Docking Algorithm", type="primary")

with col_visual:
    st.header("3. Active Viewport Rendering")
    
    if smiles_input:
        st.subheader("Optimized Ligand 3D Topology")
        success, res = convert_smiles_to_pdbqt(smiles_input)
        
        if success:
            with open(res, "r") as f:
                ligand_data = f.read()
            
            # Call our custom HTML rendering routine instead of stmol
            render_molecule_html(ligand_data)
        else:
            st.error(f"Structure Building Failed: {res}")

    if run_btn:
        if not uploaded_receptor:
            st.warning("Please upload a valid target receptor structure to begin.")
        else:
            with st.spinner("Processing cloud-based structural search calculations..."):
                with open("protein.pdbqt", "wb") as f:
                    f.write(uploaded_receptor.getbuffer())
                    
                vina_command = [
                    "./vina",
                    "--receptor", "protein.pdbqt",
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
