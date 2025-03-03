import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay
import matplotlib.tri as tri
from scipy.special import comb
import io

# ------------------------------
# Helper Functions
# ------------------------------

def generate_diamond_nodes(R):
    """
    Generate nodes arranged in a diamond shape that represents the binomial i-state
    structure. For k = 0,...,R, row k has n = comb(R, k) nodes.
    The y-coordinate is mapped from 1 (k=0) to -1 (k=R) and nodes are centered in x.
    """
    nodes = []
    for k in range(R + 1):
        y = 1 - 2 * (k / R)  # top (k=0) at y=1, bottom (k=R) at y=-1
        n_points = int(comb(R, k))
        if n_points == 1:
            xs = [0.0]
        else:
            xs = np.linspace(-1, 1, n_points)
        for x in xs:
            nodes.append([x, y])
    return np.array(nodes)

def assign_occupancy_to_nodes(nodes, R, occupancy_values):
    """
    Assign occupancy values to nodes based on the row they belong to.
    The first row (k=0) gets occupancy_values[0], the next row gets occupancy_values[1], etc.
    Assumes nodes are ordered row-by-row (top to bottom).
    """
    occupancies = []
    for k in range(R + 1):
        n_points = int(comb(R, k))
        occupancies.extend([occupancy_values[k]] * n_points)
    return np.array(occupancies)

def create_triangulation(nodes):
    """
    Create a Delaunay triangulation for the provided nodes.
    """
    triangulation = Delaunay(nodes)
    return triangulation.simplices

def fem_assemble_matrices(nodes, elements):
    """
    Assemble the FEM stiffness (A) and mass (M) matrices using linear triangular elements.
    """
    num_nodes = nodes.shape[0]
    A_mat = sp.lil_matrix((num_nodes, num_nodes))
    M_mat = sp.lil_matrix((num_nodes, num_nodes))
    
    for elem in elements:
        idx = elem
        coords = nodes[idx]
        mat = np.array([[1, coords[0, 0], coords[0, 1]],
                        [1, coords[1, 0], coords[1, 1]],
                        [1, coords[2, 0], coords[2, 1]]])
        area = 0.5 * np.abs(np.linalg.det(mat))
        if area < 1e-14:
            continue
        x = coords[:, 0]
        y = coords[:, 1]
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        K_local = np.zeros((3, 3))
        for i_local in range(3):
            for j_local in range(3):
                K_local[i_local, j_local] = (b[i_local] * b[j_local] + c[i_local] * c[j_local]) / (4 * area)
        M_local = (area / 12.0) * np.array([[2, 1, 1],
                                             [1, 2, 1],
                                             [1, 1, 2]])
        for i_local, i_global in enumerate(idx):
            for j_local, j_global in enumerate(idx):
                A_mat[i_global, j_global] += K_local[i_local, j_local]
                M_mat[i_global, j_global] += M_local[i_local, j_local]
    return A_mat.tocsr(), M_mat.tocsr()

def solve_pde(nodes, elements, occupancy, max_iter=150, tol=1e-1, damping=0.1, kappa_target=1.0, num_steps=5):
    """
    Solve the nonlinear PDE using continuation and a damped Newton–Raphson method.
    PDE: A·φ + 0.5 * M*(occupancy * exp(2φ)) = 0
    """
    num_nodes = nodes.shape[0]
    A_mat, M_mat = fem_assemble_matrices(nodes, elements)
    phi = np.zeros(num_nodes)
    kappa_values = np.linspace(0, kappa_target, num_steps + 1)
    
    for kappa in kappa_values[1:]:
        for it in range(max_iter):
            nonlin = occupancy * np.exp(2 * phi)
            F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlin)
            # Jacobian: A + M*diag(occupancy * exp(2φ))
            J = A_mat + M_mat.dot(sp.diags(occupancy * np.exp(2 * phi)))
            delta_phi = spla.spsolve(J, -F)
            phi += damping * delta_phi
            if np.linalg.norm(delta_phi) < tol:
                break
    return phi, A_mat, M_mat

def compute_ricci_curvature(phi, A_mat, M_mat):
    """
    Compute an approximate Ricci curvature:
        R = -2 exp(-2φ) Δφ,
    where Δφ is approximated using a lumped mass matrix.
    """
    M_lumped = np.array(M_mat.sum(axis=1)).flatten()
    lap_phi = A_mat.dot(phi) / M_lumped
    R_curv = -2.0 * np.exp(-2 * phi) * lap_phi
    return R_curv

def generate_oxishape(R, occupancy_values):
    """
    Generate the Oxi‑Shape for a protein with R cysteines (hence R+1 k‑states)
    given the occupancy values for each k‑state.
    """
    nodes = generate_diamond_nodes(R)
    occupancy = assign_occupancy_to_nodes(nodes, R, occupancy_values)
    elements = create_triangulation(nodes)
    phi, A_mat, M_mat = solve_pde(nodes, elements, occupancy)
    R_curv = compute_ricci_curvature(phi, A_mat, M_mat)
    z = phi - occupancy  # vertical deformation
    return nodes, elements, z, R_curv

# ------------------------------
# Streamlit Application
# ------------------------------

st.title("Oxi‑Shapes Generator")
st.write("""
This interactive app generates Oxi‑Shapes for cysteine proteoforms using a theorem‑derived field equation.
The domain is represented as a diamond reflecting the binomial i‑state structure, with k‑states ranging from 0 to R.
""")

# Sidebar inputs
R = st.number_input("Enter the number of cysteines (R)", min_value=1, max_value=10, value=4, step=1)

st.write("Enter the fractional occupancy for each k‑state (from k = 0 to k = R).")
occupancy_values = []
for k in range(int(R) + 1):
    occ = st.number_input(f"Occupancy for k = {k}", min_value=0.0, max_value=1.0, value=0.25, step=0.05, key=f"occ_{k}")
    occupancy_values.append(occ)
occupancy_values = np.array(occupancy_values)

if st.button("Generate Oxi‑Shape"):
    with st.spinner("Computing Oxi‑Shape..."):
        nodes, elements, z, R_curv = generate_oxishape(int(R), occupancy_values)
    # Create a Delaunay triangulation for plotting
    triang_plot = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Normalize Ricci curvature for color mapping
    R_norm = (R_curv - R_curv.min()) / (R_curv.max() - R_curv.min() + 1e-10)
    facecolors = plt.cm.viridis(R_norm)
    surf = ax.plot_trisurf(triang_plot, z, cmap="viridis", shade=True,
                             edgecolor="none", antialiased=True, linewidth=0.2,
                             alpha=0.9, facecolors=facecolors)
    ax.set_title("Oxi‑Shape (Diamond Geometry)")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("z = φ - occupancy")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Ricci Curvature")
    
    st.pyplot(fig)
    
    # Create a BytesIO buffer for download
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    
    st.download_button(
        label="Download Oxi‑Shape Image (300 dpi PNG)",
        data=buf,
        file_name="oxishape.png",
        mime="image/png"
    )
