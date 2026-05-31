import numpy as np

from finite_groups import FiniteGroup


def compute_character_table(group) -> tuple[np.ndarray, list]:
    classes = group.conjugacy_classes()
    k = len(classes)
    n = group.order

    # 1. Map elements to their class index
    el_to_class_inx = {el: idx for idx, cls in enumerate(classes) for el in cls}

    # 2. Build Class Algebra Matrices M_i
    matrices = []
    for i in range(k):
        M_i = np.zeros((k, k), dtype=complex)
        # Fix: We only need to iterate over classes once to fill the matrix
        for j_inx, cls_j in enumerate(classes):
            for x in classes[i]:
                for y in cls_j:
                    product = group.multiply(x, y)
                    k_inx = el_to_class_inx[product]
                    M_i[j_inx, k_inx] += 1

        # Normalize c_ijk
        for row in range(k):
            for col in range(k):
                M_i[row, col] /= len(classes[col])
        matrices.append(M_i)

    # 3. Simultaneous Diagonalization
    rng = np.random.RandomState(67)
    seed_matrix = sum(rng.uniform(0.1, 1) * M for M in matrices)
    _, eigenvectors = np.linalg.eig(seed_matrix)

    # 4. Extract characters using the Eigenvalue property
    # Each column of eigenvectors is a common eigenvector v
    final_rows = []
    for col in range(k):
        v = eigenvectors[:, col]
        stable_idx = np.argmax(np.abs(v))

        omega_list = []
        for M_i in matrices:
            val = (M_i @ v)[stable_idx] / v[stable_idx]
            omega_list.append(val)

        omega_arr = np.array(omega_list)

        # 5. Row Orthogonality: d^2 * sum( |omega_i|^2 / |Ci| ) = n
        sum_sq = sum(
            (np.conj(omega_arr[i]) * omega_arr[i]).real / len(classes[i])
            for i in range(k)
        )
        d = np.sqrt(n / sum_sq)

        # 6. chi(gi) = (d * omega_i) / |Ci|
        character_row = [(d * omega_arr[i]) / len(classes[i]) for i in range(k)]
        final_rows.append(character_row)

    final_table = np.array(final_rows)
    # Sort by dimension (first column)
    final_table = final_table[np.argsort(np.abs(final_table[:, 0]))]

    return clean_table(final_table), classes


def clean_table(table, decimals=5):
    table = np.where(np.abs(np.imag(table)) < 1e-10, np.real(table), table)
    return np.round(table, decimals)


def decompose_character(
    reducible_phi: np.ndarray, group: FiniteGroup
) -> tuple[dict, np.ndarray]:
    # get necessary data out of the group
    table, classes = compute_character_table(group)
    order = group.order
    class_sizes = np.array([len(c) for c in classes])

    multiplicities = {}

    # Compute multiplicities using Shur's orthogonality
    for i, chi_i in enumerate(table):
        # a_i = <phi, chi_i> = (1/|G|) * sum(|C_J| * phi(g_j) *. conj(chi_i(g_j)))
        inner_product = np.sum(class_sizes * reducible_phi * np.conj(chi_i))
        a_i = inner_product / order
        m = int(np.round(a_i.real))

        if m > 0:
            multiplicities[i] = m
    return multiplicities, table
