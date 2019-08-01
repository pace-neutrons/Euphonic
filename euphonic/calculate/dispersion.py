import numpy as np
from euphonic.util import direction_changed


def reorder_freqs(data):
    """
    For a data object, reorders frequencies across q-points in order to
    join branches, and sets the freqs and eigenvecs attributes to the
    newly ordered frequencies

    Parameters
    ----------
    data: PhononData or InterpolationData object
        Data object containing the frequencies, eigenvectors and q-points
        required for reordering
    """
    n_qpts = data.n_qpts
    n_branches = data.n_branches
    qpts = data.qpts
    freqs = data.freqs.magnitude
    eigenvecs = data.eigenvecs

    if eigenvecs.size == 0:
        print("""No eigenvectors in data object, cannot reorder
             frequencies""")
        return

    ordered_freqs = np.zeros(freqs.shape)
    ordered_eigenvecs = np.zeros(eigenvecs.shape, dtype=np.complex128)
    qmap = np.arange(n_branches)

    # Only calculate qmap and reorder freqs if the direction hasn't changed
    # and there is no LO-TO splitting
    calculate_qmap = np.concatenate(([True], np.logical_not(
        direction_changed(qpts))))
    if hasattr(data, 'split_i'):
        split_freqs = data.split_freqs.magnitude
        split_eigenvecs = data.split_eigenvecs
        ordered_split_freqs = np.zeros(split_freqs.shape)
        ordered_split_eigenvecs = np.zeros(
            split_eigenvecs.shape, dtype=np.complex128)
        calculate_qmap[data.split_i + 1] = False

    # Don't reorder first q-point
    ordered_freqs[0, :] = freqs[0, :]
    ordered_eigenvecs[0, :] = eigenvecs[0, :]
    prev_evecs = eigenvecs[0, :, :, :]
    for i in range(1, n_qpts):
        # Initialise q-point mapping for this q-point
        qmap_tmp = np.arange(n_branches)
        # Compare eigenvectors for each mode for this q-point with every
        # mode for the previous q-point
        # Explicitly broadcast arrays with repeat and tile to ensure
        # correct multiplication of modes
        curr_evecs = eigenvecs[i, :, :, :]
        current_eigenvecs = np.repeat(curr_evecs, n_branches, axis=0)
        prev_eigenvecs = np.tile(prev_evecs, (n_branches, 1, 1))

        if calculate_qmap[i-1]:
            # Compute complex conjugated dot product of every mode of this
            # q-point with every mode of previous q-point, and sum the dot
            # products over ions (i.e. multiply eigenvectors elementwise, then
            # sum over the last 2 dimensions)
            dots = np.absolute(np.einsum('ijk,ijk->i',
                                         np.conj(prev_eigenvecs),
                                         current_eigenvecs))

            # Create matrix of dot products for each mode of this q-point with
            # each mode of the previous q-point
            dot_mat = np.reshape(dots, (n_branches, n_branches))

            # Find greates exp(-iqr)-weighted dot product
            for j in range(n_branches):
                max_i = (np.argmax(dot_mat))
                mode = int(max_i/n_branches)  # Modes are dot_mat rows
                prev_mode = max_i % n_branches  # Prev q-pt modes are columns
                # Ensure modes aren't mapped more than once
                dot_mat[mode, :] = 0
                dot_mat[:, prev_mode] = 0
                qmap_tmp[mode] = prev_mode
        # Map q-points according to previous q-point mapping
        qmap = qmap[qmap_tmp]

        prev_evecs = curr_evecs

        # Reorder frequencies and eigenvectors
        ordered_eigenvecs[i, qmap] = eigenvecs[i, :]
        ordered_freqs[i, qmap] = freqs[i, :]

        if hasattr(data, 'split_i') and i in data.split_i:
            idx = np.where(i == data.split_i)
            ordered_split_eigenvecs[idx, qmap] = split_eigenvecs[idx]
            ordered_split_freqs[idx, qmap] = split_freqs[idx]

    ordered_freqs = ordered_freqs*data.freqs.units
    data.eigenvecs = ordered_eigenvecs
    data.freqs = ordered_freqs
    if hasattr(data, 'split_i'):
        data.split_freqs = ordered_split_freqs*data.split_freqs.units
        data.split_eigenvecs = ordered_split_eigenvecs
