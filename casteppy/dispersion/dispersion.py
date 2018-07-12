import numpy as np


def reorder_freqs(freqs, qpts, eigenvecs):
    """
    Reorders frequencies across q-points in order to join branches

    Parameters
    ----------
    freqs: list of floats
        M x N list of phonon band frequencies, where M = number of q-points
        and N = number of bands, ordered according to increasing q-point number
    qpts : list of floats
        M x 3 list of q-point coordinates, where M = number of q-points
    eigenvecs: list of complex floats
        M x L x 3 list of the atomic displacements (dynamical matrix
        eigenvectors), where M = number of q-points and
        L = number of ions*number of bands

    Returns
    -------
    ordered_freqs: list of floats
        Ordered M x N list of phonon band frequencies, where M = number of
        q-points and N = number of bands
    """

    n_qpts = qpts.shape[0]
    n_branches = freqs.shape[1]
    n_ions = int(eigenvecs.shape[1]/n_branches)
    ordered_freqs = np.zeros((n_qpts,n_branches))
    qmap = np.arange(n_branches)

    # Only calculate qmap and reorder freqs if the direction hasn't changed
    calculate_qmap = np.concatenate(([True], np.logical_not(
        direction_changed(qpts))))
    # Don't reorder first q-point
    ordered_freqs[0,:] = freqs[0,:]
    for i in range(1,n_qpts):
        # Initialise q-point mapping for this q-point
        qmap_tmp = np.arange(n_branches)
        if calculate_qmap[i-1]:
            # Compare eigenvectors for each mode for this q-point with every
            # mode for the previous q-point
            # Reshape eigenvector arrays for this and previous q-point so that
            # each mode is a row and each ion is a column, for efficient
            # summing over modes later. Then explicitly broadcast arrays with
            # repeat and tile to ensure correct multiplication of modes
            current_eigenvecs = np.reshape(eigenvecs[i, :, :],
                                           (n_branches, n_ions, 3))
            current_eigenvecs = np.repeat(current_eigenvecs,
                                          n_branches, axis=0)
            prev_eigenvecs = np.reshape(eigenvecs[i-1, :, :],
                                        (n_branches, n_ions, 3))
            prev_eigenvecs = np.tile(prev_eigenvecs,
                                     (n_branches, 1, 1))
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
                mode = int(max_i/n_branches) # Modes are dot_mat rows
                prev_mode = max_i%n_branches # Prev q-pt modes are columns
                # Ensure modes aren't mapped more than once
                dot_mat[mode, :] = 0
                dot_mat[:, prev_mode] = 0
                qmap_tmp[mode] = prev_mode
        # Map q-points according to previous q-point mapping
        qmap = qmap[qmap_tmp]

        # Reorder frequencies
        ordered_freqs[i,qmap] = freqs[i,:]

    return ordered_freqs


def direction_changed(qpts, tolerance=5e-6):
    """
    Takes a N length list of q-points and returns an N - 2 length list of
    booleans indicating whether the direction has changed between each pair
    of q-points
    """

    # Get vectors between q-points
    delta = np.diff(qpts, axis=0)

    # Dot each vector with the next to determine the relative direction
    dot = np.einsum('ij,ij->i', delta[1:,:], delta[:-1,:])
    # Get magnitude of each vector
    modq = np.linalg.norm(delta, axis=1)
    # Determine how much the direction has changed (dot) relative to the
    # vector sizes (modq)
    direction_changed = (np.abs(np.abs(dot) - modq[1:]*modq[:-1]) > tolerance)

    return direction_changed
