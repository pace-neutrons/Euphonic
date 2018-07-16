import unittest
import seekpath
import matplotlib
# Need to set non-interactive backend before importing casteppy to avoid
# DISPLAY requirement when testing plotting functions
matplotlib.use('Agg')
import numpy as np
import numpy.testing as npt
from matplotlib import figure
from casteppy.plot.dos import plot_dos
from casteppy.plot.dispersion import (calc_abscissa, recip_space_labels,
                                      generic_qpt_labels, get_qpt_label, 
                                      plot_dispersion)


class TestCalcAbscissa(unittest.TestCase):

    def test_iron(self):
        recip = [[0., 1.15996339, 1.15996339],
                 [1.15996339, 0., 1.15996339],
                 [1.15996339, 1.15996339, 0.]]
        qpts = [[-0.37500000, -0.45833333,  0.29166667],
                [-0.37500000, -0.37500000,  0.29166667],
                [-0.37500000, -0.37500000,  0.37500000],
                [-0.37500000,  0.45833333, -0.20833333],
                [-0.29166667, -0.45833333,  0.20833333],
                [-0.29166667, -0.45833333,  0.29166667],
                [-0.29166667, -0.37500000, -0.29166667],
                [-0.29166667, -0.37500000,  0.04166667],
                [-0.29166667, -0.37500000,  0.12500000],
                [-0.29166667, -0.37500000,  0.29166667]]
        expected_abscissa = [0., 0.13670299, 0.27340598, 1.48844879,
                             2.75618022, 2.89288323, 3.78930474,
                             4.33611674, 4.47281973, 4.74622573]
        npt.assert_allclose(calc_abscissa(qpts, recip),
                            expected_abscissa)


class TestRecipSpaceLabels(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        NaH = lambda:0
        NaH.cell_vec = [[0.0, 2.3995, 2.3995],
                        [2.3995, 0.0, 2.3995],
                        [2.3995, 2.3995, 0.0]]
        NaH.ion_pos = [[0.5, 0.5, 0.5],
                       [0.0, 0.0, 0.0]]
        NaH.ion_type = ['H', 'Na']
        NaH.qpts = np.array([[-0.25, -0.25, -0.25],
                             [-0.25, -0.50, -0.50],
                             [ 0.00, -0.25, -0.25],
                             [ 0.00,  0.00,  0.00],
                             [ 0.00, -0.50, -0.50],
                             [ 0.25,  0.00, -0.25],
                             [ 0.25, -0.50, -0.25],
                             [-0.50, -0.50, -0.50]])
        NaH.expected_labels = ['', '', '', 'X', '', 'W_2', 'L']
        NaH.expected_qpts_with_labels = [0, 1, 2, 4, 5, 6, 7]
        (NaH.labels, NaH.qpts_with_labels) = recip_space_labels(
            NaH.qpts, NaH.cell_vec, NaH.ion_pos, NaH.ion_type)
        self.NaH = NaH

    def test_labels_nah(self):
        npt.assert_equal(self.NaH.labels, self.NaH.expected_labels)

    def test_qpts_with_labels_nah(self):
        npt.assert_equal(self.NaH.qpts_with_labels,
                         self.NaH.expected_qpts_with_labels)


class TestGenericQptLabels(unittest.TestCase):

    def setUp(self):
        self.generic_dict = generic_qpt_labels()

    def test_returns_dict(self):
        self.assertIsInstance(self.generic_dict, dict)

    def test_gamma_point(self):
        key = '0 0 0'
        expected_value = [0., 0., 0.]
        npt.assert_array_equal(self.generic_dict[key], expected_value)

    def test_mixed_point(self):
        key = '5/8 1/3 3/8'
        expected_value = [0.625, 1./3., 0.375]
        npt.assert_allclose(self.generic_dict[key], expected_value)


class TestGetQptLabel(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        NaH = lambda:0
        cell_vec = [[0.0, 2.3995, 2.3995],
                    [2.3995, 0.0, 2.3995],
                    [2.3995, 2.3995, 0.0]]
        ion_pos = [[0.5, 0.5, 0.5],
                   [0.0, 0.0, 0.0]]
        ion_num = [1, 2]
        cell = (cell_vec, ion_pos, ion_num)
        NaH.point_labels = seekpath.get_path(cell)["point_coords"]
        self.NaH = NaH

    def test_gamma_pt_nah(self):
        gamma_pt = [0.0, 0.0, 0.0]
        expected_label = 'GAMMA'
        self.assertEqual(get_qpt_label(gamma_pt, self.NaH.point_labels),
                         expected_label)

    def test_x_pt_nah(self):
        x_pt = [0.0, -0.5, -0.5]
        expected_label = 'X'
        self.assertEqual(get_qpt_label(x_pt, self.NaH.point_labels),
                         expected_label)

    def test_w2_pt_nah(self):
        w2_pt = [0.25, -0.5, -0.25]
        expected_label = 'W_2'
        self.assertEqual(get_qpt_label(w2_pt, self.NaH.point_labels),
                         expected_label)


class TestPlotDispersion(unittest.TestCase):

    def setUp(self):
        # Input values
        self.abscissa = [0.0, 0.13, 0.27, 1.48, 2.75, 2.89]
        self.freq_up = [[0.61, 0.71, 3.36, 4.19, 4.65, 11.76],
                        [0.75, 0.71, 3.38, 3.97, 4.55, 9.65],
                        [0.65, 0.57, 3.93, 3.93, 4.29, 9.30],
                        [1.91, 0.68, 2.32, 4.60, 3.54, 6.92],
                        [1.31, 0.83, 3.63, 4.72, 2.62, 9.91],
                        [0.94, 0.70, 3.60, 4.76, 3.30, 9.84]]
        self.freq_down = [[2.20, 2.27, 5.22, 6.19, 6.77, 12.65],
                          [2.38, 2.18, 5.24, 5.93, 6.66, 10.67],
                          [2.31, 2.03, 5.88, 5.88, 6.39, 10.28],
                          [3.44, 8.40, 4.00, 6.71, 5.55, 1.92],
                          [2.82, 2.36, 5.58, 6.85, 4.39, 10.95],
                          [2.49, 2.24, 5.50, 6.91, 5.17, 10.86]]
        self.units = 'eV'
        self.title = 'Iron'
        self.xticks = [0.0, 0.13, 0.27, 1.48, 2.75, 2.89]
        self.xlabels = ['', '', '5/8 5/8 3/8', '', '', '']
        self.fermi = [4.71, 4.71]

        # Results
        self.fig = plot_dispersion(
            self.abscissa, self.freq_up, self.freq_down, self.units,
            self.title, self.xticks, self.xlabels, self.fermi)
        self.ax = self.fig.axes[0]

    def tearDown(self):
        # Ensure figures are closed after tests
        matplotlib.pyplot.close('all')

    def test_returns_fig(self):
        self.assertIsInstance(self.fig, figure.Figure)

    def test_n_series(self):
        n_series = (len(self.freq_up[0])
                  + len(self.freq_down[0])
                  + len(self.fermi))
        self.assertEqual(len(self.ax.get_lines()), n_series)

    def test_freq_xaxis(self):
        n_correct_x = 0
        for line in self.ax.get_lines():
            if np.array_equal(line.get_data()[0], self.abscissa):
                n_correct_x += 1
        # Check that there are as many lines with abscissa for the x-axis
        # values as there are both freq_up and freq_down branches
        self.assertEqual(n_correct_x,
                         len(self.freq_up[0]) + len(self.freq_down[0]))

    def test_freq_yaxis(self):
        all_freq_branches = np.vstack((np.transpose(self.freq_up),
                                       np.transpose(self.freq_down)))
        n_correct_y = 0
        for branch in all_freq_branches:
            for line in self.ax.get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y += 1
                    break
        # Check that every branch has a matching y-axis line
        self.assertEqual(n_correct_y, len(all_freq_branches))

    def test_fermi_yaxis(self):
        n_correct_y = 0
        for ef in self.fermi:
            for line in self.ax.get_lines():
                if np.all(np.array(line.get_data()[1]) == ef):
                    n_correct_y += 1
                    break
        self.assertEqual(n_correct_y, len(self.fermi))

    def test_xaxis_tick_locs(self):
        npt.assert_array_equal(self.ax.get_xticks(), self.xticks)

    def test_xaxis_tick_labels(self):
        ticklabels = [x.get_text() for x in self.ax.get_xticklabels()]
        npt.assert_array_equal(ticklabels, self.xlabels)

    def test_freq_up_empty(self):
        # Test freq down is still plotted when freq_up is empty
        fig = plot_dispersion(
            self.abscissa, [], self.freq_down, self.units,
            self.title, self.xticks, self.xlabels, self.fermi)
        n_correct_y = 0
        for branch in np.transpose(self.freq_down):
            for line in fig.axes[0].get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y += 1
                    break
        # Check that every freq down branch has a matching y-axis line
        self.assertEqual(n_correct_y, len(self.freq_down[0]))

    def test_freq_down_empty(self):
        # Test freq up is still plotted when freq_down is empty
        fig = plot_dispersion(
            self.abscissa, self.freq_up, [], self.units,
            self.title, self.xticks, self.xlabels, self.fermi)
        n_correct_y = 0
        for branch in np.transpose(self.freq_up):
            for line in fig.axes[0].get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y += 1
                    break
        # Check that every freq up branch has a matching y-axis line
        self.assertEqual(n_correct_y, len(self.freq_up[0]))

class TestPlotDispersionWithBreak(unittest.TestCase):

    def setUp(self):
        # Input values
        self.abscissa = [1.63, 1.76, 1.88, 2.01, 3.43, 3.55, 3.66, 3.78]
        self.freq_up = [[0.98, 0.98, 2.65, 4.34, 4.34, 12.14],
                        [0.51, 0.51, 3.55, 4.50, 4.50, 12.95],
                        [0.24, 0.24, 4.39, 4.66, 4.66, 13.79],
                        [0.09, 0.09, 4.64, 4.64, 4.64, 14.19],
                        [0.08, 1.62, 3.90, 4.22, 5.01,  5.12],
                        [0.03, 1.65, 3.81, 4.01, 4.43,  4.82],
                        [-0.2, 1.78, 3.10, 3.90, 4.00,  4.43],
                        [-0.9, 1.99, 2.49, 3.76, 3.91,  3.92]]
        self.freq_down = [[2.52, 2.52, 4.29, 6.28, 6.28, 13.14],
                          [2.07, 2.07, 5.35, 6.47, 6.47, 13.77],
                          [1.80, 1.80, 6.34, 6.67, 6.67, 14.40],
                          [1.66, 1.66, 6.67, 6.67, 6.67, 14.68],
                          [1.30, 3.22, 5.51, 6.00, 6.45,  7.07],
                          [1.16, 3.26, 4.98, 5.99, 6.25,  6.86],
                          [0.65, 3.41, 4.22, 5.95, 6.22,  6.41],
                          [-0.35, 3.65, 4.00, 5.82, 5.85, 6.13]]
        self.units = 'eV'
        self.title = 'Iron'
        self.xticks = [0.00, 2.01, 3.43, 4.25, 5.55, 6.13]
        self.xlabels = ['0 0 0', '1/2 1/2 1/2', '1/2 0 0', '0 0 0',
                        '3/4 1/4 3/4', '1/2 0 0']
        self.fermi = [0.17, 0.17]
        #Index at which the abscissa/frequencies are split into subplots
        self.breakpoint = 4

        # Results
        self.fig = plot_dispersion(
            self.abscissa, self.freq_up, self.freq_down, self.units,
            self.title, self.xticks, self.xlabels, self.fermi)
        self.subplots = self.fig.axes

    def tearDown(self):
        # Ensure figures are closed after tests
        matplotlib.pyplot.close('all')

    def test_returns_fig(self):
        self.assertIsInstance(self.fig, figure.Figure)

    def test_has_2_subplots(self):
        self.assertEqual(len(self.subplots), 2)

    def test_freq_xaxis(self):
        bp = self.breakpoint
        # Check x-axis for each subplot separately
        n_correct_x = np.array([0, 0])
        for line in self.subplots[0].get_lines():
            # Check x-axis for first plot, abscissa[0:4]
            if np.array_equal(line.get_data()[0], self.abscissa[:bp]):
                n_correct_x[0] += 1
        for line in self.subplots[1].get_lines():
            # Check x-axis for second plot, abscissa[4:]
            if np.array_equal(line.get_data()[0], self.abscissa[bp:]):
                n_correct_x[1] += 1
        # Check that there are as many lines with abscissa for the x-axis
        # values as there are both freq_up and freq_down branches
        self.assertTrue(np.all(
            n_correct_x == (len(self.freq_up[0]) + len(self.freq_down[0]))))

    def test_freq_yaxis(self):
        bp = self.breakpoint
        all_freq_branches = np.vstack((np.transpose(self.freq_up),
                                       np.transpose(self.freq_down)))
        # Check y-axis for each subplot separately
        n_correct_y = np.array([0, 0])
        # Subplot 0
        for branch in all_freq_branches[:, :bp]:
            for line in self.subplots[0].get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y[0] += 1
                    break
        # Subplot 1
        for branch in all_freq_branches[:, bp:]:
            for line in self.subplots[1].get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y[1] += 1
                    break
        # Check that every branch has a matching y-axis line
        self.assertTrue(np.all(n_correct_y == len(all_freq_branches)))

    def test_fermi_yaxis(self):
        n_correct_y = np.array([0, 0])
        for ef in self.fermi:
            # Subplot 0
            for line in self.subplots[0].get_lines():
                if np.all(np.array(line.get_data()[1]) == ef):
                    n_correct_y[0] += 1
                    break
            # Subplot 1
            for line in self.subplots[0].get_lines():
                if np.all(np.array(line.get_data()[1]) == ef):
                    n_correct_y[1] += 1
                    break
        self.assertTrue(np.all(n_correct_y == len(self.fermi)))

    def test_xaxis_tick_locs(self):
        for subplot in self.subplots:
            npt.assert_array_equal(subplot.get_xticks(), self.xticks)

    def test_xaxis_tick_labels(self):
        ticklabels = [[xlabel.get_text()
                         for xlabel in subplot.get_xticklabels()]
                         for subplot in self.subplots]
        for i, subplot in enumerate(self.subplots):
            npt.assert_array_equal(ticklabels[i], self.xlabels)


class TestPlotDos(unittest.TestCase):

    def setUp(self):
        # Input values
        self.dos = [2.30e-01, 1.82e-01, 8.35e-02, 3.95e-02, 2.68e-02, 3.89e-02,
                    6.15e-02, 6.75e-02, 6.55e-02, 5.12e-02, 3.60e-02, 2.80e-02,
                    5.22e-02, 1.12e-01, 1.52e-01, 1.37e-01, 9.30e-02, 6.32e-02,
                    7.92e-02, 1.32e-01, 1.53e-01, 8.88e-02, 2.26e-02, 2.43e-03,
                    1.08e-04, 2.00e-06, 8.11e-07, 4.32e-05, 9.63e-04, 8.85e-03,
                    3.35e-02, 5.22e-02, 3.35e-02, 8.85e-03, 9.63e-04, 4.32e-05,
                    7.96e-07, 6.81e-09, 9.96e-08, 5.40e-06, 1.21e-04, 1.13e-03,
                    4.71e-03, 1.19e-02, 2.98e-02, 6.07e-02, 6.91e-02, 3.79e-02,
                    9.33e-03, 9.85e-04, 4.40e-05, 2.24e-05, 4.82e-04, 4.43e-03,
                    1.67e-02, 2.61e-02, 1.67e-02, 4.43e-03, 4.82e-04, 2.16e-05,
                    3.98e-07]
        self.dos_down = [6.05e-09, 7.97e-07, 4.33e-05, 9.71e-04, 9.08e-03,
                         3.72e-02, 8.06e-02, 1.37e-01, 1.84e-01, 1.47e-01,
                         7.37e-02, 3.84e-02, 2.67e-02, 3.80e-02, 5.36e-02,
                         4.24e-02, 4.28e-02, 5.76e-02, 5.03e-02, 3.55e-02,
                         2.32e-02, 3.15e-02, 7.39e-02, 1.24e-01, 1.40e-01,
                         1.11e-01, 7.48e-02, 5.04e-02, 5.22e-02, 8.75e-02,
                         1.37e-01, 1.30e-01, 6.37e-02, 1.47e-02, 1.51e-03,
                         1.09e-04, 9.64e-04, 8.85e-03, 3.35e-02, 5.22e-02,
                         3.35e-02, 8.85e-03, 9.63e-04, 4.33e-05, 6.19e-06,
                         1.21e-04, 1.13e-03, 4.71e-03, 1.19e-02, 2.98e-02,
                         6.07e-02, 6.91e-02, 3.79e-02, 9.33e-03, 9.85e-04,
                         4.40e-05, 2.24e-05, 4.82e-04, 4.43e-03, 1.67e-02,
                         2.61e-02]
        self.bins = [ 0.58,  0.78,  0.98,  1.18,  1.38,  1.58,  1.78,  1.98,
                      2.18,  2.38,  2.58,  2.78,  2.98,  3.18,  3.38,  3.58,
                      3.78,  3.98,  4.18,  4.38,  4.58,  4.78,  4.98,  5.18,
                      5.38,  5.58,  5.78,  5.98,  6.18,  6.38,  6.58,  6.78,
                      6.98,  7.18,  7.38,  7.58,  7.78,  7.98,  8.18,  8.38,
                      8.58,  8.78,  8.98,  9.18,  9.38,  9.58,  9.78,  9.98,
                      10.18, 10.38, 10.58, 10.78, 10.98, 11.18, 11.38, 11.58,
                      11.78, 11.98, 12.18, 12.38, 12.58, 12.78]
        self.units = 'eV'
        self.title = 'Iron'
        self.fermi = [4.71, 4.71]
        self.mirror = False

        # Results
        self.fig = plot_dos(
            self.dos, self.dos_down, self.bins, self.units,
            self.title, self.fermi, self.mirror)
        self.ax = self.fig.axes[0]

    def tearDown(self):
        # Ensure figures are closed after tests
        matplotlib.pyplot.close('all')

    def test_returns_fig(self):
        self.assertIsInstance(self.fig, figure.Figure)

    def test_n_series(self):
        # 2 series, 1 for dos, 1 for dos_down
        n_series = 2 + len(self.fermi)
        self.assertEqual(len(self.ax.get_lines()), n_series)

    def test_dos_xaxis(self):
        bin_centres = np.array(self.bins[:-1]) + (self.bins[1]
                                                - self.bins[0])/2
        n_correct_x = 0
        for line in self.ax.get_lines():
            if np.array_equal(line.get_data()[0], bin_centres):
                n_correct_x += 1
        # Check there are exactly 2 lines with bin centres for the x-axis
        # (1 for dos, 1 for dos_down)
        self.assertEqual(n_correct_x, 2)

    def test_dos_yaxis(self):
        match = False
        for line in self.ax.get_lines():
            if np.array_equal(line.get_data()[1], self.dos):
                match = True
        self.assertTrue(match)

    def test_dos_down_yaxis(self):
        match = False
        for line in self.ax.get_lines():
            if np.array_equal(line.get_data()[1], self.dos_down):
                match = True
        self.assertTrue(match)

    def test_fermi_xaxis(self):
        n_correct_x = 0
        for ef in self.fermi:
            for line in self.ax.get_lines():
                if np.all(np.array(line.get_data()[0]) == ef):
                    n_correct_x += 1
                    break
        self.assertEqual(n_correct_x, len(self.fermi))

    def test_mirror_true(self):
        fig = plot_dos(
            self.dos, self.dos_down, self.bins, self.units,
            self.title, self.fermi, mirror=True)
        for line in fig.axes[0].get_lines():
            if np.array_equal(line.get_data()[1], np.negative(self.dos_down)):
                match = True
        self.assertTrue(match)

    def test_empty_dos(self):
        # Test that dos_down is still plotted when dos is empty
        fig = plot_dos(
            [], self.dos_down, self.bins, self.units,
            self.title, self.fermi, self.mirror)
        match = False
        for line in fig.axes[0].get_lines():
            if np.array_equal(line.get_data()[1], self.dos_down):
                match = True
        self.assertTrue(match)

    def test_empty_dos_down(self):
        # Test that dos is still plotted when dos is empty
        fig = plot_dos(
            self.dos, [], self.bins, self.units,
            self.title, self.fermi, self.mirror)
        match = False
        for line in fig.axes[0].get_lines():
            if np.array_equal(line.get_data()[1], self.dos):
                match = True
        self.assertTrue(match)
