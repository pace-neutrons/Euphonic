import unittest
import seekpath
import matplotlib
# Need to set non-interactive backend before importing euphonic to avoid
# DISPLAY requirement when testing plotting functions
matplotlib.use('Agg')
import numpy as np
import numpy.testing as npt
from matplotlib import figure
from euphonic import ureg
from euphonic.plot.dos import plot_dos
from euphonic.plot.dispersion import (calc_abscissa, recip_space_labels,
                                      generic_qpt_labels, get_qpt_label,
                                      plot_dispersion)


class TestCalcAbscissa(unittest.TestCase):

    def test_iron(self):
        recip = [[0., 1.15996339, 1.15996339],
                 [1.15996339, 0., 1.15996339],
                 [1.15996339, 1.15996339, 0.]]
        qpts = [[-0.37500000, -0.45833333, 0.29166667],
                [-0.37500000, -0.37500000, 0.29166667],
                [-0.37500000, -0.37500000, 0.37500000],
                [-0.37500000, 0.45833333, -0.20833333],
                [-0.29166667, -0.45833333, 0.20833333],
                [-0.29166667, -0.45833333, 0.29166667],
                [-0.29166667, -0.37500000, -0.29166667],
                [-0.29166667, -0.37500000, 0.04166667],
                [-0.29166667, -0.37500000, 0.12500000],
                [-0.29166667, -0.37500000, 0.29166667]]
        expected_abscissa = [0., 0.13670299, 0.27340598, 1.48844879,
                             2.75618022, 2.89288323, 3.78930474,
                             4.33611674, 4.47281973, 4.74622573]
        npt.assert_allclose(calc_abscissa(qpts, recip),
                            expected_abscissa)


class TestRecipSpaceLabels(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        NaH = type('', (), {})()
        data = type('', (), {})()
        data.cell_vec = np.array(
            [[0.0, 2.3995, 2.3995],
             [2.3995, 0.0, 2.3995],
             [2.3995, 2.3995, 0.0]])*ureg('angstrom')
        data.ion_r = np.array([[0.5, 0.5, 0.5],
                               [0.0, 0.0, 0.0]])
        data.ion_type = np.array(['H', 'Na'])
        data.qpts = np.array([[-0.25, -0.25, -0.25],
                              [-0.25, -0.50, -0.50],
                              [0.00, -0.25, -0.25],
                              [0.00, 0.00, 0.00],
                              [0.00, -0.50, -0.50],
                              [0.25, 0.00, -0.25],
                              [0.25, -0.50, -0.25],
                              [-0.50, -0.50, -0.50]])
        NaH.data = data
        NaH.expected_labels = ['', '', '', 'X', '', 'W_2', 'L']
        NaH.expected_qpts_with_labels = [0, 1, 2, 4, 5, 6, 7]
        (NaH.labels, NaH.qpts_with_labels) = recip_space_labels(data)
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
        NaH = type('', (), {})()
        cell_vec = [[0.0, 2.3995, 2.3995],
                    [2.3995, 0.0, 2.3995],
                    [2.3995, 2.3995, 0.0]]
        ion_r = [[0.5, 0.5, 0.5],
                 [0.0, 0.0, 0.0]]
        ion_num = [1, 2]
        cell = (cell_vec, ion_r, ion_num)
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
        data = type('', (), {})()
        data._e_units = 'E_h'
        data.qpts = np.array([[0.00, 0.00, 0.00],
                              [0.50, 0.50, 0.50],
                              [0.50, 0.00, 0.00],
                              [0.00, 0.00, 0.00],
                              [0.75, 0.25, -0.25],
                              [0.50, 0.00, 0.00]])
        data.cell_vec = np.array(
            [[-2.708355, 2.708355, 2.708355],
             [2.708355, -2.708355, 2.708355],
             [2.708355, 2.708355, -2.708355]])*ureg('bohr')
        data.freqs = np.array(
            [[-0.13347765, 0.10487180, 0.10490012,
              0.10490012, 0.14500191, 0.14500191],
             [0.00340273, 0.00340273, 0.17054412,
              0.17058441, 0.17058441, 0.52151346],
             [0.00304837, 0.05950495, 0.14329865,
              0.15504453, 0.18419962, 0.18802334],
             [-0.13347765, 0.10487180, 0.10490012,
              0.10490012, 0.14500191, 0.14500191],
             [0.00563753, 0.06967796, 0.10706959,
              0.10708863, 0.13043664, 0.18104762],
             [0.00304837, 0.05950495, 0.14329865,
              0.15504453, 0.18419962, 0.18802334]])*ureg('E_h')
        data.freq_down = np.array(
            [[-0.11824784, 0.16915727, 0.16915728,
              0.16928803, 0.22527386, 0.22527386],
             [0.06095869, 0.06095869, 0.24509240,
              0.24509241, 0.24526573, 0.53965965],
             [0.04785602, 0.11847222, 0.20247348,
              0.22062826, 0.23694067, 0.25963579],
             [-0.11824784, 0.16915727, 0.16915727,
              0.16928803, 0.22527386, 0.22527386],
             [0.03154524, 0.13717388, 0.17082060,
              0.17090552, 0.20002066, 0.25224941],
             [0.04785602, 0.11847221, 0.20247348,
              0.22062826, 0.23694067, 0.25963579]])*ureg('E_h')
        data.fermi = np.array([0.169316, 0.169316])*ureg('E_h')
        self.data = data
        self.title = 'Iron'
        self.expected_abscissa = [0.0, 2.00911553, 3.42977475, 4.24999273,
                                  5.54687123, 6.12685292]*ureg('1/bohr')
        self.expected_xticks = [0.0, 2.00911553, 3.42977475, 4.24999273,
                                5.54687123, 6.12685292]*ureg('1/bohr')
        self.expected_xlabels = ['0 0 0', '1/2 1/2 1/2', '1/2 0 0',
                                 '0 0 0', '3/4 1/4 3/4', '1/2 0 0']

        # Results
        self.fig = plot_dispersion(self.data, self.title)
        self.ax = self.fig.axes[0]

    def tearDown(self):
        # Ensure figures are closed after tests
        matplotlib.pyplot.close('all')

    def test_returns_fig(self):
        self.assertIsInstance(self.fig, figure.Figure)

    def test_n_series(self):
        n_series = (len(self.data.freqs[0])
                    + len(self.data.freq_down[0])
                    + len(self.data.fermi))
        self.assertEqual(len(self.ax.get_lines()), n_series)

    def test_freq_xaxis(self):
        n_correct_x = 0
        expected_abscissa = (self.expected_abscissa.to('1/angstrom')).magnitude
        for line in self.ax.get_lines():
            if (len(line.get_data()[0]) == len(expected_abscissa) and
                    np.allclose(line.get_data()[0], expected_abscissa)):
                n_correct_x += 1
        # Check that there are as many lines with abscissa for the x-axis
        # values as there are both freqs and freq_down branches
        self.assertEqual(n_correct_x,
                         len(self.data.freqs[0]) + len(self.data.freq_down[0]))

    def test_freq_yaxis(self):
        all_freq_branches = np.vstack((
            np.transpose(self.data.freqs.magnitude),
            np.transpose(self.data.freq_down.magnitude)))
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
        for ef in self.data.fermi.magnitude:
            for line in self.ax.get_lines():
                if np.all(np.array(line.get_data()[1]) == ef):
                    n_correct_y += 1
                    break
        self.assertEqual(n_correct_y, len(self.data.fermi))

    def test_xaxis_tick_locs(self):
        expected_xticks = (self.expected_xticks.to('1/angstrom')).magnitude
        npt.assert_allclose(self.ax.get_xticks(), expected_xticks)

    def test_xaxis_tick_labels(self):
        ticklabels = [x.get_text() for x in self.ax.get_xticklabels()]
        npt.assert_array_equal(ticklabels, self.expected_xlabels)

    def test_up_arg(self):
        # Test freqs is plotted and freq_down isn't when down=False
        fig = plot_dispersion(self.data, self.title, down=False)
        n_correct_y = 0
        freqs = self.data.freqs.magnitude
        for branch in np.transpose(freqs):
            for line in fig.axes[0].get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y += 1
                    break
        # Check that every freq up branch has a matching y-axis line
        self.assertEqual(n_correct_y, len(self.data.freqs[0]))
        # Check that freq_down isn't plotted i.e. there are only as many
        # series as there are freqs branches + fermi energies
        n_series = (len(self.data.freqs[0]) + len(self.data.fermi))
        self.assertEqual(len(fig.axes[0].get_lines()), n_series)

    def test_down_arg(self):
        # Test freq down is plotted and freqs isn't when up=False
        fig = plot_dispersion(self.data, self.title, up=False)
        n_correct_y = 0
        freq_down = self.data.freq_down.magnitude
        for branch in np.transpose(freq_down):
            for line in fig.axes[0].get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y += 1
                    break
        # Check that every freq down branch has a matching y-axis line
        self.assertEqual(n_correct_y, len(self.data.freq_down[0]))
        # Check that freqs isn't plotted i.e. there are only as many
        # series as there are freq_down branches + fermi energies
        n_series = (len(self.data.freq_down[0]) + len(self.data.fermi))
        self.assertEqual(len(fig.axes[0].get_lines()), n_series)


class TestPlotDos(unittest.TestCase):

    def setUp(self):
        data = type('', (), {})()
        # Input values
        data._e_units = 'E_h'
        data.dos = np.array(
            [2.30e-01, 1.82e-01, 8.35e-02, 3.95e-02, 2.68e-02, 3.89e-02,
             6.15e-02, 6.75e-02, 6.55e-02, 5.12e-02, 3.60e-02, 2.80e-02,
             5.22e-02, 1.12e-01, 1.52e-01, 1.37e-01, 9.30e-02, 6.32e-02,
             7.92e-02, 1.32e-01, 1.53e-01, 8.88e-02, 2.26e-02, 2.43e-03,
             1.08e-04, 2.00e-06, 8.11e-07, 4.32e-05, 9.63e-04, 8.85e-03,
             3.35e-02, 5.22e-02, 3.35e-02, 8.85e-03, 9.63e-04, 4.32e-05,
             7.96e-07, 6.81e-09, 9.96e-08, 5.40e-06, 1.21e-04, 1.13e-03,
             4.71e-03, 1.19e-02, 2.98e-02, 6.07e-02, 6.91e-02, 3.79e-02,
             9.33e-03, 9.85e-04, 4.40e-05, 2.24e-05, 4.82e-04, 4.43e-03,
             1.67e-02, 2.61e-02, 1.67e-02, 4.43e-03, 4.82e-04, 2.16e-05,
             3.98e-07])
        data.dos_down = np.array(
            [6.05e-09, 7.97e-07, 4.33e-05, 9.71e-04, 9.08e-03,
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
             2.61e-02])
        data.dos_bins = np.array(
            [0.58, 0.78, 0.98, 1.18, 1.38, 1.58, 1.78, 1.98,
             2.18, 2.38, 2.58, 2.78, 2.98, 3.18, 3.38, 3.58,
             3.78, 3.98, 4.18, 4.38, 4.58, 4.78, 4.98, 5.18,
             5.38, 5.58, 5.78, 5.98, 6.18, 6.38, 6.58, 6.78,
             6.98, 7.18, 7.38, 7.58, 7.78, 7.98, 8.18, 8.38,
             8.58, 8.78, 8.98, 9.18, 9.38, 9.58, 9.78, 9.98,
             10.18, 10.38, 10.58, 10.78, 10.98, 11.18, 11.38, 11.58,
             11.78, 11.98, 12.18, 12.38, 12.58, 12.78])*ureg('E_h')
        data.fermi = np.array([4.71, 4.71])*ureg('E_h')
        self.data = data
        self.title = 'Iron'
        self.mirror = False

        # Results
        self.fig = plot_dos(self.data, self.title, self.mirror)
        self.ax = self.fig.axes[0]

    def tearDown(self):
        # Ensure figures are closed after tests
        matplotlib.pyplot.close('all')

    def test_returns_fig(self):
        self.assertIsInstance(self.fig, figure.Figure)

    def test_n_series(self):
        # 2 series, 1 for dos, 1 for dos_down
        n_series = 2 + len(self.data.fermi)
        self.assertEqual(len(self.ax.get_lines()), n_series)

    def test_dos_xaxis(self):
        bin_centres = (
            self.data.dos_bins[:-1]
            + (self.data.dos_bins[1] - self.data.dos_bins[0])/2).magnitude
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
            if np.array_equal(line.get_data()[1], self.data.dos):
                match = True
        self.assertTrue(match)

    def test_dos_down_yaxis(self):
        match = False
        for line in self.ax.get_lines():
            if np.array_equal(line.get_data()[1], self.data.dos_down):
                match = True
        self.assertTrue(match)

    def test_fermi_xaxis(self):
        n_correct_x = 0
        for ef in self.data.fermi:
            for line in self.ax.get_lines():
                if np.all(np.array(line.get_data()[0]) == ef.magnitude):
                    n_correct_x += 1
                    break
        self.assertEqual(n_correct_x, len(self.data.fermi))

    def test_mirror_true(self):
        fig = plot_dos(self.data, self.title, mirror=True)
        for line in fig.axes[0].get_lines():
            if np.array_equal(line.get_data()[1],
                              np.negative(self.data.dos_down)):
                match = True
        self.assertTrue(match)

    def test_up_only(self):
        # Test that only dos plotted when down=False
        fig = plot_dos(self.data, self.title, down=False)
        match = False
        for line in fig.axes[0].get_lines():
            if np.array_equal(line.get_data()[1], self.data.dos):
                match = True
        self.assertTrue(match)
        # Check that dos_down isn't plotted i.e. there are only series for
        # the fermi energies and dos
        n_series = len(self.data.fermi) + 1
        self.assertEqual(len(fig.axes[0].get_lines()), n_series)

    def test_down_only(self):
        # Test that only dos_down plotted when up=False
        fig = plot_dos(self.data, self.title, up=False)
        match = False
        for line in fig.axes[0].get_lines():
            if np.array_equal(line.get_data()[1], self.data.dos_down):
                match = True
        self.assertTrue(match)
        # Check that dos isn't plotted i.e. there are only series for
        # the fermi energies and dos_down
        n_series = len(self.data.fermi) + 1
        self.assertEqual(len(fig.axes[0].get_lines()), n_series)
