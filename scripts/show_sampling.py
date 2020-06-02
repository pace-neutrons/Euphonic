#! /usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import euphonic.sampling

choices_2d = {'golden-square'}
choices_3d = {'golden-sphere', 'spherical-polar-grid'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npts', type=int)
    parser.add_argument('sampling', type=str,
                        choices=(choices_2d | choices_3d))
    args = parser.parse_args()

    if args.sampling in choices_2d:
        fig, ax = plt.subplots()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if args.sampling == 'golden-square':
        ax.plot(*zip(*euphonic.sampling.golden_square(args.npts)), 'o')
    elif args.sampling == 'golden-sphere':
        ax.scatter(*zip(*euphonic.sampling.golden_sphere(args.npts)),
                   marker='x')
    elif args.sampling == 'spherical-polar-grid':
        n_theta = int(np.ceil(np.sqrt(args.npts / 2)))
        npts = n_theta**2 * 2

        if npts != args.npts:
            print("Requested npts ∉ {2x^2, x ∈ Z, x > 1}; "
                  f"rounding up to {npts}.")

        ax.scatter(*zip(*euphonic.sampling.spherical_polar_grid(n_theta * 2,
                                                                n_theta)),
                   marker='x')

    else:
        raise ValueError("Sampling type f{args.sampling} is not implemented.")

    plt.show()

if __name__ == '__main__':
    main()
