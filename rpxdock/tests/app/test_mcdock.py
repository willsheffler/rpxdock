from rpxdock.app.mcdock import *

def main():
    debug_mcdock_compframes()

def debug_mcdock_compframes():
    sym = 'I 21 3'
    symelems = [wu.sym.SymElem(3, [1, 1, 1])]
    mcsym = McSymmetry(sym, symelems)
    mcsym.celloffset = np.array([0., 0, -1, 0])

    compcoms = np.array([[3.37315983, -12.05649296, 8.71691963, 1.]])

    samp = wu.Bunch(position=np.array([[[0.31607197, -0.52750669, -0.78856528, 11.92298762],
                                        [-0.52750669, -0.78856528, 0.31607197, 11.92298762],
                                        [-0.78856528, 0.31607197, -0.52750669, 99.53378063], [0., 0., 0.,
                                                                                              1.]]]),
                    lattice=np.array([[87.61079301, 0., 0.], [0., 87.61079301, 0.], [0., 0., 87.61079301]]))

    # hide ev; show cgo; show sph; set sphere_scale=0.05

    debug = wu.hxform(wu.hrot([1, 1, 1], [0, 120, 240]), compcoms[0])
    debug = wu.hxformpts(samp.position, debug)
    debug[..., :3] = debug[..., :3] / samp.lattice[0, 0]

    ic(debug)
    ic(compcoms)
    wu.dumppdb('/home/sheffler/project/rpxtal/orig.pdb', debug, frames=np.eye(4))
    sample = mcsym.to_canonical_asu_position(samp, compcoms, debug=debug)

if __name__ == '__main__':
    main()
