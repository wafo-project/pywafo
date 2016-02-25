from six import iteritems
from numpy.testing import (run_module_suite, assert_equal, assert_almost_equal,
                           assert_array_equal, assert_array_almost_equal,
                           TestCase, assert_,  assert_raises,)

import numpy as np
from numpy import array, cos, exp, linspace, pi, sin, diff, arange, ones
from wafo.data import sea
from wafo.misc import (JITImport, Bunch, detrendma, DotDict, findcross, ecross,
                       findextrema, findrfc, rfcfilter, findtp, findtc,
                       findoutliers, common_shape, argsreduce, stirlerr,
                       getshipchar, betaloge,
                       gravity, nextpow2, discretize, polar2cart,
                       cart2polar, tranproc,
                       rotation_matrix, rotate_2d, spaceline,
                       args_flat, sub2index, index2sub, piecewise,
                       parse_kwargs)


def test_JITImport():
    np = JITImport('numpy')
    assert_equal(1.0, np.exp(0))


def test_bunch():
    d = Bunch(test1=1, test2=3)
    assert_equal(1, d.test1)
    assert_equal(3, d.test2)


def test_dotdict():
    d = DotDict(test1=1, test2=3)
    assert_equal(1, d.test1)
    assert_equal(3, d.test2)


def test_detrendma():
    x = linspace(0, 1, 200)
    y = exp(x) + 0.1 * cos(20 * 2 * pi * x)
    y0 = detrendma(y, 20)
    tr = y - y0
    assert_array_almost_equal(
        y0,
        array(
            [-1.05815186e-02, -2.48280355e-02, -7.01800760e-02,
             -1.27193089e-01, -1.71915213e-01, -1.85125121e-01,
             -1.59745361e-01, -1.03571981e-01, -3.62676515e-02,
             1.82219951e-02, 4.09039083e-02, 2.50630186e-02,
             -2.11478040e-02, -7.78521440e-02, -1.21116040e-01,
             -1.32178923e-01, -1.04689244e-01, -4.71541301e-02,
             2.03417510e-02, 7.38826137e-02, 8.95349902e-02,
             6.68738432e-02, 1.46828486e-02, -4.68648556e-02,
             -9.39871606e-02, -1.08465407e-01, -8.46710629e-02,
             -3.17365657e-02, 2.99669288e-02, 7.66864134e-02,
             9.04482283e-02, 6.59902473e-02, 1.27914062e-02,
             -4.85841870e-02, -9.44185349e-02, -1.06987444e-01,
             -8.13964951e-02, -2.74687460e-02, 3.40438793e-02,
             7.94643163e-02, 9.13222681e-02, 6.50922520e-02,
             1.09390148e-02, -5.02028639e-02, -9.47031411e-02,
             -1.05349757e-01, -7.79872833e-02, -2.31196073e-02,
             3.81412653e-02, 8.22178144e-02, 9.21605209e-02,
             6.41850565e-02, 9.13184690e-03, -5.17149253e-02,
             -9.48363260e-02, -1.03549587e-01, -7.44424124e-02,
             -1.86890490e-02, 4.22594607e-02, 8.49486437e-02,
             9.29666543e-02, 6.32740911e-02, 7.37625254e-03,
             -5.31142920e-02, -9.48133620e-02, -1.01584110e-01,
             -7.07607748e-02, -1.41768231e-02, 4.63990484e-02,
             8.76587937e-02, 9.37446001e-02, 6.23650231e-02,
             5.67876495e-03, -5.43947621e-02, -9.46294406e-02,
             -9.94504301e-02, -6.69411601e-02, -9.58252265e-03,
             5.05608316e-02, 9.03505172e-02, 9.44985623e-02,
             6.14637631e-02, 4.04610591e-03, -5.55500040e-02,
             -9.42796647e-02, -9.71455674e-02, -6.29822440e-02,
             -4.90556961e-03, 5.47458452e-02, 9.30263409e-02,
             9.52330253e-02, 6.05764719e-02, 2.48519180e-03,
             -5.65735506e-02, -9.37590405e-02, -9.46664506e-02,
             -5.88825766e-02, -1.45202622e-04, 5.89553685e-02,
             9.56890756e-02, 9.59527629e-02, 5.97095676e-02,
             1.00314001e-03, -5.74587921e-02, -9.30624694e-02,
             -9.20099048e-02, -5.46405701e-02, 4.69953603e-03,
             6.31909369e-02, 9.83418277e-02, 9.66628470e-02,
             5.88697331e-02, -3.92724035e-04, -5.81989687e-02,
             -9.21847386e-02, -8.91726414e-02, -5.02544862e-02,
             9.62981387e-03, 6.74543554e-02, 1.00988010e-01,
             9.73686580e-02, 5.80639242e-02, -1.69485946e-03,
             -5.87871620e-02, -9.11205115e-02, -8.61512458e-02,
             -4.57224228e-02, 1.46470222e-02, 7.17477118e-02,
             1.03631355e-01, 9.80758938e-02, 5.72993780e-02,
             -2.89550192e-03, -5.92162868e-02, -8.98643173e-02,
             -8.29421650e-02, -4.10422999e-02, 1.97527907e-02,
             7.60733908e-02, 1.06275925e-01, 9.87905812e-02,
             5.65836223e-02, -3.98665495e-03, -5.94790815e-02,
             -8.84105398e-02, -7.95416952e-02, -3.62118451e-02,
             2.49490024e-02, 8.04340881e-02, 1.08926128e-01,
             9.95190863e-02, 5.59244846e-02, -4.96008086e-03,
             -5.95680980e-02, -8.67534061e-02, -7.59459673e-02,
             -3.12285782e-02, 3.02378095e-02, 8.48328258e-02,
             1.11586726e-01, 1.00268126e-01, 5.53301029e-02,
             -5.80729079e-03, -5.94756912e-02, -8.48869734e-02,
             -7.21509330e-02, -2.60897955e-02, 3.56216499e-02,
             8.92729678e-02, 1.14262857e-01, 1.01044781e-01,
             5.48089359e-02, -6.51953427e-03, -5.91940075e-02,
             -8.28051165e-02, -6.81523491e-02, -2.07925530e-02,
             4.11032641e-02, 9.37582360e-02, 1.16960041e-01,
             1.13824241e-01, 7.82451609e-02, 2.87461256e-02,
             -1.07566250e-02, -2.01779675e-02, 8.98967999e-03,
             7.03952281e-02, 1.45278564e-01, 2.09706186e-01,
             2.43802139e-01, 2.39414013e-01, 2.03257341e-01,
             1.54325635e-01, 1.16564992e-01, 1.09638547e-01,
             1.41342814e-01, 2.04600808e-01, 2.80191671e-01,
             3.44164010e-01, 3.77073744e-01
             ]))
    assert_array_almost_equal(tr, array([
        1.11058152, 1.11058152, 1.11058152, 1.11058152, 1.11058152,
        1.11058152, 1.11058152, 1.11058152, 1.11058152, 1.11058152,
        1.11058152, 1.11058152, 1.11058152, 1.11058152, 1.11058152,
        1.11058152, 1.11058152, 1.11058152, 1.11058152, 1.11058152,
        1.11599212, 1.12125245, 1.12643866, 1.13166607, 1.13704477,
        1.14263723, 1.14843422, 1.15435845, 1.16029443, 1.16613308,
        1.17181383, 1.17734804, 1.18281471, 1.18833001, 1.19400259,
        1.19989168, 1.20598434, 1.21220048, 1.21842384, 1.22454684,
        1.23051218, 1.23633498, 1.24209697, 1.24791509, 1.25389641,
        1.26009689, 1.26649987, 1.27302256, 1.27954802, 1.28597031,
        1.29223546, 1.29836228, 1.30443522, 1.31057183, 1.31687751,
        1.32340488, 1.3301336, 1.33697825, 1.34382132, 1.35055864,
        1.35713958, 1.36358668, 1.36998697, 1.37645853, 1.38310497,
        1.38997553, 1.39704621, 1.40422902, 1.41140604, 1.41847493,
        1.4253885, 1.43217295, 1.43891784, 1.44574164, 1.45274607,
        1.45997696, 1.46740665, 1.47494469, 1.48247285, 1.48989073,
        1.49715462, 1.50429437, 1.51140198, 1.51859618, 1.52597672,
        1.53358594, 1.54139257, 1.5493038, 1.55720119, 1.56498641,
        1.57261924, 1.58013316, 1.58762252, 1.5952062, 1.60298187,
        1.61098836, 1.6191908, 1.62749412, 1.63577979, 1.64395163,
        1.65197298, 1.65988092, 1.66777202, 1.67576523, 1.68395602,
        1.69237968, 1.70099778, 1.70971307, 1.71840707, 1.72698583,
        1.73541631, 1.74373911, 1.75205298, 1.76047677, 1.76910369,
        1.77796544, 1.78702008, 1.79616827, 1.80529169, 1.81429875,
        1.82316, 1.83191959, 1.84067831, 1.84955481, 1.85863994,
        1.86796178, 1.87747491, 1.88707803, 1.89665308, 1.9061109,
        1.91542572, 1.92464514, 1.9338719, 1.94322436, 1.9527909,
        1.96259596, 1.97259069, 1.9826719, 1.99272195, 2.00265419,
        2.01244653, 2.02215, 2.0318692, 2.04172204, 2.05179437,
        2.06210696, 2.07260759, 2.08319129, 2.09374092, 2.10417247,
        2.11446752, 2.12468051, 2.13491776, 2.14529665, 2.1559004,
        2.16674609, 2.17777817, 2.18889002, 2.19996511, 2.21092214,
        2.22174641, 2.23249567, 2.24327791, 2.25420982, 2.26537192,
        2.2767776, 2.28836802, 2.30003501, 2.3116628, 2.32317284,
        2.33455419, 2.34586786, 2.35722337, 2.36873665, 2.38048542,
        2.39247934, 2.4046564, 2.41690694, 2.42911606, 2.44120808,
        2.44120808, 2.44120808, 2.44120808, 2.44120808, 2.44120808,
        2.44120808, 2.44120808, 2.44120808, 2.44120808, 2.44120808,
        2.44120808, 2.44120808, 2.44120808, 2.44120808, 2.44120808,
        2.44120808, 2.44120808, 2.44120808, 2.44120808, 2.44120808]))


def test_findcross_and_ecross():
    assert_array_equal(findcross([0, 0, 1, -1, 1], 0), np.array([1, 2, 3]))
    assert_array_equal(findcross([0, 1, -1, 1], 0), np.array([0, 1, 2]))

    t = linspace(0, 7 * pi, 250)
    x = sin(t)
    ind = findcross(x, 0.75)
    assert_array_equal(ind, np.array([9, 25, 80, 97, 151, 168, 223, 239]))
    t0 = ecross(t, x, ind, 0.75)
    assert_array_almost_equal(t0, np.array([0.84910514, 2.2933879, 7.13205663,
                                            8.57630119, 13.41484739,
                                            14.85909194,
                                            19.69776067, 21.14204343]))


def test_findextrema():
    t = linspace(0, 7 * pi, 250)
    x = sin(t)
    ind = findextrema(x)
    assert_array_almost_equal(ind, np.array([18, 53, 89, 125, 160, 196, 231]))


def test_findrfc():
    t = linspace(0, 7 * pi, 250)
    x = sin(t) + 0.1 * sin(50 * t)
    ind = findextrema(x)
    assert_array_almost_equal(
        ind,
        np.array(
            [1, 3, 4, 6, 7, 9, 11, 13, 14, 16, 18, 19, 21,
             23, 25, 26, 28, 29, 31, 33, 35, 36, 38, 39, 41, 43,
             45, 46, 48, 50, 51, 53, 55, 56, 58, 60, 61, 63, 65,
             67, 68, 70, 71, 73, 75, 77, 78, 80, 81, 83, 85, 87,
             88, 90, 92, 93, 95, 97, 99, 100, 102, 103, 105, 107, 109,
             110, 112, 113, 115, 117, 119, 120, 122, 124, 125, 127, 129, 131,
             132, 134, 135, 137, 139, 141, 142, 144, 145, 147, 149, 151, 152,
             154, 156, 157, 159, 161, 162, 164, 166, 167, 169, 171, 173, 174,
             176, 177, 179, 181, 183, 184, 186, 187, 189, 191, 193, 194, 196,
             198, 199, 201, 203, 205, 206, 208, 209, 211, 213, 215, 216, 218,
             219, 221, 223, 225, 226, 228, 230, 231, 233, 235, 237, 238, 240,
             241, 243, 245, 247, 248]))
    _ti, tp = t[ind], x[ind]
    ind1 = findrfc(tp, 0.3)
    assert_array_almost_equal(
        ind1,
        np.array([0,  9, 32, 53, 74, 95, 116, 137]))
    assert_array_almost_equal(
        tp[ind1],
        np.array(
            [-0.00743352, 1.08753972, -1.07206545, 1.09550837, -1.07940458,
             1.07849396, -1.0995006, 1.08094452]))


def test_rfcfilter():
    # 1. Filtered signal y is the turning points of x.
    x = sea()
    y = rfcfilter(x[:, 1], h=0, method=1)
    assert_array_almost_equal(
        y[0:5],
        np.array([-1.2004945, 0.83950546, -0.09049454,
                  -0.02049454, -0.09049454]))

    # 2. This removes all rainflow cycles with range less than 0.5.
    y1 = rfcfilter(x[:, 1], h=0.5)
    assert_array_almost_equal(
        y1[0:5],
        np.array([-1.2004945, 0.83950546, -0.43049454,
                  0.34950546, -0.51049454]))

    t = linspace(0, 7 * pi, 250)
    x = sin(t) + 0.1 * sin(50 * t)
    ind = findextrema(x)
    assert_array_almost_equal(
        ind,
        np.array(
            [1,  3,  4,  6,  7,  9, 11, 13, 14, 16, 18, 19, 21,
             23, 25, 26, 28, 29, 31, 33, 35, 36, 38, 39, 41, 43,
             45, 46, 48, 50, 51, 53, 55, 56, 58, 60, 61, 63, 65,
             67, 68, 70, 71, 73, 75, 77, 78, 80, 81, 83, 85, 87,
             88, 90, 92, 93, 95, 97, 99, 100, 102, 103, 105, 107, 109,
             110, 112, 113, 115, 117, 119, 120, 122, 124, 125, 127, 129, 131,
             132, 134, 135, 137, 139, 141, 142, 144, 145, 147, 149, 151, 152,
             154, 156, 157, 159, 161, 162, 164, 166, 167, 169, 171, 173, 174,
             176, 177, 179, 181, 183, 184, 186, 187, 189, 191, 193, 194, 196,
             198, 199, 201, 203, 205, 206, 208, 209, 211, 213, 215, 216, 218,
             219, 221, 223, 225, 226, 228, 230, 231, 233, 235, 237, 238, 240,
             241, 243, 245, 247, 248]))
    _ti, tp = t[ind], x[ind]
    tp03 = rfcfilter(tp, 0.3)
    assert_array_almost_equal(
        tp03,
        np.array(
            [-0.00743352, 1.08753972, -1.07206545, 1.09550837, -1.07940458,
             1.07849396, -1.0995006, 1.08094452, 0.11983423]))


def test_findtp():
    x = sea()
    x1 = x[0:200, :]
    itp = findtp(x1[:, 1], 0, 'Mw')
    itph = findtp(x1[:, 1], 0.3, 'Mw')
    assert_array_almost_equal(
        itp,
        np.array(
            [11, 21, 22, 24, 26, 28, 31, 39, 43, 45, 47, 51, 56,
             64, 70, 78, 82, 84, 89, 94, 101, 108, 119, 131, 141, 148,
             149, 150, 159, 173, 184, 190, 199]))
    assert_array_almost_equal(
        itph,
        np.array(
            [11, 28, 31, 39, 47, 51, 56, 64, 70, 78, 89, 94, 101,
             108, 119, 131, 141, 148, 159, 173, 184, 190, 199]))


def test_findtc():
    x = sea()
    x1 = x[0:200, :]
    itc, iv = findtc(x1[:, 1], 0, 'dw')
    assert_array_almost_equal(
        itc,
        np.array(
            [28, 31, 39, 56, 64, 69, 78, 82, 83, 89, 94, 101, 108,
             119, 131, 140, 148, 159, 173, 184]))
    assert_array_almost_equal(
        iv,
        np.array(
            [19, 29, 34, 53, 60, 67, 76, 81, 82, 84, 90, 99, 103,
             112, 127, 137, 143, 154, 166, 180, 185]))


def test_findoutliers():
    xx = sea()
    dt = diff(xx[:2, 0])
    dcrit = 5 * dt
    ddcrit = 9.81 / 2 * dt * dt
    zcrit = 0
    [inds, indg] = findoutliers(xx[:, 1], zcrit, dcrit, ddcrit, verbose=False)
    assert_array_almost_equal(inds[np.r_[0, 1, 2, -3, -2, -1]],
                              np.array([6,   7,   8, 9509, 9510, 9511]))
    assert_array_almost_equal(indg[np.r_[0, 1, 2, -3, -2, -1]],
                              np.array([0,   1,   2, 9521, 9522, 9523]))


def test_common_shape():
    A = np.ones((4, 1))
    B = 2
    C = np.ones((1, 5)) * 5
    assert_array_equal(common_shape(A, B, C), (4, 5))
    assert_array_equal(common_shape(A, B, C, shape=(3, 4, 1)), (3, 4, 5))
    A = np.ones((4, 1))
    B = 2
    C = np.ones((1, 5)) * 5
    assert_array_equal(common_shape(A, B, C), (4, 5))
    assert_array_equal(common_shape(A, B, C, shape=(3, 4, 1)), (3, 4, 5))


def test_argsreduce():
    A = linspace(0, 19, 20).reshape((4, 5))
    B = 2
    C = range(5)
    cond = np.ones(A.shape)
    [_A1, B1, _C1] = argsreduce(cond, A, B, C)
    assert_equal(B1.shape, (20,))
    cond[2, :] = 0
    [A2, B2, C2] = argsreduce(cond, A, B, C)
    assert_equal(B2.shape, (15,))
    assert_array_equal(A2,
                       np.array([0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
                                8.,  9., 15., 16., 17., 18., 19.]))
    assert_array_equal(
        B2, np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
    assert_array_equal(
        C2, np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))


def test_stirlerr():
    assert_array_almost_equal(stirlerr(range(5)),
                              np.array([np.inf, 0.08106147, 0.0413407,
                                        0.02767793, 0.02079067]))


def test_parse_kwargs():
    opt = dict(arg1=1, arg2=3)
    opt = parse_kwargs(opt, arg1=5)
    assert(opt['arg1'] == 5)
    assert(opt['arg2'] == 3)
    opt2 = dict(arg3=15)

    opt = parse_kwargs(opt, **opt2)
    assert('arg3' not in opt)


def test_getshipchar():
    sc = getshipchar(10, 'service_speed')
    true_sc = dict(beam=29,
                   beamSTD=2.9,
                   draught=9.6,
                   draughtSTD=2.112,
                   length=216,
                   lengthSTD=2.011309883194276,
                   max_deadweight=30969,
                   max_deadweightSTD=3096.9,
                   propeller_diameter=6.761165385916601,
                   propeller_diameterSTD=0.20267047566705432,
                   service_speed=10,
                   service_speedSTD=0)

    for name, val in iteritems(true_sc):
        assert_almost_equal(val, sc[name])


def test_betaloge():
    assert_array_almost_equal(betaloge(3, arange(4)),
                              np.array([np.inf, -1.09861229, -2.48490665,
                                        -3.40119738]))


def test_gravity():
    phi = linspace(0, 45, 5)
    assert_array_almost_equal(gravity(phi),
                              np.array([9.78049, 9.78245014, 9.78803583,
                                        9.79640552, 9.80629387]))


def test_nextpow2():
    assert_equal(nextpow2(10), 4)
    assert_equal(nextpow2(np.arange(5)), 3)


def test_discretize():
    x, y = discretize(np.cos, 0, np.pi, tol=0.01)
    assert_array_almost_equal(
        x,
        np.array(
            [0., 0.19634954, 0.39269908, 0.58904862, 0.78539816,
             0.9817477, 1.17809725, 1.37444679, 1.57079633, 1.76714587,
             1.96349541, 2.15984495, 2.35619449, 2.55254403, 2.74889357,
             2.94524311, 3.14159265]))
    assert_array_almost_equal(
        y, np.array([1.00000000e+00,  9.80785280e-01,
                     9.23879533e-01,
                     8.31469612e-01,  7.07106781e-01,  5.55570233e-01,
                     3.82683432e-01,  1.95090322e-01,  6.12323400e-17,
                     -1.95090322e-01, -3.82683432e-01, -5.55570233e-01,
                     -7.07106781e-01, -8.31469612e-01, -9.23879533e-01,
                     -9.80785280e-01, -1.00000000e+00]))


def test_discretize_adaptive():
    x, y = discretize(np.cos, 0, np.pi, method='adaptive')
    assert_array_almost_equal(
        x,
        np.array(
            [0., 0.19634954, 0.39269908, 0.58904862, 0.78539816,
             0.9817477, 1.17809725, 1.37444679, 1.57079633, 1.76714587,
             1.96349541, 2.15984495, 2.35619449, 2.55254403, 2.74889357,
             2.94524311, 3.14159265]))
    assert_array_almost_equal(
        y,
        np.array(
            [1.00000000e+00,  9.80785280e-01,  9.23879533e-01,
             8.31469612e-01,  7.07106781e-01,  5.55570233e-01,
             3.82683432e-01,  1.95090322e-01,  6.12323400e-17,
             -1.95090322e-01, -3.82683432e-01, -5.55570233e-01,
             -7.07106781e-01, -8.31469612e-01, -9.23879533e-01,
             -9.80785280e-01, -1.00000000e+00]))


def test_polar2cart_n_cart2polar():
    r = 5
    t = linspace(0, pi, 20)
    x, y = polar2cart(t, r)
    assert_array_almost_equal(
        x,
        np.array(
            [5., 4.93180652, 4.72908621, 4.39736876, 3.94570255,
             3.38640786, 2.73474079, 2.00847712, 1.22742744, 0.41289673,
             -0.41289673, -1.22742744, -2.00847712, -2.73474079, -3.38640786,
             -3.94570255, -4.39736876, -4.72908621, -4.93180652, -5.]))
    assert_array_almost_equal(
        y,
        np.array(
            [0.00000000e+00,  8.22972951e-01,  1.62349735e+00,
             2.37973697e+00,  3.07106356e+00,  3.67861955e+00,
             4.18583239e+00,  4.57886663e+00,  4.84700133e+00,
             4.98292247e+00,  4.98292247e+00,  4.84700133e+00,
             4.57886663e+00,  4.18583239e+00,  3.67861955e+00,
             3.07106356e+00,  2.37973697e+00,  1.62349735e+00,
             8.22972951e-01,  6.12323400e-16]))
    ti, ri = cart2polar(x, y)
    assert_array_almost_equal(
        ti,
        np.array(
            [0., 0.16534698, 0.33069396, 0.49604095, 0.66138793,
             0.82673491, 0.99208189, 1.15742887, 1.32277585, 1.48812284,
             1.65346982, 1.8188168, 1.98416378, 2.14951076, 2.31485774,
             2.48020473, 2.64555171, 2.81089869, 2.97624567, 3.14159265]))
    assert_array_almost_equal(
        ri,
        np.array(
            [5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
             5., 5., 5., 5., 5., 5., 5.]))


def test_tranproc():
    import wafo.transform.models as wtm
    tr = wtm.TrHermite()
    x = linspace(-5, 5, 501)
    g = tr(x)
    y0, y1 = tranproc(x, g, range(5), ones(5))
    assert_array_almost_equal(
        y0,
        np.array([0.02659612, 1.00115284, 1.92872532,
                  2.81453257, 3.66292878]))
    assert_array_almost_equal(
        y1,
        np.array([1.00005295, 0.9501118, 0.90589954,
                  0.86643821, 0.83096482]))


class TestPiecewise(TestCase):
    def test_condition_is_single_bool_list(self):
        assert_raises(ValueError, piecewise, [0, 0], [True, False], [1])

    def test_condition_is_list_of_single_bool_list(self):
        x = piecewise([0, 0], [[True, False]], [1])
        assert_array_equal(x, [1, 0])

    def test_conditions_is_list_of_single_bool_array(self):
        x = piecewise([0, 0], [np.array([True, False])], [1])
        assert_array_equal(x, [1, 0])

    def test_condition_is_single_int_array(self):
        assert_raises(ValueError, piecewise, [0, 0], np.array([1, 0]), [1])

    def test_condition_is_list_of_single_int_array(self):
        x = piecewise([0, 0], [np.array([1, 0])], [1])
        assert_array_equal(x, [1, 0])

    def test_simple(self):
        x = piecewise([0, 0], [[False, True]], [lambda x:-1])
        assert_array_equal(x, [0, -1])

        x = piecewise([1, 2], [[True, False], [False, True]], [3, 4])
        assert_array_equal(x, [3, 4])

    def test_default(self):
        # No value specified for x[1], should be 0
        x = piecewise([1, 2], [[True, False]], [2])
        assert_array_equal(x, [2, 0])

        # Should set x[1] to 3
        x = piecewise([1, 2], [[True, False]], [2, 3])
        assert_array_equal(x, [2, 3])

    def test_0d(self):
        x = np.array(3)
        y = piecewise(x, [x > 3], [4, 0])
        assert_(y.ndim == 0)
        assert_(y == 0)

        x = 5
        y = piecewise(x, [[True], [False]], [1, 0])
        assert_(y.ndim == 0)
        assert_(y == 1)

    def test_abs_function(self):
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise((x,), [x < 0, x >= 0], [lambda x: -x, lambda x: x])
        assert_array_equal(vals,
                           [2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

    def test_abs_function_with_scalar(self):
        x = np.array(-2.5)
        vals = piecewise((x,), [x < 0, x >= 0], [lambda x: -x, lambda x: x])
        assert_(vals == 2.5)

    def test_otherwise_condition(self):
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise((x,), [x < 0, ], [lambda x: -x, lambda x: x])
        assert_array_equal(vals, [2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

    def test_passing_further_args_to_fun(self):
        def fun0(x, y, scale=1.):
            return -x*y/scale

        def fun1(x, y, scale=1.):
            return x*y/scale
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise((x,), [x < 0, ], [fun0, fun1], args=(2.,), scale=2.)
        assert_array_equal(vals, [2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

    def test_step_function(self):
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise(x, [x < 0, x >= 0], [-1, 1])
        assert_array_equal(vals, [-1., -1., -1.,  1.,  1.,  1.])

    def test_step_function_with_scalar(self):
        x = 1
        vals = piecewise(x, [x < 0, x >= 0], [-1, 1])
        assert_(vals == 1)

    def test_function_with_two_args(self):
        x = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, x)
        vals = piecewise(
            (X, Y), [X * Y < 0, ], [lambda x, y: -x * y, lambda x, y: x * y])
        assert_array_equal(vals, [[4.,  2., -0.,  2.,  4.],
                                  [2.,  1., -0.,  1.,  2.],
                                  [-0., -0.,  0.,  0.,  0.],
                                  [2.,  1.,  0.,  1.,  2.],
                                  [4.,  2.,  0.,  2.,  4.]])

    def test_fill_value_and_function_with_two_args(self):
        x = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, x)
        vals = piecewise((X, Y), [X * Y < -0.5, X * Y > 0.5],
                         [lambda x, y: -x * y, lambda x, y: x * y],
                         fill_value=np.nan)
        nan = np.nan
        assert_array_equal(vals, [[4.,   2.,  nan,   2.,   4.],
                                  [2.,   1.,  nan,   1.,   2.],
                                  [nan,  nan,  nan,  nan,  nan],
                                  [2.,   1.,  nan,   1.,   2.],
                                  [4.,   2.,  nan,   2.,   4.]])

    def test_fill_value2_and_function_with_two_args(self):
        x = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, x)
        vals = piecewise((X, Y), [X * Y < -0.5, X * Y > 0.5],
                         [lambda x, y: -x * y, lambda x, y: x * y, np.nan])
        nan = np.nan
        assert_array_equal(vals, [[4.,   2.,  nan,   2.,   4.],
                                  [2.,   1.,  nan,   1.,   2.],
                                  [nan,  nan,  nan,  nan,  nan],
                                  [2.,   1.,  nan,   1.,   2.],
                                  [4.,   2.,  nan,   2.,   4.]])


class TestRotationMatrix(TestCase):

    def test_h0_p0_r0(self):
        vals = rotation_matrix(heading=0, pitch=0, roll=0).tolist()
        truevals = [[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]]
        self.assertListEqual(vals, truevals)

    def test_h180_p0_r0(self):
        vals = rotation_matrix(heading=180, pitch=0, roll=0).tolist()
        truevals = [[-1.0, -1.2246467991473532e-16, 0.0],
                    [1.2246467991473532e-16, -1.0, 0.0],
                    [-0.0, 0.0, 1.0]]
        self.assertListEqual(vals, truevals)

    def test_h0_p180_r0(self):
        vals = rotation_matrix(heading=0, pitch=180, roll=0).tolist()
        truevals = [[-1.0, 0.0, 1.2246467991473532e-16],
                    [-0.0, 1.0, 0.0],
                    [-1.2246467991473532e-16, -0.0, -1.0]]
        self.assertListEqual(vals, truevals)

    def test_h0_p0_r180(self):
        vals = rotation_matrix(heading=0, pitch=180, roll=0).tolist()
        truevals = [[-1.0, 0.0, 1.2246467991473532e-16],
                    [-0.0, 1.0, 0.0],
                    [-1.2246467991473532e-16, -0.0, -1.0]]
        self.assertListEqual(vals, truevals)


class TestRotate2d(TestCase):

    def test_rotate_0deg(self):
        vals = list(rotate_2d(x=1, y=0, angle_deg=0))
        truevals = [1.0, 0.0]
        self.assertListEqual(vals, truevals)

    def test_rotate_90deg(self):
        vals = list(rotate_2d(x=1, y=0, angle_deg=90))
        truevals = [6.123233995736766e-17, 1.0]
        self.assertListEqual(vals, truevals)

    def test_rotate_180deg(self):
        vals = list(rotate_2d(x=1, y=0, angle_deg=180))
        truevals = [-1.0, 1.2246467991473532e-16]
        self.assertListEqual(vals, truevals)

    def test_rotate_360deg(self):
        vals = list(rotate_2d(x=1, y=0, angle_deg=360))
        truevals = [1.0, -2.4492935982947064e-16]
        self.assertListEqual(vals, truevals)


class TestSpaceLine(TestCase):

    def test_space_line(self):
        vals = spaceline((2, 0, 0), (3, 0, 0), num=5).tolist()
        truevals = [[2., 0., 0.],
                    [2.25, 0., 0.],
                    [2.5, 0., 0.],
                    [2.75, 0., 0.],
                    [3., 0., 0.]]
        self.assertListEqual(vals, truevals)


class TestArgsFlat(TestCase):

    def test_1_vector_and_2_scalar_args(self):
        x = [1, 2, 3]
        pos, c_shape = args_flat(x, 2, 3)
        truepos = [[1, 2, 3],
                   [2, 2, 3],
                   [3, 2, 3]]
        truec_shape = [3, ]
        self.assertListEqual(pos.tolist(), truepos)
        self.assertListEqual(list(c_shape), truec_shape)

    def test_1_vector_args(self):
        pos1, c_shape1 = args_flat([1, 2, 3])
        truepos1 = [[1, 2, 3]]
        truec_shape1 = None
        self.assertListEqual(pos1.tolist(), truepos1)
        self.assertIs(c_shape1, truec_shape1)

    def test_3_scalar_args(self):
        pos1, c_shape1 = args_flat(1, 2, 3)
        truepos1 = [[1, 2, 3]]
        truec_shape1 = []
        self.assertListEqual(pos1.tolist(), truepos1)
        self.assertListEqual(list(c_shape1), truec_shape1)

    def test_3_scalar_args_version2(self):
        pos1, c_shape1 = args_flat([1], 2, 3)
        truepos1 = [[1, 2, 3]]
        truec_shape1 = [1, ]
        self.assertListEqual(pos1.tolist(), truepos1)
        self.assertListEqual(list(c_shape1), truec_shape1)


class TestSub2index2Sub(TestCase):

    def test_sub2index_and_index2sub(self):
        shape = (3, 3, 4)
        a = np.arange(np.prod(shape)).reshape(shape)
        trueval = a[1, 2, 3]
        order = 'C'
        i = sub2index(shape, 1, 2, 3, order=order)
        self.assertEquals(i, 23)

        val = a.ravel(order)[i]
        self.assertEquals(val, trueval)

        sub = index2sub(shape, i, order=order)
        for j, true_sub_j in enumerate([1, 2, 3]):
            self.assertEquals(sub[j].tolist(), true_sub_j)

if __name__ == '__main__':
    run_module_suite()
