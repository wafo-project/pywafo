from six import iteritems
from numpy.testing import (run_module_suite, assert_equal, assert_almost_equal,
                           assert_array_equal, assert_array_almost_equal,
                           TestCase, assert_, assert_raises,)

import numpy as np
from numpy import array, cos, exp, linspace, pi, sin, diff, arange, ones
from wafo.data import sea
import wafo
from wafo.misc import (JITImport, Bunch, detrendma, DotDict, findcross, ecross,
                       findextrema, findrfc, rfcfilter, findtp, findtc,
                       findrfc_astm,
                       findoutliers, common_shape, argsreduce, stirlerr,
                       getshipchar, betaloge,
                       gravity, nextpow2, discretize, polar2cart,
                       cart2polar, tranproc,
                       rotation_matrix, rotate_2d, spaceline,
                       args_flat, sub2index, index2sub, piecewise,
                       parse_kwargs)


def test_disufq():
    d_inf = [[0., -144.3090093, -269.37681737, -375.20342419, -461.78882978,
              -529.13303412, -577.23603722, -606.09783908, -615.7184397,
              -606.09783908, -577.23603722, -529.13303412, -461.78882978,
              -375.20342419, -269.37681737, -144.3090093, 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0.00000000e+00, 0.00000000e+00, 5.65917684e-01, 2.82958842e+00,
              7.92284757e+00, 1.69775305e+01, 3.11254726e+01, 5.14985092e+01,
              7.92284757e+01, 1.15447207e+02, 1.61286540e+02, 2.17878308e+02,
              2.86354348e+02, 3.67846494e+02, 4.63486583e+02, 5.74406449e+02,
              7.01737928e+02, 8.46612855e+02, 8.46046937e+02, 8.43783266e+02,
              8.38690007e+02, 8.29635324e+02, 8.15487382e+02, 7.95114345e+02,
              7.67384379e+02, 7.31165647e+02, 6.85326315e+02, 6.28734546e+02,
              5.60258507e+02, 4.78766360e+02, 3.83126272e+02, 2.72206406e+02]]

    # depth = 10
    d_10 = [[-3.43299449, -144.58425201, -269.97386241, -376.2314858,
             -463.35503499, -531.34450329, -580.19988853, -609.92118976,
             -620.50840653, -611.96153858, -584.28058577, -537.46554798,
             -471.51642516, -386.43321726, -282.21592426, -158.8601612, 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0.87964472, 3.30251807, 8.7088916, 18.1892694,
             32.87215973, 53.88831991, 82.36912798,
             119.44619211, 166.25122204, 223.91597801, 293.57224772,
             376.35183479, 473.38655268, 585.80822113, 714.74866403,
             861.33970818, 846.05101255, 843.78326617, 838.69000702,
             829.63532408, 815.48738199, 795.11434539, 767.38437889,
             731.16564714, 685.32631478, 628.73454642, 560.25850671,
             478.76636028, 383.12627176, 272.20640579]]

    g = 9.81
    n = 32
    amp = np.ones(n) + 1j * 1
    f = np.linspace(0., 3.0, n // 2 + 1)
    nmin = 2
    nmax = n // 2 + 1
    cases = 1
    ns = n
    w = 2.0 * pi * f
    from wafo import c_library

    d_truth = [d_10, d_inf]
    for i, water_depth in enumerate([10.0, 10000000]):
        kw = wafo.wave_theory.dispersion_relation.w2k(w, 0., water_depth, g)[0]
        data2 = wafo.numba_misc.disufq(amp.real, amp.imag, w, kw, water_depth,
                                       g, nmin, nmax, cases, ns)
        data = c_library.disufq(amp.real, amp.imag, w, kw, water_depth, g,
                                nmin, nmax, cases, ns)

        # print(data[0])
        # print(data[1])
        # deep water
        assert_array_almost_equal(data, data2)
        assert_array_almost_equal(data, d_truth[i])
    # assert(False)


def test_JITImport():
    np = JITImport('numpy')
    assert_equal(1.0, np.exp(0))


def test_bunch():
    d = Bunch(test1=1, test2=3)
    assert_equal(1, getattr(d, 'test1'))
    assert_equal(3, getattr(d, 'test2'))


def test_dotdict():
    d = DotDict(test1=1, test2=3)
    assert_equal(1, d.test1)
    assert_equal(3, d.test2)


def test_detrendma():
    x = linspace(0, 1, 200)
    y = exp(x) + 0.1 * cos(20 * 2 * pi * x)
    y0 = detrendma(y, 20)
    tr = y - y0
    print(y0[::40])
    print(tr[::40])
    assert_array_almost_equal(
        y0[::40], [-0.01058152,  0.09386986,  0.0903801,   0.08510006,
                   0.07803487])
    assert_array_almost_equal(tr[::40], [1.11058152,  1.22796459,  1.50127309,
                                         1.8354286,   2.24397967])


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
    for method in ['clib', 2, 1, 0]:
        ind1 = findrfc(tp, 0.3, method=method)
        if method in [1, 0]:
            ind1 = ind1[:-1]
        assert_array_almost_equal(
            ind1,
            np.array([0, 9, 32, 53, 74, 95, 116, 137]))
        assert_array_almost_equal(
            tp[ind1],
            np.array(
                [-0.00743352, 1.08753972, -1.07206545, 1.09550837, -1.07940458,
                 1.07849396, -1.0995006, 1.08094452]))


def test_rfcfilter():

    # 1. Filtered signal y is the turning points of x.
    x = sea()
    y = rfcfilter(x[:, 1], h=0.0, method=1)
    assert_array_almost_equal(
        y[0:5],
        np.array([-1.2004945, 0.83950546, -0.09049454,
                  -0.02049454, -0.09049454]))

    # 2. This removes all rainflow cycles with range less than 0.5.
    y1 = rfcfilter(x[:, 1], h=0.5, method=0)
    assert_array_almost_equal(
        y1[0:5],
        np.array([-1.2004945, 0.83950546, -0.43049454,
                  0.34950546, -0.51049454]))
    # return
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
    tp03 = rfcfilter(tp, 0.3)
    assert_array_almost_equal(
        tp03,
        np.array(
            [-0.00743352, 1.08753972, -1.07206545, 1.09550837, -1.07940458,
             1.07849396, -1.0995006, 1.08094452, 0.11983423]))

    tp3 = findrfc_astm(tp)
    assert_array_almost_equal((77, 3), tp3.shape)
    # print(tp3[-5:])
    assert_array_almost_equal(tp3[-5:],
                              [[0.01552179, 0.42313414, 1.],
                               [1.09750448, -0.00199612, 0.5],
                               [1.09022256, -0.00927804, 0.5],
                               [0.48055514, 0.60038938, 0.5],
                               [0.03200274, 0.15183698, 0.5]])
    assert_array_almost_equal(tp3[:5],
                              [[0.03578165, 0.28906389, 1.],
                               [0.03602834, 0.56726584, 1.],
                               [0.03816623, 0.76461446, 1.],
                               [0.0638364, 0.92381302, 1.],
                               [0.07759006, 0.99628738, 1.]])

    # assert(False)


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
                              np.array([6, 7, 8, 9509, 9510, 9511]))
    assert_array_almost_equal(indg[np.r_[0, 1, 2, -3, -2, -1]],
                              np.array([0, 1, 2, 9521, 9522, 9523]))


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
    A = np.reshape(linspace(0, 19, 20), (4, 5))
    B = 2
    C = range(5)
    cond = np.ones(A.shape)
    [_A1, B1, _C1] = argsreduce(cond, A, B, C)
    assert_equal(B1.shape, (20,))
    cond[2, :] = 0
    [A2, B2, C2] = argsreduce(cond, A, B, C)
    assert_equal(B2.shape, (15,))
    assert_array_equal(A2,
                       np.array([0., 1., 2., 3., 4., 5., 6., 7.,
                                 8., 9., 15., 16., 17., 18., 19.]))
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
    sc = getshipchar(service_speed=10)
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
        y, np.array([1.00000000e+00, 9.80785280e-01,
                     9.23879533e-01,
                     8.31469612e-01, 7.07106781e-01, 5.55570233e-01,
                     3.82683432e-01, 1.95090322e-01, 6.12323400e-17,
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
            [1.00000000e+00, 9.80785280e-01, 9.23879533e-01,
             8.31469612e-01, 7.07106781e-01, 5.55570233e-01,
             3.82683432e-01, 1.95090322e-01, 6.12323400e-17,
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
            [0.00000000e+00, 8.22972951e-01, 1.62349735e+00,
             2.37973697e+00, 3.07106356e+00, 3.67861955e+00,
             4.18583239e+00, 4.57886663e+00, 4.84700133e+00,
             4.98292247e+00, 4.98292247e+00, 4.84700133e+00,
             4.57886663e+00, 4.18583239e+00, 3.67861955e+00,
             3.07106356e+00, 2.37973697e+00, 1.62349735e+00,
             8.22972951e-01, 6.12323400e-16]))
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
        assert_raises(ValueError, piecewise, [True, False], [1], [0, 0])

    def test_condition_is_list_of_single_bool_list(self):
        x = piecewise([[True, False]], [1], [0, 0])
        assert_array_equal(x, [1, 0])

    def test_conditions_is_list_of_single_bool_array(self):
        x = piecewise([np.array([True, False])], [1], [0, 0])
        assert_array_equal(x, [1, 0])

    def test_condition_is_single_int_array(self):
        assert_raises(ValueError, piecewise, np.array([1, 0]), [1], [0, 0])

    def test_condition_is_list_of_single_int_array(self):
        x = piecewise([np.array([1, 0])], [1], [0, 0])
        assert_array_equal(x, [1, 0])

    def test_simple(self):
        x = piecewise([[False, True]], [lambda x:-1], [0, 0])
        assert_array_equal(x, [0, -1])

        x = piecewise([[True, False], [False, True]], [3, 4], [1, 2])
        assert_array_equal(x, [3, 4])

    def test_default(self):
        # No value specified for x[1], should be 0
        x = piecewise([[True, False]], [2], [1, 2],)
        assert_array_equal(x, [2, 0])

        # Should set x[1] to 3
        x = piecewise([[True, False]], [2, 3], [1, 2])
        assert_array_equal(x, [2, 3])

    def test_0d(self):
        x = np.array(3)
        y = piecewise([x > 3], [4, 0], x)
        assert_(y.ndim == 0)
        assert_(y == 0)

        x = 5
        y = piecewise([[True], [False]], [1, 0], x)
        assert_(y == 1)
        assert_(y.ndim == 0)

    def test_abs_function(self):
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise([x < 0, x >= 0], [lambda x: -x, lambda x: x], (x,))
        assert_array_equal(vals,
                           [2.5, 1.5, 0.5, 0.5, 1.5, 2.5])

    def test_abs_function_with_scalar(self):
        x = np.array(-2.5)
        vals = piecewise([x < 0, x >= 0], [lambda x: -x, lambda x: x], (x,))
        assert_(vals == 2.5)

    def test_otherwise_condition(self):
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise([x < 0, ], [lambda x: -x, lambda x: x], (x,))
        assert_array_equal(vals, [2.5, 1.5, 0.5, 0.5, 1.5, 2.5])

    def test_passing_further_args_to_fun(self):
        def fun0(x, y, scale=1.):
            return -x * y / scale

        def fun1(x, y, scale=1.):
            return x * y / scale
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise([x < 0, ], [fun0, fun1], (x,), args=(2.,), scale=2.)
        assert_array_equal(vals, [2.5, 1.5, 0.5, 0.5, 1.5, 2.5])

    def test_step_function(self):
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise([x < 0, x >= 0], [-1, 1], x)
        assert_array_equal(vals, [-1., -1., -1., 1., 1., 1.])

    def test_step_function_with_scalar(self):
        x = 1
        vals = piecewise([x < 0, x >= 0], [-1, 1], x)
        assert_(vals == 1)

    def test_function_with_two_args(self):
        x = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, x)
        vals = piecewise(
            [X * Y < 0, ], [lambda x, y: -x * y, lambda x, y: x * y], (X, Y))
        assert_array_equal(vals, [[4., 2., -0., 2., 4.],
                                  [2., 1., -0., 1., 2.],
                                  [-0., -0., 0., 0., 0.],
                                  [2., 1., 0., 1., 2.],
                                  [4., 2., 0., 2., 4.]])

    def test_fill_value_and_function_with_two_args(self):
        x = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, x)
        vals = piecewise([X * Y < -0.5, X * Y > 0.5],
                         [lambda x, y: -x * y, lambda x, y: x * y], (X, Y),
                         fill_value=np.nan)
        nan = np.nan
        assert_array_equal(vals, [[4., 2., nan, 2., 4.],
                                  [2., 1., nan, 1., 2.],
                                  [nan, nan, nan, nan, nan],
                                  [2., 1., nan, 1., 2.],
                                  [4., 2., nan, 2., 4.]])

    def test_fill_value2_and_function_with_two_args(self):
        x = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, x)
        vals = piecewise([X * Y < -0.5, X * Y > 0.5],
                         [lambda x, y: -x * y, lambda x, y: x * y, np.nan],
                         (X, Y))
        nan = np.nan
        assert_array_equal(vals, [[4., 2., nan, 2., 4.],
                                  [2., 1., nan, 1., 2.],
                                  [nan, nan, nan, nan, nan],
                                  [2., 1., nan, 1., 2.],
                                  [4., 2., nan, 2., 4.]])


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
