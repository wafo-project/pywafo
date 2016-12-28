'''
Created on 20. nov. 2010

@author: pab
'''
from __future__ import division
import unittest
import numpy as np
from numpy.testing import assert_allclose
import wafo.objects as wo
import wafo.kdetools as wk
# import scipy.stats as st


class TestKde(unittest.TestCase):

    def setUp(self):

        # N = 20
        # data = np.random.rayleigh(1, size=(N,))
        self.data = np.array([0.75355792, 0.72779194, 0.94149169, 0.07841119,
                              2.32291887, 1.10419995, 0.77055114, 0.60288273,
                              1.36883635, 1.74754326, 1.09547561, 1.01671133,
                              0.73211143, 0.61891719, 0.75903487, 1.8919469,
                              0.72433808, 1.92973094, 0.44749838, 1.36508452])
        self.x = np.linspace(0, max(self.data) + 1, 10)

    def test0_KDE1D(self):
        data, x = self.data, self.x

        kde0 = wk.KDE(data, hs=0.5, alpha=0.0, inc=16)

        fx = kde0.eval_grid(x)
        assert_allclose(fx, [0.2039735,  0.40252503,  0.54595078,
                             0.52219649,  0.3906213, 0.26381501,  0.16407362,
                             0.08270612,  0.02991145,  0.00720821])

        fx = kde0.eval_grid(x, r=1)
        assert_allclose(-fx, [0.11911419724002906, 0.13440000694772541,
                              0.044400116190638696, -0.0677695267531197,
                              -0.09555596523854318, -0.07498819087690148,
                              -0.06167607128369182, -0.04678588231996062,
                              -0.024515979196411814, -0.008022010381009501])

        fx = kde0.eval_grid(x, r=2)
        assert_allclose(fx, [0.08728138131197069, 0.07558648034784508,
                             0.05093715852686607, 0.07908624791267539,
                             0.10495675573359599, 0.07916167222333347,
                             0.048168330179460386, 0.03438361415806721,
                             0.02197927811015286, 0.009222988165160621])

        ffx = kde0.eval_grid_fast(x)
        assert_allclose(ffx, [0.20729484,  0.39865044,  0.53716945,  0.5169322,
                              0.39060223, 0.26441126,  0.16388801,  0.08388527,
                              0.03227164,  0.00883579], 1e-6)

        fx = kde0.eval_grid_fast(x, r=1)
        assert_allclose(fx, [-0.11582450668441863, -0.12901768780183628,
                             -0.04402464127812092, 0.0636190549560749,
                             0.09345144501310157, 0.07573621607126926,
                             0.06149475587201987, 0.04550210608639078,
                             0.024427027615689087, 0.00885576504750473])

        fx = kde0.eval_grid_fast(x, r=2)
        assert_allclose(fx, [0.08499284131672676, 0.07572564161758065,
                             0.05329987919556978, 0.07849796347259348,
                             0.10232741197885842, 0.07869015379158453,
                             0.049431823916945394, 0.034527256372343613,
                             0.021517998409663567, 0.009527401063843402])

        f = kde0.eval_grid_fast()
        assert_allclose(np.trapz(f, kde0.args),  0.995001)
        assert_allclose(f, [0.011494108953097538, 0.0348546729842836,
                            0.08799292403553607, 0.18568717590587996,
                            0.32473136104523725, 0.46543163412700084,
                            0.5453201564089711, 0.5300582814373698,
                            0.44447650672207173, 0.3411961246641896,
                            0.25103852230993573, 0.17549519961525845,
                            0.11072988772879173, 0.05992730870218242,
                            0.02687783924833738, 0.00974982785617795])

    def test1_TKDE1D(self):
        data = self.data
        x = np.linspace(0.01, max(data) + 1, 10)
        kde = wk.TKDE(data, hs=0.5, L2=0.5)
        f = kde(x)
        assert_allclose(f, [1.03982714,  0.45839018,  0.39514782,  0.32860602,
                            0.26433318, 0.20717946,  0.15907684,  0.1201074,
                            0.08941027,  0.06574882])
        assert_allclose(np.trapz(f, x), 0.94787730659349068)
        f = kde.eval_grid_fast(x)
        assert_allclose(f, [1.0401892415290148, 0.45838973393693677,
                            0.39514689240671547, 0.32860531818532457,
                            0.2643330110605783, 0.20717975528556506,
                            0.15907696844388747, 0.12010770443337843,
                            0.08941129458260941, 0.06574899139165799])
        f = kde.eval_grid_fast2(x)
        assert_allclose(f, [1.0401892415290148, 0.45838973393693677,
                            0.39514689240671547, 0.32860531818532457,
                            0.2643330110605783, 0.20717975528556506,
                            0.15907696844388747, 0.12010770443337843,
                            0.08941129458260941, 0.06574899139165799])
        assert_allclose(np.trapz(f, x), 0.9479438058416647)

    def test1_KDE1D(self):
        data, x = self.data, self.x
        kde = wk.KDE(data, hs=0.5)
        f = kde(x)
        assert_allclose(f, [0.2039735,  0.40252503,  0.54595078,  0.52219649,
                            0.3906213, 0.26381501,  0.16407362,  0.08270612,
                            0.02991145, 0.00720821])

        assert_allclose(np.trapz(f, x), 0.92576174424281876)

    def test2_KDE1D(self):
        # data, x = self.data, self.x

        data = np.asarray([1, 2])
        x = np.linspace(0, max(np.ravel(data)) + 1, 10)
        kde = wk.KDE(data, hs=0.5)
        f = kde(x)
        assert_allclose(f, [0.0541248,  0.16555235,  0.33084399,  0.45293325,
                            0.48345808, 0.48345808,  0.45293325,  0.33084399,
                            0.16555235,  0.0541248])

        assert_allclose(np.trapz(f, x), 0.97323338046725172)
        f0 = kde(output='plot')
        self.assertIsInstance(f0, wo.PlotData)
        assert_allclose(np.trapz(f0.data, f0.args), 0.9319800260106625)

        f0 = kde.eval_grid_fast(output='plot')
        self.assertIsInstance(f0, wo.PlotData)
        assert_allclose(np.trapz(f0.data, f0.args), 0.9319799696210691)

    def test1a_KDE1D(self):
        data, x = self.data, self.x
        kde = wk.KDE(data, hs=0.5, alpha=0.5)
        f = kde(x)
        assert_allclose(f, [0.17252055,  0.41014271,  0.61349072,  0.57023834,
                            0.37198073, 0.21409279,  0.12738463,  0.07460326,
                            0.03956191,  0.01887164])

        assert_allclose(np.trapz(f, x), 0.92938023659047952)

        f0 = kde(output='plot')
        self.assertIsInstance(f0, wo.PlotData)
        assert_allclose(np.trapz(f0.data, f0.args), 0.9871189376720593)

        f0 = kde.eval_grid_fast(output='plot')
        self.assertIsInstance(f0, wo.PlotData)
        assert_allclose(np.trapz(f0.data, f0.args), 0.9962507385131669)

    def test2a_KDE_1D_hs_5_alpha_5(self):
        # data, x = self.data, self.x
        data = np.asarray([1, 2])
        x = np.linspace(0, max(np.ravel(data)) + 1, 10)
        kde = wk.KDE(data, hs=0.5, alpha=0.5)
        f = kde(x)
        assert_allclose(f, [0.0541248,  0.16555235,  0.33084399,  0.45293325,
                            0.48345808, 0.48345808,  0.45293325,  0.33084399,
                            0.16555235,  0.0541248])

        assert_allclose(np.trapz(f, x), 0.97323338046725172)

    def test_KDE2D(self):
        # N = 20
        # data = np.random.rayleigh(1, size=(2, N))
        data = np.array([
            [0.38103275, 0.35083136, 0.90024207, 1.88230239, 0.96815399,
             0.57392873, 1.63367908, 1.20944125, 2.03887811, 0.81789145,
             0.69302049, 1.40856592, 0.92156032, 2.14791432, 2.04373821,
             0.69800708, 0.58428735, 1.59128776, 2.05771405, 0.87021964],
            [1.44080694, 0.39973751, 1.331243, 2.48895822, 1.18894158,
             1.40526085, 1.01967897, 0.81196474, 1.37978932, 2.03334689,
             0.870329, 1.25106862, 0.5346619, 0.47541236, 1.51930093,
             0.58861519, 1.19780448, 0.81548296, 1.56859488, 1.60653533]])

        x = np.linspace(0, max(np.ravel(data)) + 1, 3)

        kde0 = wk.KDE(data, hs=0.5, alpha=0.0, inc=512)

        assert_allclose(kde0.eval_grid(x, x),
                        [[3.27260963e-02, 4.21654678e-02, 5.85338634e-04],
                         [6.78845466e-02, 1.42195839e-01, 1.41676003e-03],
                         [1.39466746e-04, 4.26983850e-03, 2.52736185e-05]])

        t = [[0.0443506097653615, 0.06433530873456418, 0.0041353838654317856],
             [0.07218297149063724, 0.1235819591878892, 0.009288890372002473],
             [0.001613328022214066, 0.00794857884864038, 0.0005874786787715641]
             ]
        assert_allclose(kde0.eval_grid_fast(x, x), t)


class TestRegression(unittest.TestCase):
    def test_KRegression(self):

        N = 51
        x = np.linspace(0, 1, N)
        # ei = np.random.normal(loc=0, scale=0.075, size=(N,))
        ei = [0.0514233500271586, 0.00165101982431131, 0.042827107319028994,
              -0.084351702283385, 0.05978024392552100, -0.07121894535738457,
              0.0855578119920183, -0.0061865198365448, 0.060986773136137415,
              0.0467717713275598, -0.0852368434029634, 0.09790798995780517,
              -0.174003547831554, 0.1100349974247687, 0.12934695904976257,
              -0.036688944487546, -0.0279545148054110, 0.09660222791922815,
              -0.108463847524115, -0.0635162550551463, 0.017192887741329627,
              -0.031520480101878, 0.03939880367791403, -0.06343921941793985,
              0.0574763321274059, -0.1186005160931940, 0.023007133904660495,
              0.0572646924609536, -0.0334012844057809, -0.03444460758658313,
              0.0325434547422866, 0.06063111859444784, 0.0010264474321885913,
              -0.162288671571205, 0.01334616853351956, -0.020490428895193084,
              0.0446047497979159, 0.02924587567502737, 0.021177586536616458,
              0.0634083218094540, -0.1506377646036794, -0.03214553797245153,
              0.1850745187671265, -0.0151240946088902, -0.10599562843454335,
              0.0317357805015679, -0.0736187558312158, 0.04791463883941161,
              0.0660021138871709, -0.1049359954387588, 0.0034961490852392463]
        # print(ei.tolist())
        y0 = 2*np.exp(-x**2/(2*0.3**2))+3*np.exp(-(x-1)**2/(2*0.7**2))
        y = y0 + ei
        kreg = wk.KRegression(x, y)
        f = kreg(output='plotobj', title='Kernel regression', plotflag=1)

        kreg.p = 1
        f1 = kreg(output='plot', title='Kernel regression', plotflag=1)

#         import matplotlib.pyplot as plt
#         plt.figure(0)
#         f.plot(label='p=0')
#         f1.plot(label='p=1')
#         # print(f1.data)
#         plt.plot(x, y, '.', label='data')
#         plt.plot(x, y0, 'k', label='True model')
#         plt.legend()
#         plt.show('hold')

        assert_allclose(f.data[::5],
                        [3.14313544673463, 3.14582567119112, 3.149199078830904,
                         3.153335095194225, 3.15813722171621, 3.16302709623568,
                         3.16631430398602, 3.164138775969285, 3.14947062082316,
                         3.11341295908516, 3.05213808272656, 2.976097561057097,
                         2.908020176929025, 2.867826513276857, 2.8615179445705,
                         2.88155232529645, 2.91307482047679, 2.942469210090470,
                         2.96350144269953, 2.976399025328952, 2.9836554385038,
                         2.987516554300354, 2.9894470264681, 2.990311688080114,
                         2.9906144224522406, 2.9906534916935743])

        print(f1.data[::5].tolist())
        assert_allclose(f1.data[::5],
                        [2.7832831899382, 2.83222307174095, 2.891112685251379,
                         2.9588984473431, 3.03155510969298, 3.1012027219652127,
                         3.1565263737763, 3.18517573180120, 3.177939796091202,
                         3.13336188049535, 3.06057968378847, 2.978164236442354,
                         2.9082732327128, 2.867790922237915, 2.861643209932334,
                         2.88347067948676, 2.92123931823944, 2.96263190368498,
                         2.9985444322015, 3.0243198029657, 3.038629147365635,
                         3.04171702362464, 3.03475567689171, 3.020239732466334,
                         3.002434232424511, 2.987257365211814])

    def test_BKRegression(self):
        # from wafo.kdetools.kdetools import _get_data
        # n = 51
        # loc1 = 0.1
        # scale1 = 0.6
        # scale2 = 0.75
        # x, y, fun1 = _get_data(n, symmetric=True, loc1=loc1,
        #                       scale1=scale1, scale2=scale2)
        # print(x.tolist())
        # print(y.tolist())
        # dist = st.norm
        #  norm1 = scale2 * (dist.pdf(-loc1, loc=-loc1, scale=scale1) +
        #                    dist.pdf(-loc1, loc=loc1, scale=scale1))
        #  def fun1(x):
        #      return (((dist.pdf(x, loc=-loc1, scale=scale1) +
        #               dist.pdf(x, loc=loc1, scale=scale1)) /
        #               norm1).clip(max=1.0))
        x = [-2.9784022156693037, -2.923269270862857, -2.640625797489305,
             -2.592465150170373, -2.5777471766751514, -2.5597898266706323,
             -2.5411937415815604, -2.501753472506631, -2.4939048380402378,
             -2.4747969073957368, -2.3324036659351286, -2.3228634370815,
             -2.230871371173083, -2.21411949373986, -2.2035967461005335,
             -2.1927287694263082, -2.1095391808427064, -2.0942500415622503,
             -2.0774862883018708, -2.0700940505412, -2.054918428555726,
             -1.979624045501378, -1.815804869116454, -1.780636214263252,
             -1.7494324035239686, -1.723149182957688, -1.7180532497996817,
             -1.7016701153705522, -1.6120633534061788, -1.5862592143187193,
             -1.517561220921166, -1.5017798665502253, -1.4895432407186429,
             -1.4470094450898578, -1.4302454657287063, -1.3243060491576388,
             -1.293989140781724, -1.2570066577415648, -1.2332757902347795,
             -1.2306697417054666, -1.0495284321772482, -0.9923351727665026,
             -0.9047559818364217, -0.4092063139968012, -0.3845725606766721,
             -0.30700232234899083, -0.2565844426798063, -0.25415109620097187,
             -0.20223029999069952, -0.10388696244007978, -0.07822191388462896,
             0.07822191388462896, 0.10388696244007978, 0.20223029999069952,
             0.25415109620097187, 0.2565844426798063, 0.30700232234899083,
             0.3845725606766721, 0.4092063139968012, 0.9047559818364217,
             0.9923351727665026, 1.0495284321772482, 1.2306697417054666,
             1.2332757902347795, 1.2570066577415648, 1.293989140781724,
             1.3243060491576388, 1.4302454657287063, 1.4470094450898578,
             1.4895432407186429, 1.5017798665502253, 1.517561220921166,
             1.5862592143187193, 1.6120633534061788, 1.7016701153705522,
             1.7180532497996817, 1.723149182957688, 1.7494324035239686,
             1.780636214263252, 1.815804869116454, 1.979624045501378,
             2.054918428555726, 2.0700940505412, 2.0774862883018708,
             2.0942500415622503, 2.1095391808427064, 2.1927287694263082,
             2.2035967461005335, 2.21411949373986, 2.230871371173083,
             2.3228634370815, 2.3324036659351286, 2.4747969073957368,
             2.4939048380402378, 2.501753472506631, 2.5411937415815604,
             2.5597898266706323, 2.5777471766751514, 2.592465150170373,
             2.640625797489305, 2.923269270862857, 2.9784022156693037]
        y = [False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False]

        bkreg = wk.BKRegression(x, y, a=0.05, b=0.05)
        fbest = bkreg.prb_search_best(hsfun='hste', alpha=0.05, color='g')
        print(fbest.data[::10].tolist())
        assert_allclose(fbest.data[::10],

                        [1.80899736e-15, 0,  6.48351162e-16,  6.61404311e-15,
                         1.10010120e-12, 1.36709203e-10,  1.11994766e-08,
                         5.73040143e-07, 1.68974054e-05,  2.68633448e-04,
                         2.49075176e-03,  1.48687767e-02,  5.98536245e-02,
                         1.74083352e-01,  4.33339557e-01,  8.26039018e-01,
                         9.78387628e-01,  9.98137653e-01,  9.99876002e-01,
                         9.99876002e-01,   9.98137653e-01,  9.78387628e-01,
                         8.26039018e-01,  4.33339557e-01,  1.74083352e-01,
                         5.98536245e-02,  1.48687767e-02,  2.49075176e-03,
                         2.68633448e-04,  1.68974054e-05,  5.73040143e-07,
                         1.11994760e-08,  1.36708818e-10,  1.09965904e-12,
                         5.43806309e-15, 0.0, 0, 0], atol=1e-10)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
