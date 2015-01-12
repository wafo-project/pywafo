"""
Data package in WAFO Toolbox.

Contents
--------
atlantic - Significant wave-height data recorded in the Atlantic Ocean
gfaks89  - Surface elevation measured at Gullfaks C 24.12.1989
gfaksr89 - Reconstructed surface elevation measured at Gullfaks C 24.12.1989.
japansea - coastline map of The Japan Sea
northsea - coastline map of The Nortsea
sea      - Surface elevation dataset used in WAT version 1.1.
sfa89    - Wind measurements at Statfjord A 24.12.1989
sn       - Fatigue experiment, constant-amplitude loading.
yura87   - Surface elevation measured off the coast of Yura



This module gives gives detailed information and easy access to all datasets
included in WAFO

"""
from numpy import (loadtxt, nan)
import os
__path2data = os.path.dirname(os.path.realpath(__file__))

__all__ = ['atlantic', 'gfaks89', 'gfaksr89', 'japansea', 'northsea', 'sea',
           'sfa89', 'sn', 'yura87']

_NANS = set(['nan', 'NaN', '-1.#IND00+00', '1.#IND00+00', '-1.#INF00+00'])


def _tofloat(x):
    return nan if x in _NANS else float(x or 0)


_MYCONVERTER = {}
for i in range(2):
    _MYCONVERTER[i] = _tofloat


def _load(file):  # @ReservedAssignment
    """ local load function
    """
    return loadtxt(os.path.join(__path2data, file))


def _loadnan(file):  # @ReservedAssignment
    """ local load function accepting nan's
    """
    return loadtxt(os.path.join(__path2data, file), converters=_MYCONVERTER)


def atlantic():
    """
    Return Significant wave-height data recorded in the Atlantic Ocean

    Data summary
    ------------
    Size             :     582 X 1
    Sampling Rate    :     ~ 14 times a month
    Device           :
    Source           :
    Format           :    ascii

    Description
    ------------
    atlantic.dat contains average significant wave-height data recorded
    approximately 14 times a month in December-February during 7 years and
    at 2 locations in the Atlantic Ocean

    Example
    --------
    >>> import pylab
    >>> import wafo
    >>> Hs = wafo.data.atlantic()
    >>> h = pylab.plot(Hs)

    Acknowledgement:
    ---------------
    This dataset were made available by Dr. David Carter
    and Dr. David Cotton, Satellite Observing Systems, UK.
    """
    return _load('atlantic.dat')


def gfaks89():
    """
    Return Surface elevation measured at Gullfaks C 24.12.1989

    Data summary
    ------------
    Size             :    39000 X 2
    Sampling Rate    :    2.5 Hz
    Device           :    EMI laser
    Source           :    STATOIL
    Format           :    ascii, c1: time c2: surface elevation

    Description
    ------------
    The wave data was measured 24th December 1989 at the Gullfaks C platform
    in the North Sea from 17.00 to 21.20. The period from 20.00 to 20.20
    is missing and contains NaNs.  The water depth of 218 m is
    regarded as deep water for the most important wave components.
    There are two EMI laser sensors named 219 and 220. This data set is
    obtained from sensor 219, which is located in the Northwest
    corner approximately two platform leg diameters away from
    the closest leg.
    Thus the wave elevation is not expected to be significantly
    affected by diffraction effects for incoming waves in the western sector.
    The wind direction for this period is from the south.
    Some difficulties in calibration of the instruments have been reported
    resulting in several consecutive measured values being equal or almost
    equal in the observed data set.

    This dataset is for non-commercial use only.

    Hm0 = 6.8m, Tm02 = 8s, Tp = 10.5

    Example
    -------
    >>> import pylab
    >>> import wafo
    >>> x = wafo.data.gfaks89()
    >>> h = pylab.plot(x[:,0],x[:,1])

    Acknowledgement:
    ---------------
    This dataset were prepared and made available by Dr. S. Haver,
    STATOIL, Norway

    See also
    --------
    gfaksr89, northsea

    """
    return _loadnan('gfaks89.dat')


def gfaksr89():
    """
    Return a reconstruction of surface elevation measured at Gullfaks C
    24.12.1989.


    Data summary
    ------------
    Size             :    39000 X 2
    Sampling Rate    :    2.5 Hz
    Device           :    EMI laser
    Source           :    STATOIL
    Format           :    ascii, c1: time c2: surface elevation

    Description
    -----------
    This is a reconstructed version of the data in the GFAKS89.DAT file.
    The following calls were made to reconstruct the data:

                inds = findoutliers(gfaks89,.02,2,1.23);
            gfaksr89 = reconstruct(gfaks89,inds,6);

    The wave data was measured 24th December 1989 at the Gullfaks C platform
    in the North Sea from 17.00 to 21.20. The period from 20.00 to 20.20
    is missing in the original data.  The  water depth of 218 m is
    regarded as deep water for the most important wave components.
    There are two EMI laser sensors named 219 and 220. This data set is
    obtained from sensor 219, which is located in the Northwest
    corner approximately two platform leg diameters away from
    the closest leg.
    Thus the wave elevation is not expected to be significantly
    affected by diffraction effects for incoming waves in the western sector.
    The wind direction for this period is from the south.
    Some difficulties in calibration of the instruments have been reported
    resulting in several consecutive measured values being equal or almost
    equal in the observed data set.

    Hm0 = 6.8m, Tm02 = 8s, Tp = 10.5


    Example
    -------
    >>> import pylab
    >>> import wafo
    >>> x = wafo.data.gfaksr89()
    >>> h = pylab.plot(x[:,0],x[:,1])


    See also
    --------
    gfaks89
    """
    return _loadnan('gfaksr89.dat')


def japansea():
    """
    Return coastline map of The Japan Sea


    Data summary
    ------------
    Size             :     692 X 2
    Sampling Rate    :
    Device           :
    Source           :    http://crusty.er.usgs.gov/coast/getcoast.html
    Format           :    ascii, c1: longitude c2: latitude

    Description
    -----------
    JAPANSEA.DAT contains data for plotting a map of The Japan Sea.
    The data is obtained from USGS coastline extractor.

    Example:
    -------
    #the map is seen by

    >>> import pylab
    >>> import wafo
    >>> map1 = wafo.data.japansea()
    >>> h = pylab.plot(map1[:,0],map1[:,1])
    >>> lon_loc = [131,132,132,135,139.5,139]
    >>> lat_loc = [46, 43, 40, 35, 38.3, 35.7]
    >>> loc = ['China','Vladivostok','Japan Sea', 'Japan', 'Yura','Tokyo']
    >>> algn = 'right'
    >>> for lon, lat, name in zip(lon_loc,lat_loc,loc):
            pylab.text(lon,lat,name,horizontalalignment=algn)


    # If you have the m_map toolbox (see http://www.ocgy.ubc.ca/~rich/):
    m_proj('lambert','long',[130 148],'lat',[30 48]);
    m_line(map(:,1),map(:,2));
    m_grid('box','fancy','tickdir','out');
    m_text(131,46,'China');
    m_text(132,43,'Vladivostok');
    m_text(132,40,'Japan Sea');
    m_text(135,35,'Japan');
    m_text(139.5,38.3,'Yura');
    m_text(139,35.7,'Tokyo');
    """
    return _loadnan('japansea.dat')


def northsea():
    """
    NORTHSEA  coastline map of The Nortsea

    Data summary
    -------------
    Size             :     60646 X 2
    Sampling Rate    :
    Device           :
    Source           :    http://crusty.er.usgs.gov/coast/getcoast.html
    Format           :    ascii, c1: longitude c2: latitude

    Description
    -----------
    NORTHSEA.DAT contains data for plotting a map of The Northsea.
    The data is obtained from USGS coastline extractor.

    Example
    -------
    # the map is seen by

    >>> import pylab
    >>> import wafo
    >>> map1 = wafo.data.northsea()
    >>> h = pylab.plot(map1[:,0],map1[:,1])
    >>> lon_pltfrm = [1.8,   2.3,  2.,  1.9, 2.6]
    >>> lat_pltfrm = [61.2, 61.2, 59.9, 58.4, 57.7]
    >>> pltfrm = ['Statfjord A', 'Gullfaks C', 'Frigg', 'Sleipner', 'Draupner']
    >>> h = pylab.scatter(lon_pltfrm,lat_pltfrm);
    >>> algn = 'right'
    >>> for lon, lat, name in zip(lon_pltfrm,lat_pltfrm,pltfrm):
            pylab.text(lon,lat,name,horizontalalignment=algn); algn = 'left'


    >>> lon_city = [10.8, 10.8, 5.52, 5.2]
    >>> lat_city = [59.85, 63.4, 58.9, 60.3]
    >>> city = ['Oslo','Trondheim','Stavanger', 'Bergen']
    >>> h = pylab.scatter(lon_city,lat_city);
    >>> algn = 'right'
    >>> for lon, lat, name in zip(lon_city,lat_city,city):
            pylab.text(lon,lat,name,horizontalalignment=algn)

    # If you have the mpl_toolkits.basemap installed
    >>> from mpl_toolkits.basemap import Basemap
    >>> import matplotlib.pyplot as plt

    # setup Lambert Conformal basemap.
    >>> m = Basemap(width=1200000,height=900000,projection='lcc',
            resolution='f',lat_1=56.,lat_2=64,lat_0=58,lon_0=5.)
    >>> m.drawcoastlines()
    >>> h = m.scatter(lon_pltfrm,lat_pltfrm);
    >>> algn = 'right'
    >>> for lon, lat, name in zip(lon_pltfrm,lat_pltfrm,pltfrm):
            m.text(lon,lat,name,horizontalalignment=algn); algn = 'left'
    >>> m.scatter(lon_city,lat_city)
    >>> algn = 'right'
    >>> for lon, lat, name in zip(lon_city,lat_city,city):
            m.text(lon,lat,name,horizontalalignment=algn)
    """
    return _loadnan('northsea.dat')


def sea():
    """
    Return Surface elevation dataset used in WAT version 1.1.

    Data summary
    ------------
    Size             :    9524    X    2
    Sampling Rate    :    4.0 Hz
    Device           :    unknown
    Source           :    unknown
    Format           :    ascii, c1: time c2: surface elevation

    Description
    -----------
    The wave data was used in one of WAFO predecessors, i.e. the Wave
    Analysis Toolbox version 1.1 (WAT)
    Hm0 = 1.9m, Tm02 = 4.0s, Tp2 = 11.5s Tp1=5.6s

    Example
    -------
    >>> import pylab
    >>> import wafo
    >>> x = wafo.data.sea()
    >>> h = pylab.plot(x[:,0],x[:,1])
    """
    return _load('sea.dat')


def sfa89():
    """
    Return Wind measurements at Statfjord A 24.12.1989

    Data summary
    ------------
    Size             :    144 X 3
    Sampling Rate    :     1/600 Hz
    Device           :
    Source           :    DNMI (The Norwegian Meteorological Institute)
    Format           :    ascii, c1: time     (hours)
                              c2: velocity (m/s)
                              c3: direction (degrees)
    Description
    -----------
    The registration of wind speeds at the Gullfaks field
    started up on Statfjord A in 1978 and continued until 1990.
    The dataregistration was transferred to Gullfaks C in Nov 1989.
    Due to some difficulties of the windregistration on Gullfaks C in
    the beginning, they continued to use the registered data from
    Statfjord A.
    The windspeed is measured in  (meter/second), 110 m above mean water
    level (MWL) and the wind direction is given in degrees for the data.
    The data are a mean value of every 10 minutes.
    Wind directions are defined in the meteorological convention, i.e.,
    0 degrees = wind approaching from North, 90 degrees = wind from East, etc.
    This dataset is for non-commercial use only.

     Example
    -------
    >>> import pylab
    >>> import wafo
    >>> x = wafo.data.sfa89()
    >>> h = pylab.plot(x[:,0],x[:,1])

    Acknowledgement
    ----------------
    These data are made available by Knut A. Iden, DNMI.

    See also
    --------
    northsea
    """
    return _load('sfa89.dat')


def sn():
    """
    Return SN Fatigue experiment, constant-amplitude loading.


    Data summary
    ------------
    Size             :    40    X    2
    Source           :    unknown
    Format           :    ascii, c1: Amplitude MPa c2: Number of cycles

    Description
    -----------
    A fatigue experiment with constant amplitudes at five levels:
    10,15,20,25 and 30 MPa. For each level is related 8 observations of
    the number of cycles to failure.

    The origin of the data is unknown.

     Example
    -------
    >>> import pylab
    >>> import wafo
    >>> x = wafo.data.sn()
    >>> h = pylab.plot(x[:,0],x[:,1])

    See also
    --------
    The same data appear in the directory wdemos/itmkurs/
    as SN.mat.

    """
    return _load('sn.dat')


def yura87():
    """
    Return Surface elevation measured off the coast of Yura.


    Data summary
    -----------
      Size             :    85547 X 4
      Sampling Rate    :    1 Hz
      Device           :    ultrasonic wave gauges
      Source           :    SRI, Ministry of Transport, Japan
      Format           :    ascii, c1: time (sec) c2-4: surface elevation (m)

    Description
    -----------
    The wave data was measured at the Poseidon platform
    in the Japan Sea from 24th November 1987 08.12 hours to 25th November
    1987 07.57 hours. Poseidon was located 3 km off the coast of Yura
    in the Yamagata prefecture, in the Japan Sea during the measurements.
    The most important wave components are to some extent influenced by the
    water depth of 42 m. The data are measured with three ultrasonic wave
    gauges located at the sea floor and the relative coordinates of the
    gauges are as follows (x-axis points to the East, y-axis points to
    the North):
                 X (m)    Y (m)
          c2:   -4.93,    25.02
          c3:    5.80,    92.12
          c4:    0.00,     0.00

        This dataset is for non-commercial use only.

        Hm0 = 5.1m, Tm02 = 7.7s, Tp = 12.8s
    Example
    -------
    >>> import pylab
    >>> import wafo
    >>> x = wafo.data.yura87()
    >>> h = pylab.plot(x[:,0],x[:,1])

    Acknowledgement:
    -----------------
    This dataset were prepared and made available by Dr. Sc. H. Tomita,
    Ship Research Institute, Ministry of Transport, Japan.

    See also
    --------
    japansea
    """
    return _load('yura87.dat')
if __name__ == '__main__':
    import doctest
    doctest.testmod()
