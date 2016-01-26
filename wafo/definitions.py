"""
WAFO defintions and numenclature

    crossings :
    cycle_pairs :
    turning_points :
    wave_amplitudes :
    wave_periods :
    waves :

Examples
--------
In order to view the documentation do the following in an ipython window:

>>> import wafo.definitions as wd
>>> wd.crossings()

or
>>> wd.crossings?


"""


def wave_amplitudes():
    r"""
    Wave amplitudes and heights definitions and nomenclature

    Definition of wave amplitudes and wave heights
    ---------------------------------------------

                <----- Direction of wave propagation


               |..............c_..........|
               |             /| \         |
           Hd  |           _/ |  \        |  Hu
       M       |          /   |   \       |
      / \      |     M   / Ac |    \_     |     c_
     F   \     |    / \m/     |      \    |    /  \
    ------d----|---u------------------d---|---u----d------ level v
           \   |  /|                   \  |  /      \L
            \_ | / | At                 \_|_/
              \|/..|                      t
               t

    Parameters
    ----------
    Ac : crest amplitude
    At : trough amplitude
    Hd : wave height as defined for down crossing waves
    Hu : wave height as defined for up crossing waves

    See also
    --------
    waves, crossings, turning_points
    """
    print(wave_amplitudes.__doc__)


def crossings():
    r"""
    Level v crossing definitions and nomenclature

    Definition of level v crossings
    -------------------------------
          M
        .   .                  M                   M
      .      . .             .                   .   .
    F            d               .             .       L
    -----------------------u-------d-------o----------------- level v
                   .     .           .   .   u
                     .                 m
                           m

    Let the letters 'm', 'M', 'F', 'L','d' and 'u' in the
    figure above denote local minimum, maximum, first value, last
    value, down- and up-crossing, respectively. The remaining
    sampled values are indicated with a '.'. Values that are identical
    with v, but do not cross the level is indicated with the letter 'o'.
    We have a level up-crossing at index, k, if

       x(k) <  v and v < x(k+1)
    or if
       x(k) == v and v < x(k+1) and x(r) < v for some di < r <= k-1

    where di is  the index to the previous downcrossing.
    Similarly there is a level down-crossing at index, k, if

       x(k) >  v and v > x(k+1)
    or if
       x(k) == v and v > x(k+1) and x(r) > v  for some ui < r <= k-1

    where ui is  the index to the previous upcrossing.

    The first (F) value is a up crossing if  x(1) = v and x(2) > v.
    Similarly, it is a down crossing if      x(1) = v and x(2) < v.

    See also
    --------
    wave_periods, waves, turning_points, findcross, findtp
    """
    print(crossings.__doc__)


def cycle_pairs():
    r"""
    Cycle pairs definitions and numenclature

    Definition of Max2min and min2Max cycle pair
    --------------------------------------------
    A min2Max cycle pair (mM) is defined as the pair of a minimum
    and the following Maximum. Similarly a Max2min cycle pair (Mm)
    is defined as the pair of a Maximum and the following minimum.
    (all turning points possibly rainflowfiltered before pairing into cycles.)

    See also
    --------
    turning_points
    """
    print(cycle_pairs.__doc__)


def wave_periods():
    r"""
    Wave periods (lengths) definitions and nomenclature

    Definition of wave periods (lengths)
    ------------------------------------


               <----- Direction of wave propagation

                   <-------Tu--------->
                   :                  :
                   <---Tc----->       :
                   :          :       : <------Tcc---->
       M           :      c   :       : :             :
      / \          : M   / \_ :       : c_            c
     F   \         :/ \m/    \:       :/  \          / \
    ------d--------u----------d-------u----d--------u---d-------- level v
           \      /            \     /     :\_    _/:   :\_   L
            \_   /              \_t_/      :  \t_/  :   :  \m/
              \t/                 :        :        :   :
               :                  :        <---Tt--->   :
               <--------Ttt------->        :            :
                                           <-----Td----->
    Tu   = Up crossing period
    Td   = Down crossing period
    Tc   = Crest period, i.e., period between up crossing and
           the next down crossing
    Tt   = Trough period, i.e., period between down crossing and
           the next up crossing
    Ttt  = Trough2trough period
    Tcc  = Crest2crest period


               <----- Direction of wave propagation

                    <--Tcf->                              Tuc
                   :      :               <-Tcb->        <->
       M            :      c               :     :        : :
      / \           : M   / \_             c_    :        : c
     F   \          :/ \m/    \           /  \___:        :/ \
    ------d---------u----------d---------u-------d--------u---d------ level v
          :\_      /            \     __/:        \_    _/     \_   L
          :  \_   /              \_t_/   :          \t_/         \m/
          :    \t/                 :     :
          :     :                  :     :
         <-Ttf->                  <-Ttb->


    Tcf  = Crest front period, i.e., period between up crossing and crest
    Tcb  = Crest back period, i.e., period between crest and down crossing
    Ttf  = Trough front period, i.e., period between down crossing and trough
    Ttb  = Trough back period, i.e., period between trough and up crossing
        Also note that Tcf and Ttf can also be abbreviated by their crossing
        marker, e.g. Tuc (u2c)  and Tdt (d2t), respectively. Similar applies
        to all the other wave periods and wave lengths.

    (The nomenclature for wave length is similar, just substitute T and
     period with L and length, respectively)

                 <----- Direction of wave propagation

                          <--TMm-->
               <-TmM->    :       :
       M       :     :    M       :
      / \      :     M   /:\_     :     M_            M
     F   \     :    / \m/ :  \    :    /: \          / \
          \    :   /      :   \   :   / :  \        /   \
           \   :  /       :    \  :  /  :   \_    _/     \_   L
            \_ : /        :     \_m_/   :     \m_/         \m/
              \m/         :             :      :            :
                          <-----TMM----->      <----Tmm----->


    TmM = Period between minimum and the following Maximum
    TMm = Period between Maximum and the following minimum
    TMM = Period between Maximum and the following Maximum
    Tmm = Period between minimum and the following minimum

    See also
    --------
    waves,
    wave_amplitudes,
    crossings,
    turning_points
    """
    print(wave_periods.__doc__)


def turning_points():
    r"""
    Turning points definitions and numenclature

    Definition of turningpoints
    ---------------------------
                      <----- Direction of wave propagation

       M                  M
      / \       .... M   /:\_           M_            M
     F   \     |    / \m/ :  \         /: \          / \
          \  h |   /      :   \       / :  \        /   \
           \   |  /       :    \     /  :   \_    _/     \_   L
            \_ | /        :     \_m_/   :     \m_/         \m/
              \m/         :             :      :            :
                          <------Mw----->      <-----mw----->

    Local minimum or maximum are indicated with the
    letters 'm' or 'M'. Turning points in this connection are all
    local max (M) and min (m) and the last (L) value and the
    first (F) value if the first local extremum is a max.

    (This choice is made in order to get the exact up-crossing intensity
    from rfc by mm2lc(tp2mm(rfc)) )


    See also
    --------
    waves,
    crossings,
    cycle_pairs
    findtp

    """
    print(turning_points.__doc__)


def waves():
    r"""
    Wave definitions and nomenclature

    Definition of trough and crest
    ------------------------------
    A trough (t) is defined as the global minimum between a
    level v down-crossing (d) and the next up-crossing (u)
    and a crest (c) is defined as the global maximum between a
    level v up-crossing and the following down-crossing.

    Definition of down- and up -crossing waves
    ------------------------------------------
    A level v-down-crossing wave (dw) is a wave from a
    down-crossing to the following down-crossing.
    Similarly, a level v-up-crossing wave (uw) is a wave from an up-crossing
    to the next up-crossing.

    Definition of trough and crest waves
    ------------------------------------
    A trough-to-trough wave (tw) is a wave from a trough (t) to the
    following trough. The crest-to-crest wave (cw) is defined similarly.


    Definition of min2min and Max2Max wave
    --------------------------------------
    A min2min wave (mw) is defined starting from a minimum (m) and
    ending in the following minimum.
    Similarly a Max2Max wave (Mw) is thus a wave from a maximum (M)
    to the next maximum (all waves optionally rainflow filtered).

               <----- Direction of wave propagation


       <------Mw-----> <----mw---->
       M             : :  c       :
      / \            M : / \_     :     c_            c
     F   \          / \m/    \    :    /: \          /:\
    ------d--------u----------d-------u----d--------u---d------ level v
           \      /:           \  :  /: :  :\_    _/  : :\_   L
            \_   / :            \_t_/ : :  :  \t_/    : :  \m/
              \t/  <-------uw---------> :  <-----dw----->
               :                  :     :             :
               <--------tw-------->     <------cw----->

     (F=first value and L=last value).

    See also
    --------
    turning_points,
    crossings,
    wave_periods
    findtc,
    findcross
    """
    print(waves.__doc__)
