# -*- coding: utf-8 -*-
# Created by makepy.py version 0.5.00
# By python version 2.6.2 (r262:71605, Apr 14 2009, 22:40:02) [MSC v.1500 32 bit (Intel)]
# From type library 'MSO.DLL'
# On Tue Dec 15 09:44:59 2009
"""Microsoft Office 11.0 Object Library"""
makepy_version = '0.5.00'
python_version = 0x20602f0

import win32com.client.CLSIDToClass, pythoncom#, pywintypes
import win32com.client.util
from pywintypes import IID #@UnresolvedImport
from win32com.client import Dispatch

# The following 3 lines may need tweaking for the particular server
# Candidates are pythoncom.Missing, .Empty and .ArgNotFound
defaultNamedOptArg=pythoncom.Empty #@UndefinedVariable
defaultNamedNotOptArg=pythoncom.Empty #@UndefinedVariable
defaultUnnamedArg=pythoncom.Empty #@UndefinedVariable

global_Missing = pythoncom.Missing #@UndefinedVariable
global_IID_IDispatch = pythoncom.IID_IDispatch #@UndefinedVariable
global_IID_IConnectionPointContainer = pythoncom.IID_IConnectionPointContainer #@UndefinedVariable
global_com_error = pythoncom.com_error #@UndefinedVariable
global_error = pythoncom.error #@UndefinedVariable

CLSID = IID('{2DF8D04C-5BFA-101B-BDE5-00AA0044DE52}')
MajorVersion = 2
MinorVersion = 3
LibraryFlags = 8
LCID = 0x0

class constants:
	offPropertyTypeBoolean        =2          # from enum DocProperties
	offPropertyTypeDate           =3          # from enum DocProperties
	offPropertyTypeFloat          =5          # from enum DocProperties
	offPropertyTypeNumber         =1          # from enum DocProperties
	offPropertyTypeString         =4          # from enum DocProperties
	mfHTML                        =2          # from enum MailFormat
	mfPlainText                   =1          # from enum MailFormat
	mfRTF                         =3          # from enum MailFormat
	msoAlertButtonAbortRetryIgnore=2          # from enum MsoAlertButtonType
	msoAlertButtonOK              =0          # from enum MsoAlertButtonType
	msoAlertButtonOKCancel        =1          # from enum MsoAlertButtonType
	msoAlertButtonRetryCancel     =5          # from enum MsoAlertButtonType
	msoAlertButtonYesAllNoCancel  =6          # from enum MsoAlertButtonType
	msoAlertButtonYesNo           =4          # from enum MsoAlertButtonType
	msoAlertButtonYesNoCancel     =3          # from enum MsoAlertButtonType
	msoAlertCancelDefault         =-1         # from enum MsoAlertCancelType
	msoAlertCancelFifth           =4          # from enum MsoAlertCancelType
	msoAlertCancelFirst           =0          # from enum MsoAlertCancelType
	msoAlertCancelFourth          =3          # from enum MsoAlertCancelType
	msoAlertCancelSecond          =1          # from enum MsoAlertCancelType
	msoAlertCancelThird           =2          # from enum MsoAlertCancelType
	msoAlertDefaultFifth          =4          # from enum MsoAlertDefaultType
	msoAlertDefaultFirst          =0          # from enum MsoAlertDefaultType
	msoAlertDefaultFourth         =3          # from enum MsoAlertDefaultType
	msoAlertDefaultSecond         =1          # from enum MsoAlertDefaultType
	msoAlertDefaultThird          =2          # from enum MsoAlertDefaultType
	msoAlertIconCritical          =1          # from enum MsoAlertIconType
	msoAlertIconInfo              =4          # from enum MsoAlertIconType
	msoAlertIconNoIcon            =0          # from enum MsoAlertIconType
	msoAlertIconQuery             =2          # from enum MsoAlertIconType
	msoAlertIconWarning           =3          # from enum MsoAlertIconType
	msoAlignBottoms               =5          # from enum MsoAlignCmd
	msoAlignCenters               =1          # from enum MsoAlignCmd
	msoAlignLefts                 =0          # from enum MsoAlignCmd
	msoAlignMiddles               =4          # from enum MsoAlignCmd
	msoAlignRights                =2          # from enum MsoAlignCmd
	msoAlignTops                  =3          # from enum MsoAlignCmd
	msoAnimationAppear            =32         # from enum MsoAnimationType
	msoAnimationBeginSpeaking     =4          # from enum MsoAnimationType
	msoAnimationCharacterSuccessMajor=6          # from enum MsoAnimationType
	msoAnimationCheckingSomething =103        # from enum MsoAnimationType
	msoAnimationDisappear         =31         # from enum MsoAnimationType
	msoAnimationEmptyTrash        =116        # from enum MsoAnimationType
	msoAnimationGestureDown       =113        # from enum MsoAnimationType
	msoAnimationGestureLeft       =114        # from enum MsoAnimationType
	msoAnimationGestureRight      =19         # from enum MsoAnimationType
	msoAnimationGestureUp         =115        # from enum MsoAnimationType
	msoAnimationGetArtsy          =100        # from enum MsoAnimationType
	msoAnimationGetAttentionMajor =11         # from enum MsoAnimationType
	msoAnimationGetAttentionMinor =12         # from enum MsoAnimationType
	msoAnimationGetTechy          =101        # from enum MsoAnimationType
	msoAnimationGetWizardy        =102        # from enum MsoAnimationType
	msoAnimationGoodbye           =3          # from enum MsoAnimationType
	msoAnimationGreeting          =2          # from enum MsoAnimationType
	msoAnimationIdle              =1          # from enum MsoAnimationType
	msoAnimationListensToComputer =26         # from enum MsoAnimationType
	msoAnimationLookDown          =104        # from enum MsoAnimationType
	msoAnimationLookDownLeft      =105        # from enum MsoAnimationType
	msoAnimationLookDownRight     =106        # from enum MsoAnimationType
	msoAnimationLookLeft          =107        # from enum MsoAnimationType
	msoAnimationLookRight         =108        # from enum MsoAnimationType
	msoAnimationLookUp            =109        # from enum MsoAnimationType
	msoAnimationLookUpLeft        =110        # from enum MsoAnimationType
	msoAnimationLookUpRight       =111        # from enum MsoAnimationType
	msoAnimationPrinting          =18         # from enum MsoAnimationType
	msoAnimationRestPose          =5          # from enum MsoAnimationType
	msoAnimationSaving            =112        # from enum MsoAnimationType
	msoAnimationSearching         =13         # from enum MsoAnimationType
	msoAnimationSendingMail       =25         # from enum MsoAnimationType
	msoAnimationThinking          =24         # from enum MsoAnimationType
	msoAnimationWorkingAtSomething=23         # from enum MsoAnimationType
	msoAnimationWritingNotingSomething=22         # from enum MsoAnimationType
	msoLanguageIDExeMode          =4          # from enum MsoAppLanguageID
	msoLanguageIDHelp             =3          # from enum MsoAppLanguageID
	msoLanguageIDInstall          =1          # from enum MsoAppLanguageID
	msoLanguageIDUI               =2          # from enum MsoAppLanguageID
	msoLanguageIDUIPrevious       =5          # from enum MsoAppLanguageID
	msoArrowheadLengthMedium      =2          # from enum MsoArrowheadLength
	msoArrowheadLengthMixed       =-2         # from enum MsoArrowheadLength
	msoArrowheadLong              =3          # from enum MsoArrowheadLength
	msoArrowheadShort             =1          # from enum MsoArrowheadLength
	msoArrowheadDiamond           =5          # from enum MsoArrowheadStyle
	msoArrowheadNone              =1          # from enum MsoArrowheadStyle
	msoArrowheadOpen              =3          # from enum MsoArrowheadStyle
	msoArrowheadOval              =6          # from enum MsoArrowheadStyle
	msoArrowheadStealth           =4          # from enum MsoArrowheadStyle
	msoArrowheadStyleMixed        =-2         # from enum MsoArrowheadStyle
	msoArrowheadTriangle          =2          # from enum MsoArrowheadStyle
	msoArrowheadNarrow            =1          # from enum MsoArrowheadWidth
	msoArrowheadWide              =3          # from enum MsoArrowheadWidth
	msoArrowheadWidthMedium       =2          # from enum MsoArrowheadWidth
	msoArrowheadWidthMixed        =-2         # from enum MsoArrowheadWidth
	msoShape16pointStar           =94         # from enum MsoAutoShapeType
	msoShape24pointStar           =95         # from enum MsoAutoShapeType
	msoShape32pointStar           =96         # from enum MsoAutoShapeType
	msoShape4pointStar            =91         # from enum MsoAutoShapeType
	msoShape5pointStar            =92         # from enum MsoAutoShapeType
	msoShape8pointStar            =93         # from enum MsoAutoShapeType
	msoShapeActionButtonBackorPrevious=129        # from enum MsoAutoShapeType
	msoShapeActionButtonBeginning =131        # from enum MsoAutoShapeType
	msoShapeActionButtonCustom    =125        # from enum MsoAutoShapeType
	msoShapeActionButtonDocument  =134        # from enum MsoAutoShapeType
	msoShapeActionButtonEnd       =132        # from enum MsoAutoShapeType
	msoShapeActionButtonForwardorNext=130        # from enum MsoAutoShapeType
	msoShapeActionButtonHelp      =127        # from enum MsoAutoShapeType
	msoShapeActionButtonHome      =126        # from enum MsoAutoShapeType
	msoShapeActionButtonInformation=128        # from enum MsoAutoShapeType
	msoShapeActionButtonMovie     =136        # from enum MsoAutoShapeType
	msoShapeActionButtonReturn    =133        # from enum MsoAutoShapeType
	msoShapeActionButtonSound     =135        # from enum MsoAutoShapeType
	msoShapeArc                   =25         # from enum MsoAutoShapeType
	msoShapeBalloon               =137        # from enum MsoAutoShapeType
	msoShapeBentArrow             =41         # from enum MsoAutoShapeType
	msoShapeBentUpArrow           =44         # from enum MsoAutoShapeType
	msoShapeBevel                 =15         # from enum MsoAutoShapeType
	msoShapeBlockArc              =20         # from enum MsoAutoShapeType
	msoShapeCan                   =13         # from enum MsoAutoShapeType
	msoShapeChevron               =52         # from enum MsoAutoShapeType
	msoShapeCircularArrow         =60         # from enum MsoAutoShapeType
	msoShapeCloudCallout          =108        # from enum MsoAutoShapeType
	msoShapeCross                 =11         # from enum MsoAutoShapeType
	msoShapeCube                  =14         # from enum MsoAutoShapeType
	msoShapeCurvedDownArrow       =48         # from enum MsoAutoShapeType
	msoShapeCurvedDownRibbon      =100        # from enum MsoAutoShapeType
	msoShapeCurvedLeftArrow       =46         # from enum MsoAutoShapeType
	msoShapeCurvedRightArrow      =45         # from enum MsoAutoShapeType
	msoShapeCurvedUpArrow         =47         # from enum MsoAutoShapeType
	msoShapeCurvedUpRibbon        =99         # from enum MsoAutoShapeType
	msoShapeDiamond               =4          # from enum MsoAutoShapeType
	msoShapeDonut                 =18         # from enum MsoAutoShapeType
	msoShapeDoubleBrace           =27         # from enum MsoAutoShapeType
	msoShapeDoubleBracket         =26         # from enum MsoAutoShapeType
	msoShapeDoubleWave            =104        # from enum MsoAutoShapeType
	msoShapeDownArrow             =36         # from enum MsoAutoShapeType
	msoShapeDownArrowCallout      =56         # from enum MsoAutoShapeType
	msoShapeDownRibbon            =98         # from enum MsoAutoShapeType
	msoShapeExplosion1            =89         # from enum MsoAutoShapeType
	msoShapeExplosion2            =90         # from enum MsoAutoShapeType
	msoShapeFlowchartAlternateProcess=62         # from enum MsoAutoShapeType
	msoShapeFlowchartCard         =75         # from enum MsoAutoShapeType
	msoShapeFlowchartCollate      =79         # from enum MsoAutoShapeType
	msoShapeFlowchartConnector    =73         # from enum MsoAutoShapeType
	msoShapeFlowchartData         =64         # from enum MsoAutoShapeType
	msoShapeFlowchartDecision     =63         # from enum MsoAutoShapeType
	msoShapeFlowchartDelay        =84         # from enum MsoAutoShapeType
	msoShapeFlowchartDirectAccessStorage=87         # from enum MsoAutoShapeType
	msoShapeFlowchartDisplay      =88         # from enum MsoAutoShapeType
	msoShapeFlowchartDocument     =67         # from enum MsoAutoShapeType
	msoShapeFlowchartExtract      =81         # from enum MsoAutoShapeType
	msoShapeFlowchartInternalStorage=66         # from enum MsoAutoShapeType
	msoShapeFlowchartMagneticDisk =86         # from enum MsoAutoShapeType
	msoShapeFlowchartManualInput  =71         # from enum MsoAutoShapeType
	msoShapeFlowchartManualOperation=72         # from enum MsoAutoShapeType
	msoShapeFlowchartMerge        =82         # from enum MsoAutoShapeType
	msoShapeFlowchartMultidocument=68         # from enum MsoAutoShapeType
	msoShapeFlowchartOffpageConnector=74         # from enum MsoAutoShapeType
	msoShapeFlowchartOr           =78         # from enum MsoAutoShapeType
	msoShapeFlowchartPredefinedProcess=65         # from enum MsoAutoShapeType
	msoShapeFlowchartPreparation  =70         # from enum MsoAutoShapeType
	msoShapeFlowchartProcess      =61         # from enum MsoAutoShapeType
	msoShapeFlowchartPunchedTape  =76         # from enum MsoAutoShapeType
	msoShapeFlowchartSequentialAccessStorage=85         # from enum MsoAutoShapeType
	msoShapeFlowchartSort         =80         # from enum MsoAutoShapeType
	msoShapeFlowchartStoredData   =83         # from enum MsoAutoShapeType
	msoShapeFlowchartSummingJunction=77         # from enum MsoAutoShapeType
	msoShapeFlowchartTerminator   =69         # from enum MsoAutoShapeType
	msoShapeFoldedCorner          =16         # from enum MsoAutoShapeType
	msoShapeHeart                 =21         # from enum MsoAutoShapeType
	msoShapeHexagon               =10         # from enum MsoAutoShapeType
	msoShapeHorizontalScroll      =102        # from enum MsoAutoShapeType
	msoShapeIsoscelesTriangle     =7          # from enum MsoAutoShapeType
	msoShapeLeftArrow             =34         # from enum MsoAutoShapeType
	msoShapeLeftArrowCallout      =54         # from enum MsoAutoShapeType
	msoShapeLeftBrace             =31         # from enum MsoAutoShapeType
	msoShapeLeftBracket           =29         # from enum MsoAutoShapeType
	msoShapeLeftRightArrow        =37         # from enum MsoAutoShapeType
	msoShapeLeftRightArrowCallout =57         # from enum MsoAutoShapeType
	msoShapeLeftRightUpArrow      =40         # from enum MsoAutoShapeType
	msoShapeLeftUpArrow           =43         # from enum MsoAutoShapeType
	msoShapeLightningBolt         =22         # from enum MsoAutoShapeType
	msoShapeLineCallout1          =109        # from enum MsoAutoShapeType
	msoShapeLineCallout1AccentBar =113        # from enum MsoAutoShapeType
	msoShapeLineCallout1BorderandAccentBar=121        # from enum MsoAutoShapeType
	msoShapeLineCallout1NoBorder  =117        # from enum MsoAutoShapeType
	msoShapeLineCallout2          =110        # from enum MsoAutoShapeType
	msoShapeLineCallout2AccentBar =114        # from enum MsoAutoShapeType
	msoShapeLineCallout2BorderandAccentBar=122        # from enum MsoAutoShapeType
	msoShapeLineCallout2NoBorder  =118        # from enum MsoAutoShapeType
	msoShapeLineCallout3          =111        # from enum MsoAutoShapeType
	msoShapeLineCallout3AccentBar =115        # from enum MsoAutoShapeType
	msoShapeLineCallout3BorderandAccentBar=123        # from enum MsoAutoShapeType
	msoShapeLineCallout3NoBorder  =119        # from enum MsoAutoShapeType
	msoShapeLineCallout4          =112        # from enum MsoAutoShapeType
	msoShapeLineCallout4AccentBar =116        # from enum MsoAutoShapeType
	msoShapeLineCallout4BorderandAccentBar=124        # from enum MsoAutoShapeType
	msoShapeLineCallout4NoBorder  =120        # from enum MsoAutoShapeType
	msoShapeMixed                 =-2         # from enum MsoAutoShapeType
	msoShapeMoon                  =24         # from enum MsoAutoShapeType
	msoShapeNoSymbol              =19         # from enum MsoAutoShapeType
	msoShapeNotPrimitive          =138        # from enum MsoAutoShapeType
	msoShapeNotchedRightArrow     =50         # from enum MsoAutoShapeType
	msoShapeOctagon               =6          # from enum MsoAutoShapeType
	msoShapeOval                  =9          # from enum MsoAutoShapeType
	msoShapeOvalCallout           =107        # from enum MsoAutoShapeType
	msoShapeParallelogram         =2          # from enum MsoAutoShapeType
	msoShapePentagon              =51         # from enum MsoAutoShapeType
	msoShapePlaque                =28         # from enum MsoAutoShapeType
	msoShapeQuadArrow             =39         # from enum MsoAutoShapeType
	msoShapeQuadArrowCallout      =59         # from enum MsoAutoShapeType
	msoShapeRectangle             =1          # from enum MsoAutoShapeType
	msoShapeRectangularCallout    =105        # from enum MsoAutoShapeType
	msoShapeRegularPentagon       =12         # from enum MsoAutoShapeType
	msoShapeRightArrow            =33         # from enum MsoAutoShapeType
	msoShapeRightArrowCallout     =53         # from enum MsoAutoShapeType
	msoShapeRightBrace            =32         # from enum MsoAutoShapeType
	msoShapeRightBracket          =30         # from enum MsoAutoShapeType
	msoShapeRightTriangle         =8          # from enum MsoAutoShapeType
	msoShapeRoundedRectangle      =5          # from enum MsoAutoShapeType
	msoShapeRoundedRectangularCallout=106        # from enum MsoAutoShapeType
	msoShapeSmileyFace            =17         # from enum MsoAutoShapeType
	msoShapeStripedRightArrow     =49         # from enum MsoAutoShapeType
	msoShapeSun                   =23         # from enum MsoAutoShapeType
	msoShapeTrapezoid             =3          # from enum MsoAutoShapeType
	msoShapeUTurnArrow            =42         # from enum MsoAutoShapeType
	msoShapeUpArrow               =35         # from enum MsoAutoShapeType
	msoShapeUpArrowCallout        =55         # from enum MsoAutoShapeType
	msoShapeUpDownArrow           =38         # from enum MsoAutoShapeType
	msoShapeUpDownArrowCallout    =58         # from enum MsoAutoShapeType
	msoShapeUpRibbon              =97         # from enum MsoAutoShapeType
	msoShapeVerticalScroll        =101        # from enum MsoAutoShapeType
	msoShapeWave                  =103        # from enum MsoAutoShapeType
	msoAutomationSecurityByUI     =2          # from enum MsoAutomationSecurity
	msoAutomationSecurityForceDisable=3          # from enum MsoAutomationSecurity
	msoAutomationSecurityLow      =1          # from enum MsoAutomationSecurity
	msoBalloonButtonAbort         =-8         # from enum MsoBalloonButtonType
	msoBalloonButtonBack          =-5         # from enum MsoBalloonButtonType
	msoBalloonButtonCancel        =-2         # from enum MsoBalloonButtonType
	msoBalloonButtonClose         =-12        # from enum MsoBalloonButtonType
	msoBalloonButtonIgnore        =-9         # from enum MsoBalloonButtonType
	msoBalloonButtonNext          =-6         # from enum MsoBalloonButtonType
	msoBalloonButtonNo            =-4         # from enum MsoBalloonButtonType
	msoBalloonButtonNull          =0          # from enum MsoBalloonButtonType
	msoBalloonButtonOK            =-1         # from enum MsoBalloonButtonType
	msoBalloonButtonOptions       =-14        # from enum MsoBalloonButtonType
	msoBalloonButtonRetry         =-7         # from enum MsoBalloonButtonType
	msoBalloonButtonSearch        =-10        # from enum MsoBalloonButtonType
	msoBalloonButtonSnooze        =-11        # from enum MsoBalloonButtonType
	msoBalloonButtonTips          =-13        # from enum MsoBalloonButtonType
	msoBalloonButtonYes           =-3         # from enum MsoBalloonButtonType
	msoBalloonButtonYesToAll      =-15        # from enum MsoBalloonButtonType
	msoBalloonErrorBadCharacter   =8          # from enum MsoBalloonErrorType
	msoBalloonErrorBadPictureRef  =4          # from enum MsoBalloonErrorType
	msoBalloonErrorBadReference   =5          # from enum MsoBalloonErrorType
	msoBalloonErrorButtonModeless =7          # from enum MsoBalloonErrorType
	msoBalloonErrorButtonlessModal=6          # from enum MsoBalloonErrorType
	msoBalloonErrorCOMFailure     =9          # from enum MsoBalloonErrorType
	msoBalloonErrorCharNotTopmostForModal=10         # from enum MsoBalloonErrorType
	msoBalloonErrorNone           =0          # from enum MsoBalloonErrorType
	msoBalloonErrorOther          =1          # from enum MsoBalloonErrorType
	msoBalloonErrorOutOfMemory    =3          # from enum MsoBalloonErrorType
	msoBalloonErrorTooBig         =2          # from enum MsoBalloonErrorType
	msoBalloonErrorTooManyControls=11         # from enum MsoBalloonErrorType
	msoBalloonTypeBullets         =1          # from enum MsoBalloonType
	msoBalloonTypeButtons         =0          # from enum MsoBalloonType
	msoBalloonTypeNumbers         =2          # from enum MsoBalloonType
	msoBarBottom                  =3          # from enum MsoBarPosition
	msoBarFloating                =4          # from enum MsoBarPosition
	msoBarLeft                    =0          # from enum MsoBarPosition
	msoBarMenuBar                 =6          # from enum MsoBarPosition
	msoBarPopup                   =5          # from enum MsoBarPosition
	msoBarRight                   =2          # from enum MsoBarPosition
	msoBarTop                     =1          # from enum MsoBarPosition
	msoBarNoChangeDock            =16         # from enum MsoBarProtection
	msoBarNoChangeVisible         =8          # from enum MsoBarProtection
	msoBarNoCustomize             =1          # from enum MsoBarProtection
	msoBarNoHorizontalDock        =64         # from enum MsoBarProtection
	msoBarNoMove                  =4          # from enum MsoBarProtection
	msoBarNoProtection            =0          # from enum MsoBarProtection
	msoBarNoResize                =2          # from enum MsoBarProtection
	msoBarNoVerticalDock          =32         # from enum MsoBarProtection
	msoBarRowFirst                =0          # from enum MsoBarRow
	msoBarRowLast                 =-1         # from enum MsoBarRow
	msoBarTypeMenuBar             =1          # from enum MsoBarType
	msoBarTypeNormal              =0          # from enum MsoBarType
	msoBarTypePopup               =2          # from enum MsoBarType
	msoBlackWhiteAutomatic        =1          # from enum MsoBlackWhiteMode
	msoBlackWhiteBlack            =8          # from enum MsoBlackWhiteMode
	msoBlackWhiteBlackTextAndLine =6          # from enum MsoBlackWhiteMode
	msoBlackWhiteDontShow         =10         # from enum MsoBlackWhiteMode
	msoBlackWhiteGrayOutline      =5          # from enum MsoBlackWhiteMode
	msoBlackWhiteGrayScale        =2          # from enum MsoBlackWhiteMode
	msoBlackWhiteHighContrast     =7          # from enum MsoBlackWhiteMode
	msoBlackWhiteInverseGrayScale =4          # from enum MsoBlackWhiteMode
	msoBlackWhiteLightGrayScale   =3          # from enum MsoBlackWhiteMode
	msoBlackWhiteMixed            =-2         # from enum MsoBlackWhiteMode
	msoBlackWhiteWhite            =9          # from enum MsoBlackWhiteMode
	msoButtonSetAbortRetryIgnore  =10         # from enum MsoButtonSetType
	msoButtonSetBackClose         =6          # from enum MsoButtonSetType
	msoButtonSetBackNextClose     =8          # from enum MsoButtonSetType
	msoButtonSetBackNextSnooze    =12         # from enum MsoButtonSetType
	msoButtonSetCancel            =2          # from enum MsoButtonSetType
	msoButtonSetNextClose         =7          # from enum MsoButtonSetType
	msoButtonSetNone              =0          # from enum MsoButtonSetType
	msoButtonSetOK                =1          # from enum MsoButtonSetType
	msoButtonSetOkCancel          =3          # from enum MsoButtonSetType
	msoButtonSetRetryCancel       =9          # from enum MsoButtonSetType
	msoButtonSetSearchClose       =11         # from enum MsoButtonSetType
	msoButtonSetTipsOptionsClose  =13         # from enum MsoButtonSetType
	msoButtonSetYesAllNoCancel    =14         # from enum MsoButtonSetType
	msoButtonSetYesNo             =4          # from enum MsoButtonSetType
	msoButtonSetYesNoCancel       =5          # from enum MsoButtonSetType
	msoButtonDown                 =-1         # from enum MsoButtonState
	msoButtonMixed                =2          # from enum MsoButtonState
	msoButtonUp                   =0          # from enum MsoButtonState
	msoButtonAutomatic            =0          # from enum MsoButtonStyle
	msoButtonCaption              =2          # from enum MsoButtonStyle
	msoButtonIcon                 =1          # from enum MsoButtonStyle
	msoButtonIconAndCaption       =3          # from enum MsoButtonStyle
	msoButtonIconAndCaptionBelow  =11         # from enum MsoButtonStyle
	msoButtonIconAndWrapCaption   =7          # from enum MsoButtonStyle
	msoButtonIconAndWrapCaptionBelow=15         # from enum MsoButtonStyle
	msoButtonWrapCaption          =14         # from enum MsoButtonStyle
	msoButtonTextBelow            =8          # from enum MsoButtonStyleHidden
	msoButtonWrapText             =4          # from enum MsoButtonStyleHidden
	msoCalloutAngle30             =2          # from enum MsoCalloutAngleType
	msoCalloutAngle45             =3          # from enum MsoCalloutAngleType
	msoCalloutAngle60             =4          # from enum MsoCalloutAngleType
	msoCalloutAngle90             =5          # from enum MsoCalloutAngleType
	msoCalloutAngleAutomatic      =1          # from enum MsoCalloutAngleType
	msoCalloutAngleMixed          =-2         # from enum MsoCalloutAngleType
	msoCalloutDropBottom          =4          # from enum MsoCalloutDropType
	msoCalloutDropCenter          =3          # from enum MsoCalloutDropType
	msoCalloutDropCustom          =1          # from enum MsoCalloutDropType
	msoCalloutDropMixed           =-2         # from enum MsoCalloutDropType
	msoCalloutDropTop             =2          # from enum MsoCalloutDropType
	msoCalloutFour                =4          # from enum MsoCalloutType
	msoCalloutMixed               =-2         # from enum MsoCalloutType
	msoCalloutOne                 =1          # from enum MsoCalloutType
	msoCalloutThree               =3          # from enum MsoCalloutType
	msoCalloutTwo                 =2          # from enum MsoCalloutType
	msoCharacterSetArabic         =1          # from enum MsoCharacterSet
	msoCharacterSetCyrillic       =2          # from enum MsoCharacterSet
	msoCharacterSetEnglishWesternEuropeanOtherLatinScript=3          # from enum MsoCharacterSet
	msoCharacterSetGreek          =4          # from enum MsoCharacterSet
	msoCharacterSetHebrew         =5          # from enum MsoCharacterSet
	msoCharacterSetJapanese       =6          # from enum MsoCharacterSet
	msoCharacterSetKorean         =7          # from enum MsoCharacterSet
	msoCharacterSetMultilingualUnicode=8          # from enum MsoCharacterSet
	msoCharacterSetSimplifiedChinese=9          # from enum MsoCharacterSet
	msoCharacterSetThai           =10         # from enum MsoCharacterSet
	msoCharacterSetTraditionalChinese=11         # from enum MsoCharacterSet
	msoCharacterSetVietnamese     =12         # from enum MsoCharacterSet
	msoColorTypeCMS               =4          # from enum MsoColorType
	msoColorTypeCMYK              =3          # from enum MsoColorType
	msoColorTypeInk               =5          # from enum MsoColorType
	msoColorTypeMixed             =-2         # from enum MsoColorType
	msoColorTypeRGB               =1          # from enum MsoColorType
	msoColorTypeScheme            =2          # from enum MsoColorType
	msoComboLabel                 =1          # from enum MsoComboStyle
	msoComboNormal                =0          # from enum MsoComboStyle
	msoCommandBarButtonHyperlinkInsertPicture=2          # from enum MsoCommandBarButtonHyperlinkType
	msoCommandBarButtonHyperlinkNone=0          # from enum MsoCommandBarButtonHyperlinkType
	msoCommandBarButtonHyperlinkOpen=1          # from enum MsoCommandBarButtonHyperlinkType
	msoConditionAnyNumberBetween  =34         # from enum MsoCondition
	msoConditionAnytime           =25         # from enum MsoCondition
	msoConditionAnytimeBetween    =26         # from enum MsoCondition
	msoConditionAtLeast           =36         # from enum MsoCondition
	msoConditionAtMost            =35         # from enum MsoCondition
	msoConditionBeginsWith        =11         # from enum MsoCondition
	msoConditionDoesNotEqual      =33         # from enum MsoCondition
	msoConditionEndsWith          =12         # from enum MsoCondition
	msoConditionEquals            =32         # from enum MsoCondition
	msoConditionEqualsCompleted   =66         # from enum MsoCondition
	msoConditionEqualsDeferred    =68         # from enum MsoCondition
	msoConditionEqualsHigh        =60         # from enum MsoCondition
	msoConditionEqualsInProgress  =65         # from enum MsoCondition
	msoConditionEqualsLow         =58         # from enum MsoCondition
	msoConditionEqualsNormal      =59         # from enum MsoCondition
	msoConditionEqualsNotStarted  =64         # from enum MsoCondition
	msoConditionEqualsWaitingForSomeoneElse=67         # from enum MsoCondition
	msoConditionFileTypeAllFiles  =1          # from enum MsoCondition
	msoConditionFileTypeBinders   =6          # from enum MsoCondition
	msoConditionFileTypeCalendarItem=45         # from enum MsoCondition
	msoConditionFileTypeContactItem=46         # from enum MsoCondition
	msoConditionFileTypeDataConnectionFiles=51         # from enum MsoCondition
	msoConditionFileTypeDatabases =7          # from enum MsoCondition
	msoConditionFileTypeDesignerFiles=56         # from enum MsoCondition
	msoConditionFileTypeDocumentImagingFiles=54         # from enum MsoCondition
	msoConditionFileTypeExcelWorkbooks=4          # from enum MsoCondition
	msoConditionFileTypeJournalItem=48         # from enum MsoCondition
	msoConditionFileTypeMailItem  =44         # from enum MsoCondition
	msoConditionFileTypeNoteItem  =47         # from enum MsoCondition
	msoConditionFileTypeOfficeFiles=2          # from enum MsoCondition
	msoConditionFileTypeOutlookItems=43         # from enum MsoCondition
	msoConditionFileTypePhotoDrawFiles=50         # from enum MsoCondition
	msoConditionFileTypePowerPointPresentations=5          # from enum MsoCondition
	msoConditionFileTypeProjectFiles=53         # from enum MsoCondition
	msoConditionFileTypePublisherFiles=52         # from enum MsoCondition
	msoConditionFileTypeTaskItem  =49         # from enum MsoCondition
	msoConditionFileTypeTemplates =8          # from enum MsoCondition
	msoConditionFileTypeVisioFiles=55         # from enum MsoCondition
	msoConditionFileTypeWebPages  =57         # from enum MsoCondition
	msoConditionFileTypeWordDocuments=3          # from enum MsoCondition
	msoConditionFreeText          =42         # from enum MsoCondition
	msoConditionInTheLast         =31         # from enum MsoCondition
	msoConditionInTheNext         =30         # from enum MsoCondition
	msoConditionIncludes          =9          # from enum MsoCondition
	msoConditionIncludesFormsOf   =41         # from enum MsoCondition
	msoConditionIncludesNearEachOther=13         # from enum MsoCondition
	msoConditionIncludesPhrase    =10         # from enum MsoCondition
	msoConditionIsExactly         =14         # from enum MsoCondition
	msoConditionIsNo              =40         # from enum MsoCondition
	msoConditionIsNot             =15         # from enum MsoCondition
	msoConditionIsYes             =39         # from enum MsoCondition
	msoConditionLastMonth         =22         # from enum MsoCondition
	msoConditionLastWeek          =19         # from enum MsoCondition
	msoConditionLessThan          =38         # from enum MsoCondition
	msoConditionMoreThan          =37         # from enum MsoCondition
	msoConditionNextMonth         =24         # from enum MsoCondition
	msoConditionNextWeek          =21         # from enum MsoCondition
	msoConditionNotEqualToCompleted=71         # from enum MsoCondition
	msoConditionNotEqualToDeferred=73         # from enum MsoCondition
	msoConditionNotEqualToHigh    =63         # from enum MsoCondition
	msoConditionNotEqualToInProgress=70         # from enum MsoCondition
	msoConditionNotEqualToLow     =61         # from enum MsoCondition
	msoConditionNotEqualToNormal  =62         # from enum MsoCondition
	msoConditionNotEqualToNotStarted=69         # from enum MsoCondition
	msoConditionNotEqualToWaitingForSomeoneElse=72         # from enum MsoCondition
	msoConditionOn                =27         # from enum MsoCondition
	msoConditionOnOrAfter         =28         # from enum MsoCondition
	msoConditionOnOrBefore        =29         # from enum MsoCondition
	msoConditionThisMonth         =23         # from enum MsoCondition
	msoConditionThisWeek          =20         # from enum MsoCondition
	msoConditionToday             =17         # from enum MsoCondition
	msoConditionTomorrow          =18         # from enum MsoCondition
	msoConditionYesterday         =16         # from enum MsoCondition
	msoConnectorAnd               =1          # from enum MsoConnector
	msoConnectorOr                =2          # from enum MsoConnector
	msoConnectorCurve             =3          # from enum MsoConnectorType
	msoConnectorElbow             =2          # from enum MsoConnectorType
	msoConnectorStraight          =1          # from enum MsoConnectorType
	msoConnectorTypeMixed         =-2         # from enum MsoConnectorType
	msoControlOLEUsageBoth        =3          # from enum MsoControlOLEUsage
	msoControlOLEUsageClient      =2          # from enum MsoControlOLEUsage
	msoControlOLEUsageNeither     =0          # from enum MsoControlOLEUsage
	msoControlOLEUsageServer      =1          # from enum MsoControlOLEUsage
	msoControlActiveX             =22         # from enum MsoControlType
	msoControlAutoCompleteCombo   =26         # from enum MsoControlType
	msoControlButton              =1          # from enum MsoControlType
	msoControlButtonDropdown      =5          # from enum MsoControlType
	msoControlButtonPopup         =12         # from enum MsoControlType
	msoControlComboBox            =4          # from enum MsoControlType
	msoControlCustom              =0          # from enum MsoControlType
	msoControlDropdown            =3          # from enum MsoControlType
	msoControlEdit                =2          # from enum MsoControlType
	msoControlExpandingGrid       =16         # from enum MsoControlType
	msoControlGauge               =19         # from enum MsoControlType
	msoControlGenericDropdown     =8          # from enum MsoControlType
	msoControlGraphicCombo        =20         # from enum MsoControlType
	msoControlGraphicDropdown     =9          # from enum MsoControlType
	msoControlGraphicPopup        =11         # from enum MsoControlType
	msoControlGrid                =18         # from enum MsoControlType
	msoControlLabel               =15         # from enum MsoControlType
	msoControlLabelEx             =24         # from enum MsoControlType
	msoControlOCXDropdown         =7          # from enum MsoControlType
	msoControlPane                =21         # from enum MsoControlType
	msoControlPopup               =10         # from enum MsoControlType
	msoControlSpinner             =23         # from enum MsoControlType
	msoControlSplitButtonMRUPopup =14         # from enum MsoControlType
	msoControlSplitButtonPopup    =13         # from enum MsoControlType
	msoControlSplitDropdown       =6          # from enum MsoControlType
	msoControlSplitExpandingGrid  =17         # from enum MsoControlType
	msoControlWorkPane            =25         # from enum MsoControlType
	msoDiagramAssistant           =2          # from enum MsoDiagramNodeType
	msoDiagramNode                =1          # from enum MsoDiagramNodeType
	msoDiagramCycle               =2          # from enum MsoDiagramType
	msoDiagramMixed               =-2         # from enum MsoDiagramType
	msoDiagramOrgChart            =1          # from enum MsoDiagramType
	msoDiagramPyramid             =4          # from enum MsoDiagramType
	msoDiagramRadial              =3          # from enum MsoDiagramType
	msoDiagramTarget              =6          # from enum MsoDiagramType
	msoDiagramVenn                =5          # from enum MsoDiagramType
	msoDistributeHorizontally     =0          # from enum MsoDistributeCmd
	msoDistributeVertically       =1          # from enum MsoDistributeCmd
	msoPropertyTypeBoolean        =2          # from enum MsoDocProperties
	msoPropertyTypeDate           =3          # from enum MsoDocProperties
	msoPropertyTypeFloat          =5          # from enum MsoDocProperties
	msoPropertyTypeNumber         =1          # from enum MsoDocProperties
	msoPropertyTypeString         =4          # from enum MsoDocProperties
	msoEditingAuto                =0          # from enum MsoEditingType
	msoEditingCorner              =1          # from enum MsoEditingType
	msoEditingSmooth              =2          # from enum MsoEditingType
	msoEditingSymmetric           =3          # from enum MsoEditingType
	msoEncodingArabic             =1256       # from enum MsoEncoding
	msoEncodingArabicASMO         =708        # from enum MsoEncoding
	msoEncodingArabicAutoDetect   =51256      # from enum MsoEncoding
	msoEncodingArabicTransparentASMO=720        # from enum MsoEncoding
	msoEncodingAutoDetect         =50001      # from enum MsoEncoding
	msoEncodingBaltic             =1257       # from enum MsoEncoding
	msoEncodingCentralEuropean    =1250       # from enum MsoEncoding
	msoEncodingCyrillic           =1251       # from enum MsoEncoding
	msoEncodingCyrillicAutoDetect =51251      # from enum MsoEncoding
	msoEncodingEBCDICArabic       =20420      # from enum MsoEncoding
	msoEncodingEBCDICDenmarkNorway=20277      # from enum MsoEncoding
	msoEncodingEBCDICFinlandSweden=20278      # from enum MsoEncoding
	msoEncodingEBCDICFrance       =20297      # from enum MsoEncoding
	msoEncodingEBCDICGermany      =20273      # from enum MsoEncoding
	msoEncodingEBCDICGreek        =20423      # from enum MsoEncoding
	msoEncodingEBCDICGreekModern  =875        # from enum MsoEncoding
	msoEncodingEBCDICHebrew       =20424      # from enum MsoEncoding
	msoEncodingEBCDICIcelandic    =20871      # from enum MsoEncoding
	msoEncodingEBCDICInternational=500        # from enum MsoEncoding
	msoEncodingEBCDICItaly        =20280      # from enum MsoEncoding
	msoEncodingEBCDICJapaneseKatakanaExtended=20290      # from enum MsoEncoding
	msoEncodingEBCDICJapaneseKatakanaExtendedAndJapanese=50930      # from enum MsoEncoding
	msoEncodingEBCDICJapaneseLatinExtendedAndJapanese=50939      # from enum MsoEncoding
	msoEncodingEBCDICKoreanExtended=20833      # from enum MsoEncoding
	msoEncodingEBCDICKoreanExtendedAndKorean=50933      # from enum MsoEncoding
	msoEncodingEBCDICLatinAmericaSpain=20284      # from enum MsoEncoding
	msoEncodingEBCDICMultilingualROECELatin2=870        # from enum MsoEncoding
	msoEncodingEBCDICRussian      =20880      # from enum MsoEncoding
	msoEncodingEBCDICSerbianBulgarian=21025      # from enum MsoEncoding
	msoEncodingEBCDICSimplifiedChineseExtendedAndSimplifiedChinese=50935      # from enum MsoEncoding
	msoEncodingEBCDICThai         =20838      # from enum MsoEncoding
	msoEncodingEBCDICTurkish      =20905      # from enum MsoEncoding
	msoEncodingEBCDICTurkishLatin5=1026       # from enum MsoEncoding
	msoEncodingEBCDICUSCanada     =37         # from enum MsoEncoding
	msoEncodingEBCDICUSCanadaAndJapanese=50931      # from enum MsoEncoding
	msoEncodingEBCDICUSCanadaAndTraditionalChinese=50937      # from enum MsoEncoding
	msoEncodingEBCDICUnitedKingdom=20285      # from enum MsoEncoding
	msoEncodingEUCChineseSimplifiedChinese=51936      # from enum MsoEncoding
	msoEncodingEUCJapanese        =51932      # from enum MsoEncoding
	msoEncodingEUCKorean          =51949      # from enum MsoEncoding
	msoEncodingEUCTaiwaneseTraditionalChinese=51950      # from enum MsoEncoding
	msoEncodingEuropa3            =29001      # from enum MsoEncoding
	msoEncodingExtAlphaLowercase  =21027      # from enum MsoEncoding
	msoEncodingGreek              =1253       # from enum MsoEncoding
	msoEncodingGreekAutoDetect    =51253      # from enum MsoEncoding
	msoEncodingHZGBSimplifiedChinese=52936      # from enum MsoEncoding
	msoEncodingHebrew             =1255       # from enum MsoEncoding
	msoEncodingIA5German          =20106      # from enum MsoEncoding
	msoEncodingIA5IRV             =20105      # from enum MsoEncoding
	msoEncodingIA5Norwegian       =20108      # from enum MsoEncoding
	msoEncodingIA5Swedish         =20107      # from enum MsoEncoding
	msoEncodingISCIIAssamese      =57006      # from enum MsoEncoding
	msoEncodingISCIIBengali       =57003      # from enum MsoEncoding
	msoEncodingISCIIDevanagari    =57002      # from enum MsoEncoding
	msoEncodingISCIIGujarati      =57010      # from enum MsoEncoding
	msoEncodingISCIIKannada       =57008      # from enum MsoEncoding
	msoEncodingISCIIMalayalam     =57009      # from enum MsoEncoding
	msoEncodingISCIIOriya         =57007      # from enum MsoEncoding
	msoEncodingISCIIPunjabi       =57011      # from enum MsoEncoding
	msoEncodingISCIITamil         =57004      # from enum MsoEncoding
	msoEncodingISCIITelugu        =57005      # from enum MsoEncoding
	msoEncodingISO2022CNSimplifiedChinese=50229      # from enum MsoEncoding
	msoEncodingISO2022CNTraditionalChinese=50227      # from enum MsoEncoding
	msoEncodingISO2022JPJISX02011989=50222      # from enum MsoEncoding
	msoEncodingISO2022JPJISX02021984=50221      # from enum MsoEncoding
	msoEncodingISO2022JPNoHalfwidthKatakana=50220      # from enum MsoEncoding
	msoEncodingISO2022KR          =50225      # from enum MsoEncoding
	msoEncodingISO6937NonSpacingAccent=20269      # from enum MsoEncoding
	msoEncodingISO885915Latin9    =28605      # from enum MsoEncoding
	msoEncodingISO88591Latin1     =28591      # from enum MsoEncoding
	msoEncodingISO88592CentralEurope=28592      # from enum MsoEncoding
	msoEncodingISO88593Latin3     =28593      # from enum MsoEncoding
	msoEncodingISO88594Baltic     =28594      # from enum MsoEncoding
	msoEncodingISO88595Cyrillic   =28595      # from enum MsoEncoding
	msoEncodingISO88596Arabic     =28596      # from enum MsoEncoding
	msoEncodingISO88597Greek      =28597      # from enum MsoEncoding
	msoEncodingISO88598Hebrew     =28598      # from enum MsoEncoding
	msoEncodingISO88598HebrewLogical=38598      # from enum MsoEncoding
	msoEncodingISO88599Turkish    =28599      # from enum MsoEncoding
	msoEncodingJapaneseAutoDetect =50932      # from enum MsoEncoding
	msoEncodingJapaneseShiftJIS   =932        # from enum MsoEncoding
	msoEncodingKOI8R              =20866      # from enum MsoEncoding
	msoEncodingKOI8U              =21866      # from enum MsoEncoding
	msoEncodingKorean             =949        # from enum MsoEncoding
	msoEncodingKoreanAutoDetect   =50949      # from enum MsoEncoding
	msoEncodingKoreanJohab        =1361       # from enum MsoEncoding
	msoEncodingMacArabic          =10004      # from enum MsoEncoding
	msoEncodingMacCroatia         =10082      # from enum MsoEncoding
	msoEncodingMacCyrillic        =10007      # from enum MsoEncoding
	msoEncodingMacGreek1          =10006      # from enum MsoEncoding
	msoEncodingMacHebrew          =10005      # from enum MsoEncoding
	msoEncodingMacIcelandic       =10079      # from enum MsoEncoding
	msoEncodingMacJapanese        =10001      # from enum MsoEncoding
	msoEncodingMacKorean          =10003      # from enum MsoEncoding
	msoEncodingMacLatin2          =10029      # from enum MsoEncoding
	msoEncodingMacRoman           =10000      # from enum MsoEncoding
	msoEncodingMacRomania         =10010      # from enum MsoEncoding
	msoEncodingMacSimplifiedChineseGB2312=10008      # from enum MsoEncoding
	msoEncodingMacTraditionalChineseBig5=10002      # from enum MsoEncoding
	msoEncodingMacTurkish         =10081      # from enum MsoEncoding
	msoEncodingMacUkraine         =10017      # from enum MsoEncoding
	msoEncodingOEMArabic          =864        # from enum MsoEncoding
	msoEncodingOEMBaltic          =775        # from enum MsoEncoding
	msoEncodingOEMCanadianFrench  =863        # from enum MsoEncoding
	msoEncodingOEMCyrillic        =855        # from enum MsoEncoding
	msoEncodingOEMCyrillicII      =866        # from enum MsoEncoding
	msoEncodingOEMGreek437G       =737        # from enum MsoEncoding
	msoEncodingOEMHebrew          =862        # from enum MsoEncoding
	msoEncodingOEMIcelandic       =861        # from enum MsoEncoding
	msoEncodingOEMModernGreek     =869        # from enum MsoEncoding
	msoEncodingOEMMultilingualLatinI=850        # from enum MsoEncoding
	msoEncodingOEMMultilingualLatinII=852        # from enum MsoEncoding
	msoEncodingOEMNordic          =865        # from enum MsoEncoding
	msoEncodingOEMPortuguese      =860        # from enum MsoEncoding
	msoEncodingOEMTurkish         =857        # from enum MsoEncoding
	msoEncodingOEMUnitedStates    =437        # from enum MsoEncoding
	msoEncodingSimplifiedChineseAutoDetect=50936      # from enum MsoEncoding
	msoEncodingSimplifiedChineseGB18030=54936      # from enum MsoEncoding
	msoEncodingSimplifiedChineseGBK=936        # from enum MsoEncoding
	msoEncodingT61                =20261      # from enum MsoEncoding
	msoEncodingTaiwanCNS          =20000      # from enum MsoEncoding
	msoEncodingTaiwanEten         =20002      # from enum MsoEncoding
	msoEncodingTaiwanIBM5550      =20003      # from enum MsoEncoding
	msoEncodingTaiwanTCA          =20001      # from enum MsoEncoding
	msoEncodingTaiwanTeleText     =20004      # from enum MsoEncoding
	msoEncodingTaiwanWang         =20005      # from enum MsoEncoding
	msoEncodingThai               =874        # from enum MsoEncoding
	msoEncodingTraditionalChineseAutoDetect=50950      # from enum MsoEncoding
	msoEncodingTraditionalChineseBig5=950        # from enum MsoEncoding
	msoEncodingTurkish            =1254       # from enum MsoEncoding
	msoEncodingUSASCII            =20127      # from enum MsoEncoding
	msoEncodingUTF7               =65000      # from enum MsoEncoding
	msoEncodingUTF8               =65001      # from enum MsoEncoding
	msoEncodingUnicodeBigEndian   =1201       # from enum MsoEncoding
	msoEncodingUnicodeLittleEndian=1200       # from enum MsoEncoding
	msoEncodingVietnamese         =1258       # from enum MsoEncoding
	msoEncodingWestern            =1252       # from enum MsoEncoding
	msoMethodGet                  =0          # from enum MsoExtraInfoMethod
	msoMethodPost                 =1          # from enum MsoExtraInfoMethod
	msoExtrusionColorAutomatic    =1          # from enum MsoExtrusionColorType
	msoExtrusionColorCustom       =2          # from enum MsoExtrusionColorType
	msoExtrusionColorTypeMixed    =-2         # from enum MsoExtrusionColorType
	MsoFarEastLineBreakLanguageJapanese=1041       # from enum MsoFarEastLineBreakLanguageID
	MsoFarEastLineBreakLanguageKorean=1042       # from enum MsoFarEastLineBreakLanguageID
	MsoFarEastLineBreakLanguageSimplifiedChinese=2052       # from enum MsoFarEastLineBreakLanguageID
	MsoFarEastLineBreakLanguageTraditionalChinese=1028       # from enum MsoFarEastLineBreakLanguageID
	msoFeatureInstallNone         =0          # from enum MsoFeatureInstall
	msoFeatureInstallOnDemand     =1          # from enum MsoFeatureInstall
	msoFeatureInstallOnDemandWithUI=2          # from enum MsoFeatureInstall
	msoFileDialogFilePicker       =3          # from enum MsoFileDialogType
	msoFileDialogFolderPicker     =4          # from enum MsoFileDialogType
	msoFileDialogOpen             =1          # from enum MsoFileDialogType
	msoFileDialogSaveAs           =2          # from enum MsoFileDialogType
	msoFileDialogViewDetails      =2          # from enum MsoFileDialogView
	msoFileDialogViewLargeIcons   =6          # from enum MsoFileDialogView
	msoFileDialogViewList         =1          # from enum MsoFileDialogView
	msoFileDialogViewPreview      =4          # from enum MsoFileDialogView
	msoFileDialogViewProperties   =3          # from enum MsoFileDialogView
	msoFileDialogViewSmallIcons   =7          # from enum MsoFileDialogView
	msoFileDialogViewThumbnail    =5          # from enum MsoFileDialogView
	msoFileDialogViewTiles        =9          # from enum MsoFileDialogView
	msoFileDialogViewWebView      =8          # from enum MsoFileDialogView
	msoListbyName                 =1          # from enum MsoFileFindListBy
	msoListbyTitle                =2          # from enum MsoFileFindListBy
	msoOptionsAdd                 =2          # from enum MsoFileFindOptions
	msoOptionsNew                 =1          # from enum MsoFileFindOptions
	msoOptionsWithin              =3          # from enum MsoFileFindOptions
	msoFileFindSortbyAuthor       =1          # from enum MsoFileFindSortBy
	msoFileFindSortbyDateCreated  =2          # from enum MsoFileFindSortBy
	msoFileFindSortbyDateSaved    =4          # from enum MsoFileFindSortBy
	msoFileFindSortbyFileName     =5          # from enum MsoFileFindSortBy
	msoFileFindSortbyLastSavedBy  =3          # from enum MsoFileFindSortBy
	msoFileFindSortbySize         =6          # from enum MsoFileFindSortBy
	msoFileFindSortbyTitle        =7          # from enum MsoFileFindSortBy
	msoViewFileInfo               =1          # from enum MsoFileFindView
	msoViewPreview                =2          # from enum MsoFileFindView
	msoViewSummaryInfo            =3          # from enum MsoFileFindView
	msoCreateNewFile              =1          # from enum MsoFileNewAction
	msoEditFile                   =0          # from enum MsoFileNewAction
	msoOpenFile                   =2          # from enum MsoFileNewAction
	msoBottomSection              =4          # from enum MsoFileNewSection
	msoNew                        =1          # from enum MsoFileNewSection
	msoNewfromExistingFile        =2          # from enum MsoFileNewSection
	msoNewfromTemplate            =3          # from enum MsoFileNewSection
	msoOpenDocument               =0          # from enum MsoFileNewSection
	msoFileTypeAllFiles           =1          # from enum MsoFileType
	msoFileTypeBinders            =6          # from enum MsoFileType
	msoFileTypeCalendarItem       =11         # from enum MsoFileType
	msoFileTypeContactItem        =12         # from enum MsoFileType
	msoFileTypeDataConnectionFiles=17         # from enum MsoFileType
	msoFileTypeDatabases          =7          # from enum MsoFileType
	msoFileTypeDesignerFiles      =22         # from enum MsoFileType
	msoFileTypeDocumentImagingFiles=20         # from enum MsoFileType
	msoFileTypeExcelWorkbooks     =4          # from enum MsoFileType
	msoFileTypeJournalItem        =14         # from enum MsoFileType
	msoFileTypeMailItem           =10         # from enum MsoFileType
	msoFileTypeNoteItem           =13         # from enum MsoFileType
	msoFileTypeOfficeFiles        =2          # from enum MsoFileType
	msoFileTypeOutlookItems       =9          # from enum MsoFileType
	msoFileTypePhotoDrawFiles     =16         # from enum MsoFileType
	msoFileTypePowerPointPresentations=5          # from enum MsoFileType
	msoFileTypeProjectFiles       =19         # from enum MsoFileType
	msoFileTypePublisherFiles     =18         # from enum MsoFileType
	msoFileTypeTaskItem           =15         # from enum MsoFileType
	msoFileTypeTemplates          =8          # from enum MsoFileType
	msoFileTypeVisioFiles         =21         # from enum MsoFileType
	msoFileTypeWebPages           =23         # from enum MsoFileType
	msoFileTypeWordDocuments      =3          # from enum MsoFileType
	msoFillBackground             =5          # from enum MsoFillType
	msoFillGradient               =3          # from enum MsoFillType
	msoFillMixed                  =-2         # from enum MsoFillType
	msoFillPatterned              =2          # from enum MsoFillType
	msoFillPicture                =6          # from enum MsoFillType
	msoFillSolid                  =1          # from enum MsoFillType
	msoFillTextured               =4          # from enum MsoFillType
	msoFilterComparisonContains   =8          # from enum MsoFilterComparison
	msoFilterComparisonEqual      =0          # from enum MsoFilterComparison
	msoFilterComparisonGreaterThan=3          # from enum MsoFilterComparison
	msoFilterComparisonGreaterThanEqual=5          # from enum MsoFilterComparison
	msoFilterComparisonIsBlank    =6          # from enum MsoFilterComparison
	msoFilterComparisonIsNotBlank =7          # from enum MsoFilterComparison
	msoFilterComparisonLessThan   =2          # from enum MsoFilterComparison
	msoFilterComparisonLessThanEqual=4          # from enum MsoFilterComparison
	msoFilterComparisonNotContains=9          # from enum MsoFilterComparison
	msoFilterComparisonNotEqual   =1          # from enum MsoFilterComparison
	msoFilterConjunctionAnd       =0          # from enum MsoFilterConjunction
	msoFilterConjunctionOr        =1          # from enum MsoFilterConjunction
	msoFlipHorizontal             =0          # from enum MsoFlipCmd
	msoFlipVertical               =1          # from enum MsoFlipCmd
	msoGradientColorMixed         =-2         # from enum MsoGradientColorType
	msoGradientOneColor           =1          # from enum MsoGradientColorType
	msoGradientPresetColors       =3          # from enum MsoGradientColorType
	msoGradientTwoColors          =2          # from enum MsoGradientColorType
	msoGradientDiagonalDown       =4          # from enum MsoGradientStyle
	msoGradientDiagonalUp         =3          # from enum MsoGradientStyle
	msoGradientFromCenter         =7          # from enum MsoGradientStyle
	msoGradientFromCorner         =5          # from enum MsoGradientStyle
	msoGradientFromTitle          =6          # from enum MsoGradientStyle
	msoGradientHorizontal         =1          # from enum MsoGradientStyle
	msoGradientMixed              =-2         # from enum MsoGradientStyle
	msoGradientVertical           =2          # from enum MsoGradientStyle
	msoHTMLProjectOpenSourceView  =1          # from enum MsoHTMLProjectOpen
	msoHTMLProjectOpenTextView    =2          # from enum MsoHTMLProjectOpen
	msoHTMLProjectStateDocumentLocked=1          # from enum MsoHTMLProjectState
	msoHTMLProjectStateDocumentProjectUnlocked=3          # from enum MsoHTMLProjectState
	msoHTMLProjectStateProjectLocked=2          # from enum MsoHTMLProjectState
	msoAnchorCenter               =2          # from enum MsoHorizontalAnchor
	msoAnchorNone                 =1          # from enum MsoHorizontalAnchor
	msoHorizontalAnchorMixed      =-2         # from enum MsoHorizontalAnchor
	msoHyperlinkInlineShape       =2          # from enum MsoHyperlinkType
	msoHyperlinkRange             =0          # from enum MsoHyperlinkType
	msoHyperlinkShape             =1          # from enum MsoHyperlinkType
	msoIconAlert                  =2          # from enum MsoIconType
	msoIconAlertCritical          =7          # from enum MsoIconType
	msoIconAlertInfo              =4          # from enum MsoIconType
	msoIconAlertQuery             =6          # from enum MsoIconType
	msoIconAlertWarning           =5          # from enum MsoIconType
	msoIconNone                   =0          # from enum MsoIconType
	msoIconTip                    =3          # from enum MsoIconType
	msoLanguageIDAfrikaans        =1078       # from enum MsoLanguageID
	msoLanguageIDAlbanian         =1052       # from enum MsoLanguageID
	msoLanguageIDAmharic          =1118       # from enum MsoLanguageID
	msoLanguageIDArabic           =1025       # from enum MsoLanguageID
	msoLanguageIDArabicAlgeria    =5121       # from enum MsoLanguageID
	msoLanguageIDArabicBahrain    =15361      # from enum MsoLanguageID
	msoLanguageIDArabicEgypt      =3073       # from enum MsoLanguageID
	msoLanguageIDArabicIraq       =2049       # from enum MsoLanguageID
	msoLanguageIDArabicJordan     =11265      # from enum MsoLanguageID
	msoLanguageIDArabicKuwait     =13313      # from enum MsoLanguageID
	msoLanguageIDArabicLebanon    =12289      # from enum MsoLanguageID
	msoLanguageIDArabicLibya      =4097       # from enum MsoLanguageID
	msoLanguageIDArabicMorocco    =6145       # from enum MsoLanguageID
	msoLanguageIDArabicOman       =8193       # from enum MsoLanguageID
	msoLanguageIDArabicQatar      =16385      # from enum MsoLanguageID
	msoLanguageIDArabicSyria      =10241      # from enum MsoLanguageID
	msoLanguageIDArabicTunisia    =7169       # from enum MsoLanguageID
	msoLanguageIDArabicUAE        =14337      # from enum MsoLanguageID
	msoLanguageIDArabicYemen      =9217       # from enum MsoLanguageID
	msoLanguageIDArmenian         =1067       # from enum MsoLanguageID
	msoLanguageIDAssamese         =1101       # from enum MsoLanguageID
	msoLanguageIDAzeriCyrillic    =2092       # from enum MsoLanguageID
	msoLanguageIDAzeriLatin       =1068       # from enum MsoLanguageID
	msoLanguageIDBasque           =1069       # from enum MsoLanguageID
	msoLanguageIDBelgianDutch     =2067       # from enum MsoLanguageID
	msoLanguageIDBelgianFrench    =2060       # from enum MsoLanguageID
	msoLanguageIDBengali          =1093       # from enum MsoLanguageID
	msoLanguageIDBosnian          =4122       # from enum MsoLanguageID
	msoLanguageIDBosnianBosniaHerzegovinaCyrillic=8218       # from enum MsoLanguageID
	msoLanguageIDBosnianBosniaHerzegovinaLatin=5146       # from enum MsoLanguageID
	msoLanguageIDBrazilianPortuguese=1046       # from enum MsoLanguageID
	msoLanguageIDBulgarian        =1026       # from enum MsoLanguageID
	msoLanguageIDBurmese          =1109       # from enum MsoLanguageID
	msoLanguageIDByelorussian     =1059       # from enum MsoLanguageID
	msoLanguageIDCatalan          =1027       # from enum MsoLanguageID
	msoLanguageIDCherokee         =1116       # from enum MsoLanguageID
	msoLanguageIDChineseHongKongSAR=3076       # from enum MsoLanguageID
	msoLanguageIDChineseMacaoSAR  =5124       # from enum MsoLanguageID
	msoLanguageIDChineseSingapore =4100       # from enum MsoLanguageID
	msoLanguageIDCroatian         =1050       # from enum MsoLanguageID
	msoLanguageIDCzech            =1029       # from enum MsoLanguageID
	msoLanguageIDDanish           =1030       # from enum MsoLanguageID
	msoLanguageIDDivehi           =1125       # from enum MsoLanguageID
	msoLanguageIDDutch            =1043       # from enum MsoLanguageID
	msoLanguageIDDzongkhaBhutan   =2129       # from enum MsoLanguageID
	msoLanguageIDEdo              =1126       # from enum MsoLanguageID
	msoLanguageIDEnglishAUS       =3081       # from enum MsoLanguageID
	msoLanguageIDEnglishBelize    =10249      # from enum MsoLanguageID
	msoLanguageIDEnglishCanadian  =4105       # from enum MsoLanguageID
	msoLanguageIDEnglishCaribbean =9225       # from enum MsoLanguageID
	msoLanguageIDEnglishIndonesia =14345      # from enum MsoLanguageID
	msoLanguageIDEnglishIreland   =6153       # from enum MsoLanguageID
	msoLanguageIDEnglishJamaica   =8201       # from enum MsoLanguageID
	msoLanguageIDEnglishNewZealand=5129       # from enum MsoLanguageID
	msoLanguageIDEnglishPhilippines=13321      # from enum MsoLanguageID
	msoLanguageIDEnglishSouthAfrica=7177       # from enum MsoLanguageID
	msoLanguageIDEnglishTrinidadTobago=11273      # from enum MsoLanguageID
	msoLanguageIDEnglishUK        =2057       # from enum MsoLanguageID
	msoLanguageIDEnglishUS        =1033       # from enum MsoLanguageID
	msoLanguageIDEnglishZimbabwe  =12297      # from enum MsoLanguageID
	msoLanguageIDEstonian         =1061       # from enum MsoLanguageID
	msoLanguageIDFaeroese         =1080       # from enum MsoLanguageID
	msoLanguageIDFarsi            =1065       # from enum MsoLanguageID
	msoLanguageIDFilipino         =1124       # from enum MsoLanguageID
	msoLanguageIDFinnish          =1035       # from enum MsoLanguageID
	msoLanguageIDFrench           =1036       # from enum MsoLanguageID
	msoLanguageIDFrenchCameroon   =11276      # from enum MsoLanguageID
	msoLanguageIDFrenchCanadian   =3084       # from enum MsoLanguageID
	msoLanguageIDFrenchCotedIvoire=12300      # from enum MsoLanguageID
	msoLanguageIDFrenchHaiti      =15372      # from enum MsoLanguageID
	msoLanguageIDFrenchLuxembourg =5132       # from enum MsoLanguageID
	msoLanguageIDFrenchMali       =13324      # from enum MsoLanguageID
	msoLanguageIDFrenchMonaco     =6156       # from enum MsoLanguageID
	msoLanguageIDFrenchMorocco    =14348      # from enum MsoLanguageID
	msoLanguageIDFrenchReunion    =8204       # from enum MsoLanguageID
	msoLanguageIDFrenchSenegal    =10252      # from enum MsoLanguageID
	msoLanguageIDFrenchWestIndies =7180       # from enum MsoLanguageID
	msoLanguageIDFrenchZaire      =9228       # from enum MsoLanguageID
	msoLanguageIDFrisianNetherlands=1122       # from enum MsoLanguageID
	msoLanguageIDFulfulde         =1127       # from enum MsoLanguageID
	msoLanguageIDGaelicIreland    =2108       # from enum MsoLanguageID
	msoLanguageIDGaelicScotland   =1084       # from enum MsoLanguageID
	msoLanguageIDGalician         =1110       # from enum MsoLanguageID
	msoLanguageIDGeorgian         =1079       # from enum MsoLanguageID
	msoLanguageIDGerman           =1031       # from enum MsoLanguageID
	msoLanguageIDGermanAustria    =3079       # from enum MsoLanguageID
	msoLanguageIDGermanLiechtenstein=5127       # from enum MsoLanguageID
	msoLanguageIDGermanLuxembourg =4103       # from enum MsoLanguageID
	msoLanguageIDGreek            =1032       # from enum MsoLanguageID
	msoLanguageIDGuarani          =1140       # from enum MsoLanguageID
	msoLanguageIDGujarati         =1095       # from enum MsoLanguageID
	msoLanguageIDHausa            =1128       # from enum MsoLanguageID
	msoLanguageIDHawaiian         =1141       # from enum MsoLanguageID
	msoLanguageIDHebrew           =1037       # from enum MsoLanguageID
	msoLanguageIDHindi            =1081       # from enum MsoLanguageID
	msoLanguageIDHungarian        =1038       # from enum MsoLanguageID
	msoLanguageIDIbibio           =1129       # from enum MsoLanguageID
	msoLanguageIDIcelandic        =1039       # from enum MsoLanguageID
	msoLanguageIDIgbo             =1136       # from enum MsoLanguageID
	msoLanguageIDIndonesian       =1057       # from enum MsoLanguageID
	msoLanguageIDInuktitut        =1117       # from enum MsoLanguageID
	msoLanguageIDItalian          =1040       # from enum MsoLanguageID
	msoLanguageIDJapanese         =1041       # from enum MsoLanguageID
	msoLanguageIDKannada          =1099       # from enum MsoLanguageID
	msoLanguageIDKanuri           =1137       # from enum MsoLanguageID
	msoLanguageIDKashmiri         =1120       # from enum MsoLanguageID
	msoLanguageIDKashmiriDevanagari=2144       # from enum MsoLanguageID
	msoLanguageIDKazakh           =1087       # from enum MsoLanguageID
	msoLanguageIDKhmer            =1107       # from enum MsoLanguageID
	msoLanguageIDKirghiz          =1088       # from enum MsoLanguageID
	msoLanguageIDKonkani          =1111       # from enum MsoLanguageID
	msoLanguageIDKorean           =1042       # from enum MsoLanguageID
	msoLanguageIDKyrgyz           =1088       # from enum MsoLanguageID
	msoLanguageIDLao              =1108       # from enum MsoLanguageID
	msoLanguageIDLatin            =1142       # from enum MsoLanguageID
	msoLanguageIDLatvian          =1062       # from enum MsoLanguageID
	msoLanguageIDLithuanian       =1063       # from enum MsoLanguageID
	msoLanguageIDMacedonian       =1071       # from enum MsoLanguageID
	msoLanguageIDMalayBruneiDarussalam=2110       # from enum MsoLanguageID
	msoLanguageIDMalayalam        =1100       # from enum MsoLanguageID
	msoLanguageIDMalaysian        =1086       # from enum MsoLanguageID
	msoLanguageIDMaltese          =1082       # from enum MsoLanguageID
	msoLanguageIDManipuri         =1112       # from enum MsoLanguageID
	msoLanguageIDMaori            =1153       # from enum MsoLanguageID
	msoLanguageIDMarathi          =1102       # from enum MsoLanguageID
	msoLanguageIDMexicanSpanish   =2058       # from enum MsoLanguageID
	msoLanguageIDMixed            =-2         # from enum MsoLanguageID
	msoLanguageIDMongolian        =1104       # from enum MsoLanguageID
	msoLanguageIDNepali           =1121       # from enum MsoLanguageID
	msoLanguageIDNoProofing       =1024       # from enum MsoLanguageID
	msoLanguageIDNone             =0          # from enum MsoLanguageID
	msoLanguageIDNorwegianBokmol  =1044       # from enum MsoLanguageID
	msoLanguageIDNorwegianNynorsk =2068       # from enum MsoLanguageID
	msoLanguageIDOriya            =1096       # from enum MsoLanguageID
	msoLanguageIDOromo            =1138       # from enum MsoLanguageID
	msoLanguageIDPashto           =1123       # from enum MsoLanguageID
	msoLanguageIDPolish           =1045       # from enum MsoLanguageID
	msoLanguageIDPortuguese       =2070       # from enum MsoLanguageID
	msoLanguageIDPunjabi          =1094       # from enum MsoLanguageID
	msoLanguageIDQuechuaBolivia   =1131       # from enum MsoLanguageID
	msoLanguageIDQuechuaEcuador   =2155       # from enum MsoLanguageID
	msoLanguageIDQuechuaPeru      =3179       # from enum MsoLanguageID
	msoLanguageIDRhaetoRomanic    =1047       # from enum MsoLanguageID
	msoLanguageIDRomanian         =1048       # from enum MsoLanguageID
	msoLanguageIDRomanianMoldova  =2072       # from enum MsoLanguageID
	msoLanguageIDRussian          =1049       # from enum MsoLanguageID
	msoLanguageIDRussianMoldova   =2073       # from enum MsoLanguageID
	msoLanguageIDSamiLappish      =1083       # from enum MsoLanguageID
	msoLanguageIDSanskrit         =1103       # from enum MsoLanguageID
	msoLanguageIDSepedi           =1132       # from enum MsoLanguageID
	msoLanguageIDSerbianBosniaHerzegovinaCyrillic=7194       # from enum MsoLanguageID
	msoLanguageIDSerbianBosniaHerzegovinaLatin=6170       # from enum MsoLanguageID
	msoLanguageIDSerbianCyrillic  =3098       # from enum MsoLanguageID
	msoLanguageIDSerbianLatin     =2074       # from enum MsoLanguageID
	msoLanguageIDSesotho          =1072       # from enum MsoLanguageID
	msoLanguageIDSimplifiedChinese=2052       # from enum MsoLanguageID
	msoLanguageIDSindhi           =1113       # from enum MsoLanguageID
	msoLanguageIDSindhiPakistan   =2137       # from enum MsoLanguageID
	msoLanguageIDSinhalese        =1115       # from enum MsoLanguageID
	msoLanguageIDSlovak           =1051       # from enum MsoLanguageID
	msoLanguageIDSlovenian        =1060       # from enum MsoLanguageID
	msoLanguageIDSomali           =1143       # from enum MsoLanguageID
	msoLanguageIDSorbian          =1070       # from enum MsoLanguageID
	msoLanguageIDSpanish          =1034       # from enum MsoLanguageID
	msoLanguageIDSpanishArgentina =11274      # from enum MsoLanguageID
	msoLanguageIDSpanishBolivia   =16394      # from enum MsoLanguageID
	msoLanguageIDSpanishChile     =13322      # from enum MsoLanguageID
	msoLanguageIDSpanishColombia  =9226       # from enum MsoLanguageID
	msoLanguageIDSpanishCostaRica =5130       # from enum MsoLanguageID
	msoLanguageIDSpanishDominicanRepublic=7178       # from enum MsoLanguageID
	msoLanguageIDSpanishEcuador   =12298      # from enum MsoLanguageID
	msoLanguageIDSpanishElSalvador=17418      # from enum MsoLanguageID
	msoLanguageIDSpanishGuatemala =4106       # from enum MsoLanguageID
	msoLanguageIDSpanishHonduras  =18442      # from enum MsoLanguageID
	msoLanguageIDSpanishModernSort=3082       # from enum MsoLanguageID
	msoLanguageIDSpanishNicaragua =19466      # from enum MsoLanguageID
	msoLanguageIDSpanishPanama    =6154       # from enum MsoLanguageID
	msoLanguageIDSpanishParaguay  =15370      # from enum MsoLanguageID
	msoLanguageIDSpanishPeru      =10250      # from enum MsoLanguageID
	msoLanguageIDSpanishPuertoRico=20490      # from enum MsoLanguageID
	msoLanguageIDSpanishUruguay   =14346      # from enum MsoLanguageID
	msoLanguageIDSpanishVenezuela =8202       # from enum MsoLanguageID
	msoLanguageIDSutu             =1072       # from enum MsoLanguageID
	msoLanguageIDSwahili          =1089       # from enum MsoLanguageID
	msoLanguageIDSwedish          =1053       # from enum MsoLanguageID
	msoLanguageIDSwedishFinland   =2077       # from enum MsoLanguageID
	msoLanguageIDSwissFrench      =4108       # from enum MsoLanguageID
	msoLanguageIDSwissGerman      =2055       # from enum MsoLanguageID
	msoLanguageIDSwissItalian     =2064       # from enum MsoLanguageID
	msoLanguageIDSyriac           =1114       # from enum MsoLanguageID
	msoLanguageIDTajik            =1064       # from enum MsoLanguageID
	msoLanguageIDTamazight        =1119       # from enum MsoLanguageID
	msoLanguageIDTamazightLatin   =2143       # from enum MsoLanguageID
	msoLanguageIDTamil            =1097       # from enum MsoLanguageID
	msoLanguageIDTatar            =1092       # from enum MsoLanguageID
	msoLanguageIDTelugu           =1098       # from enum MsoLanguageID
	msoLanguageIDThai             =1054       # from enum MsoLanguageID
	msoLanguageIDTibetan          =1105       # from enum MsoLanguageID
	msoLanguageIDTigrignaEritrea  =2163       # from enum MsoLanguageID
	msoLanguageIDTigrignaEthiopic =1139       # from enum MsoLanguageID
	msoLanguageIDTraditionalChinese=1028       # from enum MsoLanguageID
	msoLanguageIDTsonga           =1073       # from enum MsoLanguageID
	msoLanguageIDTswana           =1074       # from enum MsoLanguageID
	msoLanguageIDTurkish          =1055       # from enum MsoLanguageID
	msoLanguageIDTurkmen          =1090       # from enum MsoLanguageID
	msoLanguageIDUkrainian        =1058       # from enum MsoLanguageID
	msoLanguageIDUrdu             =1056       # from enum MsoLanguageID
	msoLanguageIDUzbekCyrillic    =2115       # from enum MsoLanguageID
	msoLanguageIDUzbekLatin       =1091       # from enum MsoLanguageID
	msoLanguageIDVenda            =1075       # from enum MsoLanguageID
	msoLanguageIDVietnamese       =1066       # from enum MsoLanguageID
	msoLanguageIDWelsh            =1106       # from enum MsoLanguageID
	msoLanguageIDXhosa            =1076       # from enum MsoLanguageID
	msoLanguageIDYi               =1144       # from enum MsoLanguageID
	msoLanguageIDYiddish          =1085       # from enum MsoLanguageID
	msoLanguageIDYoruba           =1130       # from enum MsoLanguageID
	msoLanguageIDZulu             =1077       # from enum MsoLanguageID
	msoLanguageIDChineseHongKong  =3076       # from enum MsoLanguageIDHidden
	msoLanguageIDChineseMacao     =5124       # from enum MsoLanguageIDHidden
	msoLanguageIDEnglishTrinidad  =11273      # from enum MsoLanguageIDHidden
	msoLastModifiedAnyTime        =7          # from enum MsoLastModified
	msoLastModifiedLastMonth      =5          # from enum MsoLastModified
	msoLastModifiedLastWeek       =3          # from enum MsoLastModified
	msoLastModifiedThisMonth      =6          # from enum MsoLastModified
	msoLastModifiedThisWeek       =4          # from enum MsoLastModified
	msoLastModifiedToday          =2          # from enum MsoLastModified
	msoLastModifiedYesterday      =1          # from enum MsoLastModified
	msoLineDash                   =4          # from enum MsoLineDashStyle
	msoLineDashDot                =5          # from enum MsoLineDashStyle
	msoLineDashDotDot             =6          # from enum MsoLineDashStyle
	msoLineDashStyleMixed         =-2         # from enum MsoLineDashStyle
	msoLineLongDash               =7          # from enum MsoLineDashStyle
	msoLineLongDashDot            =8          # from enum MsoLineDashStyle
	msoLineRoundDot               =3          # from enum MsoLineDashStyle
	msoLineSolid                  =1          # from enum MsoLineDashStyle
	msoLineSquareDot              =2          # from enum MsoLineDashStyle
	msoLineSingle                 =1          # from enum MsoLineStyle
	msoLineStyleMixed             =-2         # from enum MsoLineStyle
	msoLineThickBetweenThin       =5          # from enum MsoLineStyle
	msoLineThickThin              =4          # from enum MsoLineStyle
	msoLineThinThick              =3          # from enum MsoLineStyle
	msoLineThinThin               =2          # from enum MsoLineStyle
	msoMenuAnimationNone          =0          # from enum MsoMenuAnimation
	msoMenuAnimationRandom        =1          # from enum MsoMenuAnimation
	msoMenuAnimationSlide         =3          # from enum MsoMenuAnimation
	msoMenuAnimationUnfold        =2          # from enum MsoMenuAnimation
	msoIntegerMixed               =32768      # from enum MsoMixedType
	msoSingleMixed                =-2147483648 # from enum MsoMixedType
	msoModeAutoDown               =1          # from enum MsoModeType
	msoModeModal                  =0          # from enum MsoModeType
	msoModeModeless               =2          # from enum MsoModeType
	msoMoveRowFirst               =-4         # from enum MsoMoveRow
	msoMoveRowNbr                 =-1         # from enum MsoMoveRow
	msoMoveRowNext                =-2         # from enum MsoMoveRow
	msoMoveRowPrev                =-3         # from enum MsoMoveRow
	msoOLEMenuGroupContainer      =2          # from enum MsoOLEMenuGroup
	msoOLEMenuGroupEdit           =1          # from enum MsoOLEMenuGroup
	msoOLEMenuGroupFile           =0          # from enum MsoOLEMenuGroup
	msoOLEMenuGroupHelp           =5          # from enum MsoOLEMenuGroup
	msoOLEMenuGroupNone           =-1         # from enum MsoOLEMenuGroup
	msoOLEMenuGroupObject         =3          # from enum MsoOLEMenuGroup
	msoOLEMenuGroupWindow         =4          # from enum MsoOLEMenuGroup
	msoOrgChartLayoutBothHanging  =2          # from enum MsoOrgChartLayoutType
	msoOrgChartLayoutLeftHanging  =3          # from enum MsoOrgChartLayoutType
	msoOrgChartLayoutMixed        =-2         # from enum MsoOrgChartLayoutType
	msoOrgChartLayoutRightHanging =4          # from enum MsoOrgChartLayoutType
	msoOrgChartLayoutStandard     =1          # from enum MsoOrgChartLayoutType
	msoOrgChartOrientationMixed   =-2         # from enum MsoOrgChartOrientation
	msoOrgChartOrientationVertical=1          # from enum MsoOrgChartOrientation
	msoOrientationHorizontal      =1          # from enum MsoOrientation
	msoOrientationMixed           =-2         # from enum MsoOrientation
	msoOrientationVertical        =2          # from enum MsoOrientation
	msoPattern10Percent           =2          # from enum MsoPatternType
	msoPattern20Percent           =3          # from enum MsoPatternType
	msoPattern25Percent           =4          # from enum MsoPatternType
	msoPattern30Percent           =5          # from enum MsoPatternType
	msoPattern40Percent           =6          # from enum MsoPatternType
	msoPattern50Percent           =7          # from enum MsoPatternType
	msoPattern5Percent            =1          # from enum MsoPatternType
	msoPattern60Percent           =8          # from enum MsoPatternType
	msoPattern70Percent           =9          # from enum MsoPatternType
	msoPattern75Percent           =10         # from enum MsoPatternType
	msoPattern80Percent           =11         # from enum MsoPatternType
	msoPattern90Percent           =12         # from enum MsoPatternType
	msoPatternDarkDownwardDiagonal=15         # from enum MsoPatternType
	msoPatternDarkHorizontal      =13         # from enum MsoPatternType
	msoPatternDarkUpwardDiagonal  =16         # from enum MsoPatternType
	msoPatternDarkVertical        =14         # from enum MsoPatternType
	msoPatternDashedDownwardDiagonal=28         # from enum MsoPatternType
	msoPatternDashedHorizontal    =32         # from enum MsoPatternType
	msoPatternDashedUpwardDiagonal=27         # from enum MsoPatternType
	msoPatternDashedVertical      =31         # from enum MsoPatternType
	msoPatternDiagonalBrick       =40         # from enum MsoPatternType
	msoPatternDivot               =46         # from enum MsoPatternType
	msoPatternDottedDiamond       =24         # from enum MsoPatternType
	msoPatternDottedGrid          =45         # from enum MsoPatternType
	msoPatternHorizontalBrick     =35         # from enum MsoPatternType
	msoPatternLargeCheckerBoard   =36         # from enum MsoPatternType
	msoPatternLargeConfetti       =33         # from enum MsoPatternType
	msoPatternLargeGrid           =34         # from enum MsoPatternType
	msoPatternLightDownwardDiagonal=21         # from enum MsoPatternType
	msoPatternLightHorizontal     =19         # from enum MsoPatternType
	msoPatternLightUpwardDiagonal =22         # from enum MsoPatternType
	msoPatternLightVertical       =20         # from enum MsoPatternType
	msoPatternMixed               =-2         # from enum MsoPatternType
	msoPatternNarrowHorizontal    =30         # from enum MsoPatternType
	msoPatternNarrowVertical      =29         # from enum MsoPatternType
	msoPatternOutlinedDiamond     =41         # from enum MsoPatternType
	msoPatternPlaid               =42         # from enum MsoPatternType
	msoPatternShingle             =47         # from enum MsoPatternType
	msoPatternSmallCheckerBoard   =17         # from enum MsoPatternType
	msoPatternSmallConfetti       =37         # from enum MsoPatternType
	msoPatternSmallGrid           =23         # from enum MsoPatternType
	msoPatternSolidDiamond        =39         # from enum MsoPatternType
	msoPatternSphere              =43         # from enum MsoPatternType
	msoPatternTrellis             =18         # from enum MsoPatternType
	msoPatternWave                =48         # from enum MsoPatternType
	msoPatternWeave               =44         # from enum MsoPatternType
	msoPatternWideDownwardDiagonal=25         # from enum MsoPatternType
	msoPatternWideUpwardDiagonal  =26         # from enum MsoPatternType
	msoPatternZigZag              =38         # from enum MsoPatternType
	msoPermissionChange           =15         # from enum MsoPermission
	msoPermissionEdit             =2          # from enum MsoPermission
	msoPermissionExtract          =8          # from enum MsoPermission
	msoPermissionFullControl      =64         # from enum MsoPermission
	msoPermissionObjModel         =32         # from enum MsoPermission
	msoPermissionPrint            =16         # from enum MsoPermission
	msoPermissionRead             =1          # from enum MsoPermission
	msoPermissionSave             =4          # from enum MsoPermission
	msoPermissionView             =1          # from enum MsoPermission
	msoPictureAutomatic           =1          # from enum MsoPictureColorType
	msoPictureBlackAndWhite       =3          # from enum MsoPictureColorType
	msoPictureGrayscale           =2          # from enum MsoPictureColorType
	msoPictureMixed               =-2         # from enum MsoPictureColorType
	msoPictureWatermark           =4          # from enum MsoPictureColorType
	msoExtrusionBottom            =2          # from enum MsoPresetExtrusionDirection
	msoExtrusionBottomLeft        =3          # from enum MsoPresetExtrusionDirection
	msoExtrusionBottomRight       =1          # from enum MsoPresetExtrusionDirection
	msoExtrusionLeft              =6          # from enum MsoPresetExtrusionDirection
	msoExtrusionNone              =5          # from enum MsoPresetExtrusionDirection
	msoExtrusionRight             =4          # from enum MsoPresetExtrusionDirection
	msoExtrusionTop               =8          # from enum MsoPresetExtrusionDirection
	msoExtrusionTopLeft           =9          # from enum MsoPresetExtrusionDirection
	msoExtrusionTopRight          =7          # from enum MsoPresetExtrusionDirection
	msoPresetExtrusionDirectionMixed=-2         # from enum MsoPresetExtrusionDirection
	msoGradientBrass              =20         # from enum MsoPresetGradientType
	msoGradientCalmWater          =8          # from enum MsoPresetGradientType
	msoGradientChrome             =21         # from enum MsoPresetGradientType
	msoGradientChromeII           =22         # from enum MsoPresetGradientType
	msoGradientDaybreak           =4          # from enum MsoPresetGradientType
	msoGradientDesert             =6          # from enum MsoPresetGradientType
	msoGradientEarlySunset        =1          # from enum MsoPresetGradientType
	msoGradientFire               =9          # from enum MsoPresetGradientType
	msoGradientFog                =10         # from enum MsoPresetGradientType
	msoGradientGold               =18         # from enum MsoPresetGradientType
	msoGradientGoldII             =19         # from enum MsoPresetGradientType
	msoGradientHorizon            =5          # from enum MsoPresetGradientType
	msoGradientLateSunset         =2          # from enum MsoPresetGradientType
	msoGradientMahogany           =15         # from enum MsoPresetGradientType
	msoGradientMoss               =11         # from enum MsoPresetGradientType
	msoGradientNightfall          =3          # from enum MsoPresetGradientType
	msoGradientOcean              =7          # from enum MsoPresetGradientType
	msoGradientParchment          =14         # from enum MsoPresetGradientType
	msoGradientPeacock            =12         # from enum MsoPresetGradientType
	msoGradientRainbow            =16         # from enum MsoPresetGradientType
	msoGradientRainbowII          =17         # from enum MsoPresetGradientType
	msoGradientSapphire           =24         # from enum MsoPresetGradientType
	msoGradientSilver             =23         # from enum MsoPresetGradientType
	msoGradientWheat              =13         # from enum MsoPresetGradientType
	msoPresetGradientMixed        =-2         # from enum MsoPresetGradientType
	msoLightingBottom             =8          # from enum MsoPresetLightingDirection
	msoLightingBottomLeft         =7          # from enum MsoPresetLightingDirection
	msoLightingBottomRight        =9          # from enum MsoPresetLightingDirection
	msoLightingLeft               =4          # from enum MsoPresetLightingDirection
	msoLightingNone               =5          # from enum MsoPresetLightingDirection
	msoLightingRight              =6          # from enum MsoPresetLightingDirection
	msoLightingTop                =2          # from enum MsoPresetLightingDirection
	msoLightingTopLeft            =1          # from enum MsoPresetLightingDirection
	msoLightingTopRight           =3          # from enum MsoPresetLightingDirection
	msoPresetLightingDirectionMixed=-2         # from enum MsoPresetLightingDirection
	msoLightingBright             =3          # from enum MsoPresetLightingSoftness
	msoLightingDim                =1          # from enum MsoPresetLightingSoftness
	msoLightingNormal             =2          # from enum MsoPresetLightingSoftness
	msoPresetLightingSoftnessMixed=-2         # from enum MsoPresetLightingSoftness
	msoMaterialMatte              =1          # from enum MsoPresetMaterial
	msoMaterialMetal              =3          # from enum MsoPresetMaterial
	msoMaterialPlastic            =2          # from enum MsoPresetMaterial
	msoMaterialWireFrame          =4          # from enum MsoPresetMaterial
	msoPresetMaterialMixed        =-2         # from enum MsoPresetMaterial
	msoTextEffect1                =0          # from enum MsoPresetTextEffect
	msoTextEffect10               =9          # from enum MsoPresetTextEffect
	msoTextEffect11               =10         # from enum MsoPresetTextEffect
	msoTextEffect12               =11         # from enum MsoPresetTextEffect
	msoTextEffect13               =12         # from enum MsoPresetTextEffect
	msoTextEffect14               =13         # from enum MsoPresetTextEffect
	msoTextEffect15               =14         # from enum MsoPresetTextEffect
	msoTextEffect16               =15         # from enum MsoPresetTextEffect
	msoTextEffect17               =16         # from enum MsoPresetTextEffect
	msoTextEffect18               =17         # from enum MsoPresetTextEffect
	msoTextEffect19               =18         # from enum MsoPresetTextEffect
	msoTextEffect2                =1          # from enum MsoPresetTextEffect
	msoTextEffect20               =19         # from enum MsoPresetTextEffect
	msoTextEffect21               =20         # from enum MsoPresetTextEffect
	msoTextEffect22               =21         # from enum MsoPresetTextEffect
	msoTextEffect23               =22         # from enum MsoPresetTextEffect
	msoTextEffect24               =23         # from enum MsoPresetTextEffect
	msoTextEffect25               =24         # from enum MsoPresetTextEffect
	msoTextEffect26               =25         # from enum MsoPresetTextEffect
	msoTextEffect27               =26         # from enum MsoPresetTextEffect
	msoTextEffect28               =27         # from enum MsoPresetTextEffect
	msoTextEffect29               =28         # from enum MsoPresetTextEffect
	msoTextEffect3                =2          # from enum MsoPresetTextEffect
	msoTextEffect30               =29         # from enum MsoPresetTextEffect
	msoTextEffect4                =3          # from enum MsoPresetTextEffect
	msoTextEffect5                =4          # from enum MsoPresetTextEffect
	msoTextEffect6                =5          # from enum MsoPresetTextEffect
	msoTextEffect7                =6          # from enum MsoPresetTextEffect
	msoTextEffect8                =7          # from enum MsoPresetTextEffect
	msoTextEffect9                =8          # from enum MsoPresetTextEffect
	msoTextEffectMixed            =-2         # from enum MsoPresetTextEffect
	msoTextEffectShapeArchDownCurve=10         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeArchDownPour=14         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeArchUpCurve =9          # from enum MsoPresetTextEffectShape
	msoTextEffectShapeArchUpPour  =13         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeButtonCurve =12         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeButtonPour  =16         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeCanDown     =20         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeCanUp       =19         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeCascadeDown =40         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeCascadeUp   =39         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeChevronDown =6          # from enum MsoPresetTextEffectShape
	msoTextEffectShapeChevronUp   =5          # from enum MsoPresetTextEffectShape
	msoTextEffectShapeCircleCurve =11         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeCirclePour  =15         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeCurveDown   =18         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeCurveUp     =17         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeDeflate     =26         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeDeflateBottom=28         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeDeflateInflate=31         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeDeflateInflateDeflate=32         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeDeflateTop  =30         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeDoubleWave1 =23         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeDoubleWave2 =24         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeFadeDown    =36         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeFadeLeft    =34         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeFadeRight   =33         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeFadeUp      =35         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeInflate     =25         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeInflateBottom=27         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeInflateTop  =29         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeMixed       =-2         # from enum MsoPresetTextEffectShape
	msoTextEffectShapePlainText   =1          # from enum MsoPresetTextEffectShape
	msoTextEffectShapeRingInside  =7          # from enum MsoPresetTextEffectShape
	msoTextEffectShapeRingOutside =8          # from enum MsoPresetTextEffectShape
	msoTextEffectShapeSlantDown   =38         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeSlantUp     =37         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeStop        =2          # from enum MsoPresetTextEffectShape
	msoTextEffectShapeTriangleDown=4          # from enum MsoPresetTextEffectShape
	msoTextEffectShapeTriangleUp  =3          # from enum MsoPresetTextEffectShape
	msoTextEffectShapeWave1       =21         # from enum MsoPresetTextEffectShape
	msoTextEffectShapeWave2       =22         # from enum MsoPresetTextEffectShape
	msoPresetTextureMixed         =-2         # from enum MsoPresetTexture
	msoTextureBlueTissuePaper     =17         # from enum MsoPresetTexture
	msoTextureBouquet             =20         # from enum MsoPresetTexture
	msoTextureBrownMarble         =11         # from enum MsoPresetTexture
	msoTextureCanvas              =2          # from enum MsoPresetTexture
	msoTextureCork                =21         # from enum MsoPresetTexture
	msoTextureDenim               =3          # from enum MsoPresetTexture
	msoTextureFishFossil          =7          # from enum MsoPresetTexture
	msoTextureGranite             =12         # from enum MsoPresetTexture
	msoTextureGreenMarble         =9          # from enum MsoPresetTexture
	msoTextureMediumWood          =24         # from enum MsoPresetTexture
	msoTextureNewsprint           =13         # from enum MsoPresetTexture
	msoTextureOak                 =23         # from enum MsoPresetTexture
	msoTexturePaperBag            =6          # from enum MsoPresetTexture
	msoTexturePapyrus             =1          # from enum MsoPresetTexture
	msoTextureParchment           =15         # from enum MsoPresetTexture
	msoTexturePinkTissuePaper     =18         # from enum MsoPresetTexture
	msoTexturePurpleMesh          =19         # from enum MsoPresetTexture
	msoTextureRecycledPaper       =14         # from enum MsoPresetTexture
	msoTextureSand                =8          # from enum MsoPresetTexture
	msoTextureStationery          =16         # from enum MsoPresetTexture
	msoTextureWalnut              =22         # from enum MsoPresetTexture
	msoTextureWaterDroplets       =5          # from enum MsoPresetTexture
	msoTextureWhiteMarble         =10         # from enum MsoPresetTexture
	msoTextureWovenMat            =4          # from enum MsoPresetTexture
	msoPresetThreeDFormatMixed    =-2         # from enum MsoPresetThreeDFormat
	msoThreeD1                    =1          # from enum MsoPresetThreeDFormat
	msoThreeD10                   =10         # from enum MsoPresetThreeDFormat
	msoThreeD11                   =11         # from enum MsoPresetThreeDFormat
	msoThreeD12                   =12         # from enum MsoPresetThreeDFormat
	msoThreeD13                   =13         # from enum MsoPresetThreeDFormat
	msoThreeD14                   =14         # from enum MsoPresetThreeDFormat
	msoThreeD15                   =15         # from enum MsoPresetThreeDFormat
	msoThreeD16                   =16         # from enum MsoPresetThreeDFormat
	msoThreeD17                   =17         # from enum MsoPresetThreeDFormat
	msoThreeD18                   =18         # from enum MsoPresetThreeDFormat
	msoThreeD19                   =19         # from enum MsoPresetThreeDFormat
	msoThreeD2                    =2          # from enum MsoPresetThreeDFormat
	msoThreeD20                   =20         # from enum MsoPresetThreeDFormat
	msoThreeD3                    =3          # from enum MsoPresetThreeDFormat
	msoThreeD4                    =4          # from enum MsoPresetThreeDFormat
	msoThreeD5                    =5          # from enum MsoPresetThreeDFormat
	msoThreeD6                    =6          # from enum MsoPresetThreeDFormat
	msoThreeD7                    =7          # from enum MsoPresetThreeDFormat
	msoThreeD8                    =8          # from enum MsoPresetThreeDFormat
	msoThreeD9                    =9          # from enum MsoPresetThreeDFormat
	msoAfterLastSibling           =4          # from enum MsoRelativeNodePosition
	msoAfterNode                  =2          # from enum MsoRelativeNodePosition
	msoBeforeFirstSibling         =3          # from enum MsoRelativeNodePosition
	msoBeforeNode                 =1          # from enum MsoRelativeNodePosition
	msoScaleFromBottomRight       =2          # from enum MsoScaleFrom
	msoScaleFromMiddle            =1          # from enum MsoScaleFrom
	msoScaleFromTopLeft           =0          # from enum MsoScaleFrom
	msoScreenSize1024x768         =4          # from enum MsoScreenSize
	msoScreenSize1152x882         =5          # from enum MsoScreenSize
	msoScreenSize1152x900         =6          # from enum MsoScreenSize
	msoScreenSize1280x1024        =7          # from enum MsoScreenSize
	msoScreenSize1600x1200        =8          # from enum MsoScreenSize
	msoScreenSize1800x1440        =9          # from enum MsoScreenSize
	msoScreenSize1920x1200        =10         # from enum MsoScreenSize
	msoScreenSize544x376          =0          # from enum MsoScreenSize
	msoScreenSize640x480          =1          # from enum MsoScreenSize
	msoScreenSize720x512          =2          # from enum MsoScreenSize
	msoScreenSize800x600          =3          # from enum MsoScreenSize
	msoScriptLanguageASP          =3          # from enum MsoScriptLanguage
	msoScriptLanguageJava         =1          # from enum MsoScriptLanguage
	msoScriptLanguageOther        =4          # from enum MsoScriptLanguage
	msoScriptLanguageVisualBasic  =2          # from enum MsoScriptLanguage
	msoScriptLocationInBody       =2          # from enum MsoScriptLocation
	msoScriptLocationInHead       =1          # from enum MsoScriptLocation
	msoSearchInCustom             =3          # from enum MsoSearchIn
	msoSearchInMyComputer         =0          # from enum MsoSearchIn
	msoSearchInMyNetworkPlaces    =2          # from enum MsoSearchIn
	msoSearchInOutlook            =1          # from enum MsoSearchIn
	msoSegmentCurve               =1          # from enum MsoSegmentType
	msoSegmentLine                =0          # from enum MsoSegmentType
	msoShadow1                    =1          # from enum MsoShadowType
	msoShadow10                   =10         # from enum MsoShadowType
	msoShadow11                   =11         # from enum MsoShadowType
	msoShadow12                   =12         # from enum MsoShadowType
	msoShadow13                   =13         # from enum MsoShadowType
	msoShadow14                   =14         # from enum MsoShadowType
	msoShadow15                   =15         # from enum MsoShadowType
	msoShadow16                   =16         # from enum MsoShadowType
	msoShadow17                   =17         # from enum MsoShadowType
	msoShadow18                   =18         # from enum MsoShadowType
	msoShadow19                   =19         # from enum MsoShadowType
	msoShadow2                    =2          # from enum MsoShadowType
	msoShadow20                   =20         # from enum MsoShadowType
	msoShadow3                    =3          # from enum MsoShadowType
	msoShadow4                    =4          # from enum MsoShadowType
	msoShadow5                    =5          # from enum MsoShadowType
	msoShadow6                    =6          # from enum MsoShadowType
	msoShadow7                    =7          # from enum MsoShadowType
	msoShadow8                    =8          # from enum MsoShadowType
	msoShadow9                    =9          # from enum MsoShadowType
	msoShadowMixed                =-2         # from enum MsoShadowType
	msoAutoShape                  =1          # from enum MsoShapeType
	msoCallout                    =2          # from enum MsoShapeType
	msoCanvas                     =20         # from enum MsoShapeType
	msoChart                      =3          # from enum MsoShapeType
	msoComment                    =4          # from enum MsoShapeType
	msoDiagram                    =21         # from enum MsoShapeType
	msoEmbeddedOLEObject          =7          # from enum MsoShapeType
	msoFormControl                =8          # from enum MsoShapeType
	msoFreeform                   =5          # from enum MsoShapeType
	msoGroup                      =6          # from enum MsoShapeType
	msoInk                        =22         # from enum MsoShapeType
	msoInkComment                 =23         # from enum MsoShapeType
	msoLine                       =9          # from enum MsoShapeType
	msoLinkedOLEObject            =10         # from enum MsoShapeType
	msoLinkedPicture              =11         # from enum MsoShapeType
	msoMedia                      =16         # from enum MsoShapeType
	msoOLEControlObject           =12         # from enum MsoShapeType
	msoPicture                    =13         # from enum MsoShapeType
	msoPlaceholder                =14         # from enum MsoShapeType
	msoScriptAnchor               =18         # from enum MsoShapeType
	msoShapeTypeMixed             =-2         # from enum MsoShapeType
	msoTable                      =19         # from enum MsoShapeType
	msoTextBox                    =17         # from enum MsoShapeType
	msoTextEffect                 =15         # from enum MsoShapeType
	msoSharedWorkspaceTaskPriorityHigh=1          # from enum MsoSharedWorkspaceTaskPriority
	msoSharedWorkspaceTaskPriorityLow=3          # from enum MsoSharedWorkspaceTaskPriority
	msoSharedWorkspaceTaskPriorityNormal=2          # from enum MsoSharedWorkspaceTaskPriority
	msoSharedWorkspaceTaskStatusCompleted=3          # from enum MsoSharedWorkspaceTaskStatus
	msoSharedWorkspaceTaskStatusDeferred=4          # from enum MsoSharedWorkspaceTaskStatus
	msoSharedWorkspaceTaskStatusInProgress=2          # from enum MsoSharedWorkspaceTaskStatus
	msoSharedWorkspaceTaskStatusNotStarted=1          # from enum MsoSharedWorkspaceTaskStatus
	msoSharedWorkspaceTaskStatusWaiting=5          # from enum MsoSharedWorkspaceTaskStatus
	msoSortByFileName             =1          # from enum MsoSortBy
	msoSortByFileType             =3          # from enum MsoSortBy
	msoSortByLastModified         =4          # from enum MsoSortBy
	msoSortByNone                 =5          # from enum MsoSortBy
	msoSortBySize                 =2          # from enum MsoSortBy
	msoSortOrderAscending         =1          # from enum MsoSortOrder
	msoSortOrderDescending        =2          # from enum MsoSortOrder
	msoSyncAvailableAnywhere      =2          # from enum MsoSyncAvailableType
	msoSyncAvailableNone          =0          # from enum MsoSyncAvailableType
	msoSyncAvailableOffline       =1          # from enum MsoSyncAvailableType
	msoSyncCompareAndMerge        =0          # from enum MsoSyncCompareType
	msoSyncCompareSideBySide      =1          # from enum MsoSyncCompareType
	msoSyncConflictClientWins     =0          # from enum MsoSyncConflictResolutionType
	msoSyncConflictMerge          =2          # from enum MsoSyncConflictResolutionType
	msoSyncConflictServerWins     =1          # from enum MsoSyncConflictResolutionType
	msoSyncErrorCouldNotCompare   =13         # from enum MsoSyncErrorType
	msoSyncErrorCouldNotConnect   =2          # from enum MsoSyncErrorType
	msoSyncErrorCouldNotOpen      =11         # from enum MsoSyncErrorType
	msoSyncErrorCouldNotResolve   =14         # from enum MsoSyncErrorType
	msoSyncErrorCouldNotUpdate    =12         # from enum MsoSyncErrorType
	msoSyncErrorFileInUse         =6          # from enum MsoSyncErrorType
	msoSyncErrorFileNotFound      =4          # from enum MsoSyncErrorType
	msoSyncErrorFileTooLarge      =5          # from enum MsoSyncErrorType
	msoSyncErrorNoNetwork         =15         # from enum MsoSyncErrorType
	msoSyncErrorNone              =0          # from enum MsoSyncErrorType
	msoSyncErrorOutOfSpace        =3          # from enum MsoSyncErrorType
	msoSyncErrorUnauthorizedUser  =1          # from enum MsoSyncErrorType
	msoSyncErrorUnknown           =16         # from enum MsoSyncErrorType
	msoSyncErrorUnknownDownload   =10         # from enum MsoSyncErrorType
	msoSyncErrorUnknownUpload     =9          # from enum MsoSyncErrorType
	msoSyncErrorVirusDownload     =8          # from enum MsoSyncErrorType
	msoSyncErrorVirusUpload       =7          # from enum MsoSyncErrorType
	msoSyncEventDownloadFailed    =2          # from enum MsoSyncEventType
	msoSyncEventDownloadInitiated =0          # from enum MsoSyncEventType
	msoSyncEventDownloadNoChange  =6          # from enum MsoSyncEventType
	msoSyncEventDownloadSucceeded =1          # from enum MsoSyncEventType
	msoSyncEventOffline           =7          # from enum MsoSyncEventType
	msoSyncEventUploadFailed      =5          # from enum MsoSyncEventType
	msoSyncEventUploadInitiated   =3          # from enum MsoSyncEventType
	msoSyncEventUploadSucceeded   =4          # from enum MsoSyncEventType
	msoSyncStatusConflict         =4          # from enum MsoSyncStatusType
	msoSyncStatusError            =6          # from enum MsoSyncStatusType
	msoSyncStatusLatest           =1          # from enum MsoSyncStatusType
	msoSyncStatusLocalChanges     =3          # from enum MsoSyncStatusType
	msoSyncStatusNewerAvailable   =2          # from enum MsoSyncStatusType
	msoSyncStatusNoSharedWorkspace=0          # from enum MsoSyncStatusType
	msoSyncStatusSuspended        =5          # from enum MsoSyncStatusType
	msoSyncVersionLastViewed      =0          # from enum MsoSyncVersionType
	msoSyncVersionServer          =1          # from enum MsoSyncVersionType
	msoTargetBrowserIE4           =2          # from enum MsoTargetBrowser
	msoTargetBrowserIE5           =3          # from enum MsoTargetBrowser
	msoTargetBrowserIE6           =4          # from enum MsoTargetBrowser
	msoTargetBrowserV3            =0          # from enum MsoTargetBrowser
	msoTargetBrowserV4            =1          # from enum MsoTargetBrowser
	msoTextEffectAlignmentCentered=2          # from enum MsoTextEffectAlignment
	msoTextEffectAlignmentLeft    =1          # from enum MsoTextEffectAlignment
	msoTextEffectAlignmentLetterJustify=4          # from enum MsoTextEffectAlignment
	msoTextEffectAlignmentMixed   =-2         # from enum MsoTextEffectAlignment
	msoTextEffectAlignmentRight   =3          # from enum MsoTextEffectAlignment
	msoTextEffectAlignmentStretchJustify=6          # from enum MsoTextEffectAlignment
	msoTextEffectAlignmentWordJustify=5          # from enum MsoTextEffectAlignment
	msoTextOrientationDownward    =3          # from enum MsoTextOrientation
	msoTextOrientationHorizontal  =1          # from enum MsoTextOrientation
	msoTextOrientationHorizontalRotatedFarEast=6          # from enum MsoTextOrientation
	msoTextOrientationMixed       =-2         # from enum MsoTextOrientation
	msoTextOrientationUpward      =2          # from enum MsoTextOrientation
	msoTextOrientationVertical    =5          # from enum MsoTextOrientation
	msoTextOrientationVerticalFarEast=4          # from enum MsoTextOrientation
	msoTexturePreset              =1          # from enum MsoTextureType
	msoTextureTypeMixed           =-2         # from enum MsoTextureType
	msoTextureUserDefined         =2          # from enum MsoTextureType
	msoCTrue                      =1          # from enum MsoTriState
	msoFalse                      =0          # from enum MsoTriState
	msoTriStateMixed              =-2         # from enum MsoTriState
	msoTriStateToggle             =-3         # from enum MsoTriState
	msoTrue                       =-1         # from enum MsoTriState
	msoAnchorBottom               =4          # from enum MsoVerticalAnchor
	msoAnchorBottomBaseLine       =5          # from enum MsoVerticalAnchor
	msoAnchorMiddle               =3          # from enum MsoVerticalAnchor
	msoAnchorTop                  =1          # from enum MsoVerticalAnchor
	msoAnchorTopBaseline          =2          # from enum MsoVerticalAnchor
	msoVerticalAnchorMixed        =-2         # from enum MsoVerticalAnchor
	msoWizardActActive            =1          # from enum MsoWizardActType
	msoWizardActInactive          =0          # from enum MsoWizardActType
	msoWizardActResume            =3          # from enum MsoWizardActType
	msoWizardActSuspend           =2          # from enum MsoWizardActType
	msoWizardMsgLocalStateOff     =2          # from enum MsoWizardMsgType
	msoWizardMsgLocalStateOn      =1          # from enum MsoWizardMsgType
	msoWizardMsgResuming          =5          # from enum MsoWizardMsgType
	msoWizardMsgShowHelp          =3          # from enum MsoWizardMsgType
	msoWizardMsgSuspending        =4          # from enum MsoWizardMsgType
	msoBringForward               =2          # from enum MsoZOrderCmd
	msoBringInFrontOfText         =4          # from enum MsoZOrderCmd
	msoBringToFront               =0          # from enum MsoZOrderCmd
	msoSendBackward               =3          # from enum MsoZOrderCmd
	msoSendBehindText             =5          # from enum MsoZOrderCmd
	msoSendToBack                 =1          # from enum MsoZOrderCmd

from win32com.client import DispatchBaseClass
class Adjustments(DispatchBaseClass):
	CLSID = IID('{000C0310-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(0, LCID, 2, (4, 0), ((3, 1),),Index
			)

	# The method SetItem is actually a property, but must be used as a method to correctly pass the arguments
	def SetItem(self, Index=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(0, LCID, 4, (24, 0), ((3, 1), (4, 1)),Index
			, arg1)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (2, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(0, LCID, 2, (4, 0), ((3, 1),),Index
			)

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(2, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class AnswerWizard(DispatchBaseClass):
	CLSID = IID('{000C0360-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def ClearFileList(self):
		return self._oleobj_.InvokeTypes(1610809346, LCID, 1, (24, 0), (),)

	def ResetFileList(self):
		return self._oleobj_.InvokeTypes(1610809347, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'Files' returns object of type 'AnswerWizardFiles'
		"Files": (1610809345, 2, (9, 0), (), "Files", '{000C0361-0000-0000-C000-000000000046}'),
		"Parent": (1610809344, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}

class AnswerWizardFiles(DispatchBaseClass):
	CLSID = IID('{000C0361-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Add(self, FileName=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1610809347, LCID, 1, (24, 0), ((8, 1),),FileName
			)

	def Delete(self, FileName=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1610809348, LCID, 1, (24, 0), ((8, 1),),FileName
			)

	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(0, LCID, 2, (8, 0), ((3, 1),),Index
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1610809346, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (1610809344, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(0, LCID, 2, (8, 0), ((3, 1),),Index
			)

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1610809346, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class Assistant(DispatchBaseClass):
	CLSID = IID('{000C0322-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def ActivateWizard(self, WizardID=defaultNamedNotOptArg, act=defaultNamedNotOptArg, Animation=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1610809353, LCID, 1, (24, 0), ((3, 1), (3, 1), (12, 17)),WizardID
			, act, Animation)

	def DoAlert(self, bstrAlertTitle=defaultNamedNotOptArg, bstrAlertText=defaultNamedNotOptArg, alb=defaultNamedNotOptArg, alc=defaultNamedNotOptArg
			, ald=defaultNamedNotOptArg, alq=defaultNamedNotOptArg, varfSysAlert=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1610809393, LCID, 1, (3, 0), ((8, 1), (8, 1), (3, 1), (3, 1), (3, 1), (3, 1), (11, 1)),bstrAlertTitle
			, bstrAlertText, alb, alc, ald, alq
			, varfSysAlert)

	def EndWizard(self, WizardID=defaultNamedNotOptArg, varfSuccess=defaultNamedNotOptArg, Animation=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1610809352, LCID, 1, (24, 0), ((3, 1), (11, 1), (12, 17)),WizardID
			, varfSuccess, Animation)

	def Help(self):
		return self._oleobj_.InvokeTypes(1610809350, LCID, 1, (24, 0), (),)

	def Move(self, xLeft=defaultNamedNotOptArg, yTop=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1610809345, LCID, 1, (24, 0), ((3, 1), (3, 1)),xLeft
			, yTop)

	def ResetTips(self):
		return self._oleobj_.InvokeTypes(1610809354, LCID, 1, (24, 0), (),)

	def StartWizard(self, On=defaultNamedNotOptArg, Callback=defaultNamedNotOptArg, PrivateX=defaultNamedNotOptArg, Animation=defaultNamedOptArg
			, CustomTeaser=defaultNamedOptArg, Top=defaultNamedOptArg, Left=defaultNamedOptArg, Bottom=defaultNamedOptArg, Right=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1610809351, LCID, 1, (3, 0), ((11, 1), (8, 1), (3, 1), (12, 17), (12, 17), (12, 17), (12, 17), (12, 17), (12, 17)),On
			, Callback, PrivateX, Animation, CustomTeaser, Top
			, Left, Bottom, Right)

	_prop_map_get_ = {
		"Animation": (1610809359, 2, (3, 0), (), "Animation", None),
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"AssistWithAlerts": (1610809367, 2, (11, 0), (), "AssistWithAlerts", None),
		"AssistWithHelp": (1610809363, 2, (11, 0), (), "AssistWithHelp", None),
		"AssistWithWizards": (1610809365, 2, (11, 0), (), "AssistWithWizards", None),
		"BalloonError": (1610809356, 2, (3, 0), (), "BalloonError", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"FeatureTips": (1610809373, 2, (11, 0), (), "FeatureTips", None),
		"FileName": (1610809388, 2, (8, 0), (), "FileName", None),
		"GuessHelp": (1610809383, 2, (11, 0), (), "GuessHelp", None),
		"HighPriorityTips": (1610809379, 2, (11, 0), (), "HighPriorityTips", None),
		"Item": (0, 2, (8, 0), (), "Item", None),
		"KeyboardShortcutTips": (1610809377, 2, (11, 0), (), "KeyboardShortcutTips", None),
		"Left": (1610809348, 2, (3, 0), (), "Left", None),
		"MouseTips": (1610809375, 2, (11, 0), (), "MouseTips", None),
		"MoveWhenInTheWay": (1610809369, 2, (11, 0), (), "MoveWhenInTheWay", None),
		"Name": (1610809390, 2, (8, 0), (), "Name", None),
		# Method 'NewBalloon' returns object of type 'Balloon'
		"NewBalloon": (1610809355, 2, (9, 0), (), "NewBalloon", '{000C0324-0000-0000-C000-000000000046}'),
		"On": (1610809391, 2, (11, 0), (), "On", None),
		"Parent": (1610809344, 2, (9, 0), (), "Parent", None),
		"Reduced": (1610809361, 2, (11, 0), (), "Reduced", None),
		"SearchWhenProgramming": (1610809385, 2, (11, 0), (), "SearchWhenProgramming", None),
		"Sounds": (1610809371, 2, (11, 0), (), "Sounds", None),
		"TipOfDay": (1610809381, 2, (11, 0), (), "TipOfDay", None),
		"Top": (1610809346, 2, (3, 0), (), "Top", None),
		"Visible": (1610809357, 2, (11, 0), (), "Visible", None),
	}
	_prop_map_put_ = {
		"Animation": ((1610809359, LCID, 4, 0),()),
		"AssistWithAlerts": ((1610809367, LCID, 4, 0),()),
		"AssistWithHelp": ((1610809363, LCID, 4, 0),()),
		"AssistWithWizards": ((1610809365, LCID, 4, 0),()),
		"FeatureTips": ((1610809373, LCID, 4, 0),()),
		"FileName": ((1610809388, LCID, 4, 0),()),
		"GuessHelp": ((1610809383, LCID, 4, 0),()),
		"HighPriorityTips": ((1610809379, LCID, 4, 0),()),
		"KeyboardShortcutTips": ((1610809377, LCID, 4, 0),()),
		"Left": ((1610809348, LCID, 4, 0),()),
		"MouseTips": ((1610809375, LCID, 4, 0),()),
		"MoveWhenInTheWay": ((1610809369, LCID, 4, 0),()),
		"On": ((1610809391, LCID, 4, 0),()),
		"Reduced": ((1610809361, LCID, 4, 0),()),
		"SearchWhenProgramming": ((1610809385, LCID, 4, 0),()),
		"Sounds": ((1610809371, LCID, 4, 0),()),
		"TipOfDay": ((1610809381, LCID, 4, 0),()),
		"Top": ((1610809346, LCID, 4, 0),()),
		"Visible": ((1610809357, LCID, 4, 0),()),
	}
	# Default property for this class is 'Item'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "Item", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class Balloon(DispatchBaseClass):
	CLSID = IID('{000C0324-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Close(self):
		return self._oleobj_.InvokeTypes(1610809368, LCID, 1, (24, 0), (),)

	def SetAvoidRectangle(self, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg, Right=defaultNamedNotOptArg, Bottom=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1610809365, LCID, 1, (24, 0), ((3, 1), (3, 1), (3, 1), (3, 1)),Left
			, Top, Right, Bottom)

	def Show(self):
		return self._oleobj_.InvokeTypes(1610809367, LCID, 1, (3, 0), (),)

	_prop_map_get_ = {
		"Animation": (1610809357, 2, (3, 0), (), "Animation", None),
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"BalloonType": (1610809347, 2, (3, 0), (), "BalloonType", None),
		"Button": (1610809359, 2, (3, 0), (), "Button", None),
		"Callback": (1610809361, 2, (8, 0), (), "Callback", None),
		"Checkboxes": (1610809345, 2, (9, 0), (), "Checkboxes", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Heading": (1610809351, 2, (8, 0), (), "Heading", None),
		"Icon": (1610809349, 2, (3, 0), (), "Icon", None),
		"Labels": (1610809346, 2, (9, 0), (), "Labels", None),
		"Mode": (1610809355, 2, (3, 0), (), "Mode", None),
		"Name": (1610809366, 2, (8, 0), (), "Name", None),
		"Parent": (1610809344, 2, (9, 0), (), "Parent", None),
		"Private": (1610809363, 2, (3, 0), (), "Private", None),
		"Text": (1610809353, 2, (8, 0), (), "Text", None),
	}
	_prop_map_put_ = {
		"Animation": ((1610809357, LCID, 4, 0),()),
		"BalloonType": ((1610809347, LCID, 4, 0),()),
		"Button": ((1610809359, LCID, 4, 0),()),
		"Callback": ((1610809361, LCID, 4, 0),()),
		"Heading": ((1610809351, LCID, 4, 0),()),
		"Icon": ((1610809349, LCID, 4, 0),()),
		"Mode": ((1610809355, LCID, 4, 0),()),
		"Private": ((1610809363, LCID, 4, 0),()),
		"Text": ((1610809353, LCID, 4, 0),()),
	}

class BalloonCheckbox(DispatchBaseClass):
	CLSID = IID('{000C0328-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Checked": (1610809347, 2, (11, 0), (), "Checked", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Item": (0, 2, (8, 0), (), "Item", None),
		"Name": (1610809345, 2, (8, 0), (), "Name", None),
		"Parent": (1610809346, 2, (9, 0), (), "Parent", None),
		"Text": (1610809349, 2, (8, 0), (), "Text", None),
	}
	_prop_map_put_ = {
		"Checked": ((1610809347, LCID, 4, 0),()),
		"Text": ((1610809349, LCID, 4, 0),()),
	}
	# Default property for this class is 'Item'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "Item", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class BalloonCheckboxes(DispatchBaseClass):
	CLSID = IID('{000C0326-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', None)
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1610809347, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Name": (1610809344, 2, (8, 0), (), "Name", None),
		"Parent": (1610809345, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
		"Count": ((1610809347, LCID, 4, 0),()),
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', None)
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, None)
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),None)
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1610809347, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class BalloonLabel(DispatchBaseClass):
	CLSID = IID('{000C0330-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Item": (0, 2, (8, 0), (), "Item", None),
		"Name": (1610809345, 2, (8, 0), (), "Name", None),
		"Parent": (1610809346, 2, (9, 0), (), "Parent", None),
		"Text": (1610809347, 2, (8, 0), (), "Text", None),
	}
	_prop_map_put_ = {
		"Text": ((1610809347, LCID, 4, 0),()),
	}
	# Default property for this class is 'Item'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "Item", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class BalloonLabels(DispatchBaseClass):
	CLSID = IID('{000C032E-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', None)
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1610809347, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Name": (1610809344, 2, (8, 0), (), "Name", None),
		"Parent": (1610809345, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
		"Count": ((1610809347, LCID, 4, 0),()),
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', None)
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, None)
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),None)
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1610809347, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class COMAddIn(DispatchBaseClass):
	CLSID = IID('{000C033A-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Connect": (6, 2, (11, 0), (), "Connect", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Description": (0, 2, (8, 0), (), "Description", None),
		"Guid": (4, 2, (8, 0), (), "Guid", None),
		"Object": (7, 2, (9, 0), (), "Object", None),
		"Parent": (8, 2, (9, 0), (), "Parent", None),
		"ProgId": (3, 2, (8, 0), (), "ProgId", None),
	}
	_prop_map_put_ = {
		"Connect": ((6, LCID, 4, 0),()),
		"Description": ((0, LCID, 4, 0),()),
		"Object": ((7, LCID, 4, 0),()),
	}
	# Default property for this class is 'Description'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "Description", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class COMAddIns(DispatchBaseClass):
	CLSID = IID('{000C0339-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type COMAddIn
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((16396, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C033A-0000-0000-C000-000000000046}')
		return ret

	def SetAppModal(self, varfModal=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (24, 0), ((11, 1),),varfModal
			)

	def Update(self):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (3, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((16396, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C033A-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C033A-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C033A-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class CalloutFormat(DispatchBaseClass):
	CLSID = IID('{000C0311-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def AutomaticLength(self):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), (),)

	def CustomDrop(self, Drop=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), ((4, 1),),Drop
			)

	def CustomLength(self, Length=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(12, LCID, 1, (24, 0), ((4, 1),),Length
			)

	def PresetDrop(self, DropType=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (24, 0), ((3, 1),),DropType
			)

	_prop_map_get_ = {
		"Accent": (100, 2, (3, 0), (), "Accent", None),
		"Angle": (101, 2, (3, 0), (), "Angle", None),
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"AutoAttach": (102, 2, (3, 0), (), "AutoAttach", None),
		"AutoLength": (103, 2, (3, 0), (), "AutoLength", None),
		"Border": (104, 2, (3, 0), (), "Border", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Drop": (105, 2, (4, 0), (), "Drop", None),
		"DropType": (106, 2, (3, 0), (), "DropType", None),
		"Gap": (107, 2, (4, 0), (), "Gap", None),
		"Length": (108, 2, (4, 0), (), "Length", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		"Type": (109, 2, (3, 0), (), "Type", None),
	}
	_prop_map_put_ = {
		"Accent": ((100, LCID, 4, 0),()),
		"Angle": ((101, LCID, 4, 0),()),
		"AutoAttach": ((102, LCID, 4, 0),()),
		"Border": ((104, LCID, 4, 0),()),
		"Gap": ((107, LCID, 4, 0),()),
		"Type": ((109, LCID, 4, 0),()),
	}

class CanvasShapes(DispatchBaseClass):
	CLSID = IID('{000C0371-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type Shape
	def AddCallout(self, Type=defaultNamedNotOptArg, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg, Width=defaultNamedNotOptArg
			, Height=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(10, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1), (4, 1), (4, 1)),Type
			, Left, Top, Width, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddCallout', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddConnector(self, Type=defaultNamedNotOptArg, BeginX=defaultNamedNotOptArg, BeginY=defaultNamedNotOptArg, EndX=defaultNamedNotOptArg
			, EndY=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(11, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1), (4, 1), (4, 1)),Type
			, BeginX, BeginY, EndX, EndY)
		if ret is not None:
			ret = Dispatch(ret, u'AddConnector', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddCurve(self, SafeArrayOfPoints=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(12, LCID, 1, (9, 0), ((12, 1),),SafeArrayOfPoints
			)
		if ret is not None:
			ret = Dispatch(ret, u'AddCurve', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddLabel(self, Orientation=defaultNamedNotOptArg, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg, Width=defaultNamedNotOptArg
			, Height=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(13, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1), (4, 1), (4, 1)),Orientation
			, Left, Top, Width, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddLabel', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddLine(self, BeginX=defaultNamedNotOptArg, BeginY=defaultNamedNotOptArg, EndX=defaultNamedNotOptArg, EndY=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(14, LCID, 1, (9, 0), ((4, 1), (4, 1), (4, 1), (4, 1)),BeginX
			, BeginY, EndX, EndY)
		if ret is not None:
			ret = Dispatch(ret, u'AddLine', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddPicture(self, FileName=defaultNamedNotOptArg, LinkToFile=defaultNamedNotOptArg, SaveWithDocument=defaultNamedNotOptArg, Left=defaultNamedNotOptArg
			, Top=defaultNamedNotOptArg, Width=-1.0, Height=-1.0):
		ret = self._oleobj_.InvokeTypes(15, LCID, 1, (9, 0), ((8, 1), (3, 1), (3, 1), (4, 1), (4, 1), (4, 49), (4, 49)),FileName
			, LinkToFile, SaveWithDocument, Left, Top, Width
			, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddPicture', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddPolyline(self, SafeArrayOfPoints=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(16, LCID, 1, (9, 0), ((12, 1),),SafeArrayOfPoints
			)
		if ret is not None:
			ret = Dispatch(ret, u'AddPolyline', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddShape(self, Type=defaultNamedNotOptArg, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg, Width=defaultNamedNotOptArg
			, Height=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(17, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1), (4, 1), (4, 1)),Type
			, Left, Top, Width, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddShape', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddTextEffect(self, PresetTextEffect=defaultNamedNotOptArg, Text=defaultNamedNotOptArg, FontName=defaultNamedNotOptArg, FontSize=defaultNamedNotOptArg
			, FontBold=defaultNamedNotOptArg, FontItalic=defaultNamedNotOptArg, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(18, LCID, 1, (9, 0), ((3, 1), (8, 1), (8, 1), (4, 1), (3, 1), (3, 1), (4, 1), (4, 1)),PresetTextEffect
			, Text, FontName, FontSize, FontBold, FontItalic
			, Left, Top)
		if ret is not None:
			ret = Dispatch(ret, u'AddTextEffect', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddTextbox(self, Orientation=defaultNamedNotOptArg, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg, Width=defaultNamedNotOptArg
			, Height=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(19, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1), (4, 1), (4, 1)),Orientation
			, Left, Top, Width, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddTextbox', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type FreeformBuilder
	def BuildFreeform(self, EditingType=defaultNamedNotOptArg, X1=defaultNamedNotOptArg, Y1=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(20, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1)),EditingType
			, X1, Y1)
		if ret is not None:
			ret = Dispatch(ret, u'BuildFreeform', '{000C0315-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type ShapeRange
	def Range(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(21, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Range', '{000C031D-0000-0000-C000-000000000046}')
		return ret

	def SelectAll(self):
		return self._oleobj_.InvokeTypes(22, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		# Method 'Background' returns object of type 'Shape'
		"Background": (100, 2, (9, 0), (), "Background", '{000C031C-0000-0000-C000-000000000046}'),
		"Count": (2, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C031C-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C031C-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(2, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class ColorFormat(DispatchBaseClass):
	CLSID = IID('{000C0312-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		"RGB": (0, 2, (3, 0), (), "RGB", None),
		"SchemeColor": (100, 2, (3, 0), (), "SchemeColor", None),
		"TintAndShade": (103, 2, (4, 0), (), "TintAndShade", None),
		"Type": (101, 2, (3, 0), (), "Type", None),
	}
	_prop_map_put_ = {
		"RGB": ((0, LCID, 4, 0),()),
		"SchemeColor": ((100, LCID, 4, 0),()),
		"TintAndShade": ((103, LCID, 4, 0),()),
	}
	# Default property for this class is 'RGB'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (3, 0), (), "RGB", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class CommandBar(DispatchBaseClass):
	CLSID = IID('{000C0304-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Delete(self):
		return self._oleobj_.InvokeTypes(1610874884, LCID, 1, (24, 0), (),)

	# Result is of type CommandBarControl
	def FindControl(self, Type=defaultNamedOptArg, Id=defaultNamedOptArg, Tag=defaultNamedOptArg, Visible=defaultNamedOptArg
			, Recursive=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610874887, LCID, 1, (9, 0), ((12, 17), (12, 17), (12, 17), (12, 17), (12, 17)),Type
			, Id, Tag, Visible, Recursive)
		if ret is not None:
			ret = Dispatch(ret, u'FindControl', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	# The method GetaccDefaultAction is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDefaultAction(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5013, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccDescription is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDescription(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5005, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelp is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelp(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5008, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelpTopic is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelpTopic(self, pszHelpFile=global_Missing, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5009, 2, (3, 0), ((16392, 2), (12, 17)), u'GetaccHelpTopic', None,pszHelpFile
			, varChild)

	# The method GetaccKeyboardShortcut is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccKeyboardShortcut(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5010, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccName(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5003, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccRole is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccRole(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5006, 2, (12, 0), ((12, 17),), u'GetaccRole', None,varChild
			)

	# The method GetaccState is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccState(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5007, 2, (12, 0), ((12, 17),), u'GetaccState', None,varChild
			)

	# The method GetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccValue(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5004, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	def Reset(self):
		return self._oleobj_.InvokeTypes(1610874905, LCID, 1, (24, 0), (),)

	# The method SetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccName(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5003, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method SetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccValue(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5004, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	def ShowPopup(self, x=defaultNamedOptArg, y=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1610874906, LCID, 1, (24, 0), ((12, 17), (12, 17)),x
			, y)

	# The method accChild is actually a property, but must be used as a method to correctly pass the arguments
	def accChild(self, varChild=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(-5002, LCID, 2, (9, 0), ((12, 1),),varChild
			)
		if ret is not None:
			ret = Dispatch(ret, u'accChild', None)
		return ret

	def accDoDefaultAction(self, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5018, LCID, 1, (24, 0), ((12, 17),),varChild
			)

	def accHitTest(self, xLeft=defaultNamedNotOptArg, yTop=defaultNamedNotOptArg):
		return self._ApplyTypes_(-5017, 1, (12, 0), ((3, 1), (3, 1)), u'accHitTest', None,xLeft
			, yTop)

	def accLocation(self, pxLeft=global_Missing, pyTop=global_Missing, pcxWidth=global_Missing, pcyHeight=global_Missing
			, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5015, 1, (24, 0), ((16387, 2), (16387, 2), (16387, 2), (16387, 2), (12, 17)), u'accLocation', None,pxLeft
			, pyTop, pcxWidth, pcyHeight, varChild)

	def accNavigate(self, navDir=defaultNamedNotOptArg, varStart=defaultNamedOptArg):
		return self._ApplyTypes_(-5016, 1, (12, 0), ((3, 1), (12, 17)), u'accNavigate', None,navDir
			, varStart)

	def accSelect(self, flagsSelect=defaultNamedNotOptArg, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5014, LCID, 1, (24, 0), ((3, 1), (12, 17)),flagsSelect
			, varChild)

	_prop_map_get_ = {
		"AdaptiveMenu": (1610874914, 2, (11, 0), (), "AdaptiveMenu", None),
		"Application": (1610809344, 2, (9, 0), (), "Application", None),
		"BuiltIn": (1610874880, 2, (11, 0), (), "BuiltIn", None),
		"Context": (1610874881, 2, (8, 0), (), "Context", None),
		# Method 'Controls' returns object of type 'CommandBarControls'
		"Controls": (1610874883, 2, (9, 0), (), "Controls", '{000C0306-0000-0000-C000-000000000046}'),
		"Creator": (1610809345, 2, (3, 0), (), "Creator", None),
		"Enabled": (1610874885, 2, (11, 0), (), "Enabled", None),
		"Height": (1610874888, 2, (3, 0), (), "Height", None),
		"Id": (1610874916, 2, (3, 0), (), "Id", None),
		"Index": (1610874890, 2, (3, 0), (), "Index", None),
		"InstanceId": (1610874891, 2, (3, 0), (), "InstanceId", None),
		"Left": (1610874892, 2, (3, 0), (), "Left", None),
		"Name": (1610874894, 2, (8, 0), (), "Name", None),
		"NameLocal": (1610874896, 2, (8, 0), (), "NameLocal", None),
		"Parent": (1610874898, 2, (9, 0), (), "Parent", None),
		"Position": (1610874899, 2, (3, 0), (), "Position", None),
		"Protection": (1610874903, 2, (3, 0), (), "Protection", None),
		"RowIndex": (1610874901, 2, (3, 0), (), "RowIndex", None),
		"Top": (1610874907, 2, (3, 0), (), "Top", None),
		"Type": (1610874909, 2, (3, 0), (), "Type", None),
		"Visible": (1610874910, 2, (11, 0), (), "Visible", None),
		"Width": (1610874912, 2, (3, 0), (), "Width", None),
		"accChildCount": (-5001, 2, (3, 0), (), "accChildCount", None),
		"accDefaultAction": (-5013, 2, (8, 0), ((12, 17),), "accDefaultAction", None),
		"accDescription": (-5005, 2, (8, 0), ((12, 17),), "accDescription", None),
		"accFocus": (-5011, 2, (12, 0), (), "accFocus", None),
		"accHelp": (-5008, 2, (8, 0), ((12, 17),), "accHelp", None),
		"accHelpTopic": (-5009, 2, (3, 0), ((16392, 2), (12, 17)), "accHelpTopic", None),
		"accKeyboardShortcut": (-5010, 2, (8, 0), ((12, 17),), "accKeyboardShortcut", None),
		"accName": (-5003, 2, (8, 0), ((12, 17),), "accName", None),
		"accParent": (-5000, 2, (9, 0), (), "accParent", None),
		"accRole": (-5006, 2, (12, 0), ((12, 17),), "accRole", None),
		"accSelection": (-5012, 2, (12, 0), (), "accSelection", None),
		"accState": (-5007, 2, (12, 0), ((12, 17),), "accState", None),
		"accValue": (-5004, 2, (8, 0), ((12, 17),), "accValue", None),
	}
	_prop_map_put_ = {
		"AdaptiveMenu": ((1610874914, LCID, 4, 0),()),
		"Context": ((1610874881, LCID, 4, 0),()),
		"Enabled": ((1610874885, LCID, 4, 0),()),
		"Height": ((1610874888, LCID, 4, 0),()),
		"Left": ((1610874892, LCID, 4, 0),()),
		"Name": ((1610874894, LCID, 4, 0),()),
		"NameLocal": ((1610874896, LCID, 4, 0),()),
		"Position": ((1610874899, LCID, 4, 0),()),
		"Protection": ((1610874903, LCID, 4, 0),()),
		"RowIndex": ((1610874901, LCID, 4, 0),()),
		"Top": ((1610874907, LCID, 4, 0),()),
		"Visible": ((1610874910, LCID, 4, 0),()),
		"Width": ((1610874912, LCID, 4, 0),()),
		"accName": ((-5003, LCID, 4, 0),()),
		"accValue": ((-5004, LCID, 4, 0),()),
	}

class CommandBarControl(DispatchBaseClass):
	CLSID = IID('{000C0308-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type CommandBarControl
	def Copy(self, Bar=defaultNamedOptArg, Before=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610874886, LCID, 1, (9, 0), ((12, 17), (12, 17)),Bar
			, Before)
		if ret is not None:
			ret = Dispatch(ret, u'Copy', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	def Delete(self, Temporary=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1610874887, LCID, 1, (24, 0), ((12, 17),),Temporary
			)

	def Execute(self):
		return self._oleobj_.InvokeTypes(1610874892, LCID, 1, (24, 0), (),)

	# The method GetaccDefaultAction is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDefaultAction(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5013, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccDescription is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDescription(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5005, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelp is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelp(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5008, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelpTopic is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelpTopic(self, pszHelpFile=global_Missing, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5009, 2, (3, 0), ((16392, 2), (12, 17)), u'GetaccHelpTopic', None,pszHelpFile
			, varChild)

	# The method GetaccKeyboardShortcut is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccKeyboardShortcut(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5010, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccName(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5003, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccRole is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccRole(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5006, 2, (12, 0), ((12, 17),), u'GetaccRole', None,varChild
			)

	# The method GetaccState is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccState(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5007, 2, (12, 0), ((12, 17),), u'GetaccState', None,varChild
			)

	# The method GetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccValue(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5004, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# Result is of type CommandBarControl
	def Move(self, Bar=defaultNamedOptArg, Before=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610874902, LCID, 1, (9, 0), ((12, 17), (12, 17)),Bar
			, Before)
		if ret is not None:
			ret = Dispatch(ret, u'Move', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	def Reserved1(self):
		return self._oleobj_.InvokeTypes(1610874926, LCID, 1, (24, 0), (),)

	def Reserved2(self):
		return self._oleobj_.InvokeTypes(1610874927, LCID, 1, (24, 0), (),)

	def Reserved3(self):
		return self._oleobj_.InvokeTypes(1610874928, LCID, 1, (24, 0), (),)

	def Reserved4(self):
		return self._oleobj_.InvokeTypes(1610874929, LCID, 1, (24, 0), (),)

	def Reserved5(self):
		return self._oleobj_.InvokeTypes(1610874930, LCID, 1, (24, 0), (),)

	def Reserved6(self):
		return self._oleobj_.InvokeTypes(1610874931, LCID, 1, (24, 0), (),)

	def Reserved7(self):
		return self._oleobj_.InvokeTypes(1610874932, LCID, 1, (24, 0), (),)

	def Reset(self):
		return self._oleobj_.InvokeTypes(1610874913, LCID, 1, (24, 0), (),)

	def SetFocus(self):
		return self._oleobj_.InvokeTypes(1610874914, LCID, 1, (24, 0), (),)

	# The method SetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccName(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5003, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method SetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccValue(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5004, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method accChild is actually a property, but must be used as a method to correctly pass the arguments
	def accChild(self, varChild=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(-5002, LCID, 2, (9, 0), ((12, 1),),varChild
			)
		if ret is not None:
			ret = Dispatch(ret, u'accChild', None)
		return ret

	def accDoDefaultAction(self, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5018, LCID, 1, (24, 0), ((12, 17),),varChild
			)

	def accHitTest(self, xLeft=defaultNamedNotOptArg, yTop=defaultNamedNotOptArg):
		return self._ApplyTypes_(-5017, 1, (12, 0), ((3, 1), (3, 1)), u'accHitTest', None,xLeft
			, yTop)

	def accLocation(self, pxLeft=global_Missing, pyTop=global_Missing, pcxWidth=global_Missing, pcyHeight=global_Missing
			, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5015, 1, (24, 0), ((16387, 2), (16387, 2), (16387, 2), (16387, 2), (12, 17)), u'accLocation', None,pxLeft
			, pyTop, pcxWidth, pcyHeight, varChild)

	def accNavigate(self, navDir=defaultNamedNotOptArg, varStart=defaultNamedOptArg):
		return self._ApplyTypes_(-5016, 1, (12, 0), ((3, 1), (12, 17)), u'accNavigate', None,navDir
			, varStart)

	def accSelect(self, flagsSelect=defaultNamedNotOptArg, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5014, LCID, 1, (24, 0), ((3, 1), (12, 17)),flagsSelect
			, varChild)

	_prop_map_get_ = {
		"Application": (1610809344, 2, (9, 0), (), "Application", None),
		"BeginGroup": (1610874880, 2, (11, 0), (), "BeginGroup", None),
		"BuiltIn": (1610874882, 2, (11, 0), (), "BuiltIn", None),
		"Caption": (1610874883, 2, (8, 0), (), "Caption", None),
		"Control": (1610874885, 2, (9, 0), (), "Control", None),
		"Creator": (1610809345, 2, (3, 0), (), "Creator", None),
		"DescriptionText": (1610874888, 2, (8, 0), (), "DescriptionText", None),
		"Enabled": (1610874890, 2, (11, 0), (), "Enabled", None),
		"Height": (1610874893, 2, (3, 0), (), "Height", None),
		"HelpContextId": (1610874895, 2, (3, 0), (), "HelpContextId", None),
		"HelpFile": (1610874897, 2, (8, 0), (), "HelpFile", None),
		"Id": (1610874899, 2, (3, 0), (), "Id", None),
		"Index": (1610874900, 2, (3, 0), (), "Index", None),
		"InstanceId": (1610874901, 2, (3, 0), (), "InstanceId", None),
		"IsPriorityDropped": (1610874925, 2, (11, 0), (), "IsPriorityDropped", None),
		"Left": (1610874903, 2, (3, 0), (), "Left", None),
		"OLEUsage": (1610874904, 2, (3, 0), (), "OLEUsage", None),
		"OnAction": (1610874906, 2, (8, 0), (), "OnAction", None),
		"Parameter": (1610874909, 2, (8, 0), (), "Parameter", None),
		# Method 'Parent' returns object of type 'CommandBar'
		"Parent": (1610874908, 2, (9, 0), (), "Parent", '{000C0304-0000-0000-C000-000000000046}'),
		"Priority": (1610874911, 2, (3, 0), (), "Priority", None),
		"Tag": (1610874915, 2, (8, 0), (), "Tag", None),
		"TooltipText": (1610874917, 2, (8, 0), (), "TooltipText", None),
		"Top": (1610874919, 2, (3, 0), (), "Top", None),
		"Type": (1610874920, 2, (3, 0), (), "Type", None),
		"Visible": (1610874921, 2, (11, 0), (), "Visible", None),
		"Width": (1610874923, 2, (3, 0), (), "Width", None),
		"accChildCount": (-5001, 2, (3, 0), (), "accChildCount", None),
		"accDefaultAction": (-5013, 2, (8, 0), ((12, 17),), "accDefaultAction", None),
		"accDescription": (-5005, 2, (8, 0), ((12, 17),), "accDescription", None),
		"accFocus": (-5011, 2, (12, 0), (), "accFocus", None),
		"accHelp": (-5008, 2, (8, 0), ((12, 17),), "accHelp", None),
		"accHelpTopic": (-5009, 2, (3, 0), ((16392, 2), (12, 17)), "accHelpTopic", None),
		"accKeyboardShortcut": (-5010, 2, (8, 0), ((12, 17),), "accKeyboardShortcut", None),
		"accName": (-5003, 2, (8, 0), ((12, 17),), "accName", None),
		"accParent": (-5000, 2, (9, 0), (), "accParent", None),
		"accRole": (-5006, 2, (12, 0), ((12, 17),), "accRole", None),
		"accSelection": (-5012, 2, (12, 0), (), "accSelection", None),
		"accState": (-5007, 2, (12, 0), ((12, 17),), "accState", None),
		"accValue": (-5004, 2, (8, 0), ((12, 17),), "accValue", None),
	}
	_prop_map_put_ = {
		"BeginGroup": ((1610874880, LCID, 4, 0),()),
		"Caption": ((1610874883, LCID, 4, 0),()),
		"DescriptionText": ((1610874888, LCID, 4, 0),()),
		"Enabled": ((1610874890, LCID, 4, 0),()),
		"Height": ((1610874893, LCID, 4, 0),()),
		"HelpContextId": ((1610874895, LCID, 4, 0),()),
		"HelpFile": ((1610874897, LCID, 4, 0),()),
		"OLEUsage": ((1610874904, LCID, 4, 0),()),
		"OnAction": ((1610874906, LCID, 4, 0),()),
		"Parameter": ((1610874909, LCID, 4, 0),()),
		"Priority": ((1610874911, LCID, 4, 0),()),
		"Tag": ((1610874915, LCID, 4, 0),()),
		"TooltipText": ((1610874917, LCID, 4, 0),()),
		"Visible": ((1610874921, LCID, 4, 0),()),
		"Width": ((1610874923, LCID, 4, 0),()),
		"accName": ((-5003, LCID, 4, 0),()),
		"accValue": ((-5004, LCID, 4, 0),()),
	}

class CommandBarControls(DispatchBaseClass):
	CLSID = IID('{000C0306-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type CommandBarControl
	def Add(self, Type=defaultNamedOptArg, Id=defaultNamedOptArg, Parameter=defaultNamedOptArg, Before=defaultNamedOptArg
			, Temporary=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610809344, LCID, 1, (9, 0), ((12, 17), (12, 17), (12, 17), (12, 17), (12, 17)),Type
			, Id, Parameter, Before, Temporary)
		if ret is not None:
			ret = Dispatch(ret, u'Add', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	# Result is of type CommandBarControl
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1610809345, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'Parent' returns object of type 'CommandBar'
		"Parent": (1610809348, 2, (9, 0), (), "Parent", '{000C0304-0000-0000-C000-000000000046}'),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0308-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0308-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1610809345, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class CommandBarPopup(DispatchBaseClass):
	CLSID = IID('{000C030A-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type CommandBarControl
	def Copy(self, Bar=defaultNamedOptArg, Before=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610874886, LCID, 1, (9, 0), ((12, 17), (12, 17)),Bar
			, Before)
		if ret is not None:
			ret = Dispatch(ret, u'Copy', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	def Delete(self, Temporary=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1610874887, LCID, 1, (24, 0), ((12, 17),),Temporary
			)

	def Execute(self):
		return self._oleobj_.InvokeTypes(1610874892, LCID, 1, (24, 0), (),)

	# The method GetaccDefaultAction is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDefaultAction(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5013, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccDescription is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDescription(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5005, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelp is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelp(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5008, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelpTopic is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelpTopic(self, pszHelpFile=global_Missing, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5009, 2, (3, 0), ((16392, 2), (12, 17)), u'GetaccHelpTopic', None,pszHelpFile
			, varChild)

	# The method GetaccKeyboardShortcut is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccKeyboardShortcut(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5010, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccName(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5003, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccRole is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccRole(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5006, 2, (12, 0), ((12, 17),), u'GetaccRole', None,varChild
			)

	# The method GetaccState is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccState(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5007, 2, (12, 0), ((12, 17),), u'GetaccState', None,varChild
			)

	# The method GetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccValue(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5004, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# Result is of type CommandBarControl
	def Move(self, Bar=defaultNamedOptArg, Before=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610874902, LCID, 1, (9, 0), ((12, 17), (12, 17)),Bar
			, Before)
		if ret is not None:
			ret = Dispatch(ret, u'Move', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	def Reserved1(self):
		return self._oleobj_.InvokeTypes(1610874926, LCID, 1, (24, 0), (),)

	def Reserved2(self):
		return self._oleobj_.InvokeTypes(1610874927, LCID, 1, (24, 0), (),)

	def Reserved3(self):
		return self._oleobj_.InvokeTypes(1610874928, LCID, 1, (24, 0), (),)

	def Reserved4(self):
		return self._oleobj_.InvokeTypes(1610874929, LCID, 1, (24, 0), (),)

	def Reserved5(self):
		return self._oleobj_.InvokeTypes(1610874930, LCID, 1, (24, 0), (),)

	def Reserved6(self):
		return self._oleobj_.InvokeTypes(1610874931, LCID, 1, (24, 0), (),)

	def Reserved7(self):
		return self._oleobj_.InvokeTypes(1610874932, LCID, 1, (24, 0), (),)

	def Reset(self):
		return self._oleobj_.InvokeTypes(1610874913, LCID, 1, (24, 0), (),)

	def SetFocus(self):
		return self._oleobj_.InvokeTypes(1610874914, LCID, 1, (24, 0), (),)

	# The method SetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccName(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5003, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method SetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccValue(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5004, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method accChild is actually a property, but must be used as a method to correctly pass the arguments
	def accChild(self, varChild=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(-5002, LCID, 2, (9, 0), ((12, 1),),varChild
			)
		if ret is not None:
			ret = Dispatch(ret, u'accChild', None)
		return ret

	def accDoDefaultAction(self, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5018, LCID, 1, (24, 0), ((12, 17),),varChild
			)

	def accHitTest(self, xLeft=defaultNamedNotOptArg, yTop=defaultNamedNotOptArg):
		return self._ApplyTypes_(-5017, 1, (12, 0), ((3, 1), (3, 1)), u'accHitTest', None,xLeft
			, yTop)

	def accLocation(self, pxLeft=global_Missing, pyTop=global_Missing, pcxWidth=global_Missing, pcyHeight=global_Missing
			, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5015, 1, (24, 0), ((16387, 2), (16387, 2), (16387, 2), (16387, 2), (12, 17)), u'accLocation', None,pxLeft
			, pyTop, pcxWidth, pcyHeight, varChild)

	def accNavigate(self, navDir=defaultNamedNotOptArg, varStart=defaultNamedOptArg):
		return self._ApplyTypes_(-5016, 1, (12, 0), ((3, 1), (12, 17)), u'accNavigate', None,navDir
			, varStart)

	def accSelect(self, flagsSelect=defaultNamedNotOptArg, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5014, LCID, 1, (24, 0), ((3, 1), (12, 17)),flagsSelect
			, varChild)

	_prop_map_get_ = {
		"Application": (1610809344, 2, (9, 0), (), "Application", None),
		"BeginGroup": (1610874880, 2, (11, 0), (), "BeginGroup", None),
		"BuiltIn": (1610874882, 2, (11, 0), (), "BuiltIn", None),
		"Caption": (1610874883, 2, (8, 0), (), "Caption", None),
		# Method 'CommandBar' returns object of type 'CommandBar'
		"CommandBar": (1610940416, 2, (9, 0), (), "CommandBar", '{000C0304-0000-0000-C000-000000000046}'),
		"Control": (1610874885, 2, (9, 0), (), "Control", None),
		# Method 'Controls' returns object of type 'CommandBarControls'
		"Controls": (1610940417, 2, (9, 0), (), "Controls", '{000C0306-0000-0000-C000-000000000046}'),
		"Creator": (1610809345, 2, (3, 0), (), "Creator", None),
		"DescriptionText": (1610874888, 2, (8, 0), (), "DescriptionText", None),
		"Enabled": (1610874890, 2, (11, 0), (), "Enabled", None),
		"Height": (1610874893, 2, (3, 0), (), "Height", None),
		"HelpContextId": (1610874895, 2, (3, 0), (), "HelpContextId", None),
		"HelpFile": (1610874897, 2, (8, 0), (), "HelpFile", None),
		"Id": (1610874899, 2, (3, 0), (), "Id", None),
		"Index": (1610874900, 2, (3, 0), (), "Index", None),
		"InstanceId": (1610874901, 2, (3, 0), (), "InstanceId", None),
		"IsPriorityDropped": (1610874925, 2, (11, 0), (), "IsPriorityDropped", None),
		"Left": (1610874903, 2, (3, 0), (), "Left", None),
		"OLEMenuGroup": (1610940418, 2, (3, 0), (), "OLEMenuGroup", None),
		"OLEUsage": (1610874904, 2, (3, 0), (), "OLEUsage", None),
		"OnAction": (1610874906, 2, (8, 0), (), "OnAction", None),
		"Parameter": (1610874909, 2, (8, 0), (), "Parameter", None),
		# Method 'Parent' returns object of type 'CommandBar'
		"Parent": (1610874908, 2, (9, 0), (), "Parent", '{000C0304-0000-0000-C000-000000000046}'),
		"Priority": (1610874911, 2, (3, 0), (), "Priority", None),
		"Tag": (1610874915, 2, (8, 0), (), "Tag", None),
		"TooltipText": (1610874917, 2, (8, 0), (), "TooltipText", None),
		"Top": (1610874919, 2, (3, 0), (), "Top", None),
		"Type": (1610874920, 2, (3, 0), (), "Type", None),
		"Visible": (1610874921, 2, (11, 0), (), "Visible", None),
		"Width": (1610874923, 2, (3, 0), (), "Width", None),
		"accChildCount": (-5001, 2, (3, 0), (), "accChildCount", None),
		"accDefaultAction": (-5013, 2, (8, 0), ((12, 17),), "accDefaultAction", None),
		"accDescription": (-5005, 2, (8, 0), ((12, 17),), "accDescription", None),
		"accFocus": (-5011, 2, (12, 0), (), "accFocus", None),
		"accHelp": (-5008, 2, (8, 0), ((12, 17),), "accHelp", None),
		"accHelpTopic": (-5009, 2, (3, 0), ((16392, 2), (12, 17)), "accHelpTopic", None),
		"accKeyboardShortcut": (-5010, 2, (8, 0), ((12, 17),), "accKeyboardShortcut", None),
		"accName": (-5003, 2, (8, 0), ((12, 17),), "accName", None),
		"accParent": (-5000, 2, (9, 0), (), "accParent", None),
		"accRole": (-5006, 2, (12, 0), ((12, 17),), "accRole", None),
		"accSelection": (-5012, 2, (12, 0), (), "accSelection", None),
		"accState": (-5007, 2, (12, 0), ((12, 17),), "accState", None),
		"accValue": (-5004, 2, (8, 0), ((12, 17),), "accValue", None),
	}
	_prop_map_put_ = {
		"BeginGroup": ((1610874880, LCID, 4, 0),()),
		"Caption": ((1610874883, LCID, 4, 0),()),
		"DescriptionText": ((1610874888, LCID, 4, 0),()),
		"Enabled": ((1610874890, LCID, 4, 0),()),
		"Height": ((1610874893, LCID, 4, 0),()),
		"HelpContextId": ((1610874895, LCID, 4, 0),()),
		"HelpFile": ((1610874897, LCID, 4, 0),()),
		"OLEMenuGroup": ((1610940418, LCID, 4, 0),()),
		"OLEUsage": ((1610874904, LCID, 4, 0),()),
		"OnAction": ((1610874906, LCID, 4, 0),()),
		"Parameter": ((1610874909, LCID, 4, 0),()),
		"Priority": ((1610874911, LCID, 4, 0),()),
		"Tag": ((1610874915, LCID, 4, 0),()),
		"TooltipText": ((1610874917, LCID, 4, 0),()),
		"Visible": ((1610874921, LCID, 4, 0),()),
		"Width": ((1610874923, LCID, 4, 0),()),
		"accName": ((-5003, LCID, 4, 0),()),
		"accValue": ((-5004, LCID, 4, 0),()),
	}

class ConnectorFormat(DispatchBaseClass):
	CLSID = IID('{000C0313-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def BeginConnect(self, ConnectedShape=defaultNamedNotOptArg, ConnectionSite=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), ((9, 1), (3, 1)),ConnectedShape
			, ConnectionSite)

	def BeginDisconnect(self):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), (),)

	def EndConnect(self, ConnectedShape=defaultNamedNotOptArg, ConnectionSite=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(12, LCID, 1, (24, 0), ((9, 1), (3, 1)),ConnectedShape
			, ConnectionSite)

	def EndDisconnect(self):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"BeginConnected": (100, 2, (3, 0), (), "BeginConnected", None),
		# Method 'BeginConnectedShape' returns object of type 'Shape'
		"BeginConnectedShape": (101, 2, (9, 0), (), "BeginConnectedShape", '{000C031C-0000-0000-C000-000000000046}'),
		"BeginConnectionSite": (102, 2, (3, 0), (), "BeginConnectionSite", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"EndConnected": (103, 2, (3, 0), (), "EndConnected", None),
		# Method 'EndConnectedShape' returns object of type 'Shape'
		"EndConnectedShape": (104, 2, (9, 0), (), "EndConnectedShape", '{000C031C-0000-0000-C000-000000000046}'),
		"EndConnectionSite": (105, 2, (3, 0), (), "EndConnectionSite", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		"Type": (106, 2, (3, 0), (), "Type", None),
	}
	_prop_map_put_ = {
		"Type": ((106, LCID, 4, 0),()),
	}

class DiagramNode(DispatchBaseClass):
	CLSID = IID('{000C0370-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type DiagramNode
	def AddNode(self, Pos=2, NodeType=1):
		ret = self._oleobj_.InvokeTypes(10, LCID, 1, (9, 0), ((3, 49), (3, 49)),Pos
			, NodeType)
		if ret is not None:
			ret = Dispatch(ret, u'AddNode', '{000C0370-0000-0000-C000-000000000046}')
		return ret

	# Result is of type DiagramNode
	def CloneNode(self, CopyChildren=defaultNamedNotOptArg, TargetNode=defaultNamedNotOptArg, Pos=2):
		ret = self._oleobj_.InvokeTypes(15, LCID, 1, (9, 0), ((11, 1), (9, 1), (3, 49)),CopyChildren
			, TargetNode, Pos)
		if ret is not None:
			ret = Dispatch(ret, u'CloneNode', '{000C0370-0000-0000-C000-000000000046}')
		return ret

	def Delete(self):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), (),)

	def MoveNode(self, TargetNode=defaultNamedNotOptArg, Pos=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(12, LCID, 1, (24, 0), ((9, 1), (3, 1)),TargetNode
			, Pos)

	# Result is of type DiagramNode
	def NextNode(self):
		ret = self._oleobj_.InvokeTypes(17, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'NextNode', '{000C0370-0000-0000-C000-000000000046}')
		return ret

	# Result is of type DiagramNode
	def PrevNode(self):
		ret = self._oleobj_.InvokeTypes(18, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'PrevNode', '{000C0370-0000-0000-C000-000000000046}')
		return ret

	def ReplaceNode(self, TargetNode=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (24, 0), ((9, 1),),TargetNode
			)

	def SwapNode(self, TargetNode=defaultNamedNotOptArg, SwapChildren=True):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (24, 0), ((9, 1), (11, 49)),TargetNode
			, SwapChildren)

	def TransferChildren(self, ReceivingNode=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(16, LCID, 1, (24, 0), ((9, 1),),ReceivingNode
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		# Method 'Children' returns object of type 'DiagramNodeChildren'
		"Children": (101, 2, (9, 0), (), "Children", '{000C036F-0000-0000-C000-000000000046}'),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'Diagram' returns object of type 'IMsoDiagram'
		"Diagram": (104, 2, (9, 0), (), "Diagram", '{000C036D-0000-0000-C000-000000000046}'),
		"Layout": (105, 2, (3, 0), (), "Layout", None),
		"Parent": (100, 2, (9, 0), (), "Parent", None),
		# Method 'Root' returns object of type 'DiagramNode'
		"Root": (103, 2, (9, 0), (), "Root", '{000C0370-0000-0000-C000-000000000046}'),
		# Method 'Shape' returns object of type 'Shape'
		"Shape": (102, 2, (9, 0), (), "Shape", '{000C031C-0000-0000-C000-000000000046}'),
		# Method 'TextShape' returns object of type 'Shape'
		"TextShape": (106, 2, (9, 0), (), "TextShape", '{000C031C-0000-0000-C000-000000000046}'),
	}
	_prop_map_put_ = {
		"Layout": ((105, LCID, 4, 0),()),
	}

class DiagramNodeChildren(DispatchBaseClass):
	CLSID = IID('{000C036F-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type DiagramNode
	def AddNode(self, Index=-1, NodeType=1):
		ret = self._oleobj_.InvokeTypes(10, LCID, 1, (9, 0), ((12, 49), (3, 49)),Index
			, NodeType)
		if ret is not None:
			ret = Dispatch(ret, u'AddNode', '{000C0370-0000-0000-C000-000000000046}')
		return ret

	# Result is of type DiagramNode
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0370-0000-0000-C000-000000000046}')
		return ret

	def SelectAll(self):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (101, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'FirstChild' returns object of type 'DiagramNode'
		"FirstChild": (103, 2, (9, 0), (), "FirstChild", '{000C0370-0000-0000-C000-000000000046}'),
		# Method 'LastChild' returns object of type 'DiagramNode'
		"LastChild": (104, 2, (9, 0), (), "LastChild", '{000C0370-0000-0000-C000-000000000046}'),
		"Parent": (100, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0370-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0370-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0370-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(101, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class DiagramNodes(DispatchBaseClass):
	CLSID = IID('{000C036E-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type DiagramNode
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0370-0000-0000-C000-000000000046}')
		return ret

	def SelectAll(self):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (101, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (100, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0370-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0370-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0370-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(101, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class DocumentLibraryVersion(DispatchBaseClass):
	CLSID = IID('{000C0387-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Delete(self):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (24, 0), (),)

	def Open(self):
		ret = self._oleobj_.InvokeTypes(6, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'Open', None)
		return ret

	def Restore(self):
		ret = self._oleobj_.InvokeTypes(7, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'Restore', None)
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Comments": (4, 2, (8, 0), (), "Comments", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Index": (1, 2, (3, 0), (), "Index", None),
		"Modified": (0, 2, (12, 0), (), "Modified", None),
		"ModifiedBy": (3, 2, (8, 0), (), "ModifiedBy", None),
		"Parent": (2, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default property for this class is 'Modified'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (12, 0), (), "Modified", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class DocumentLibraryVersions(DispatchBaseClass):
	CLSID = IID('{000C0388-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type DocumentLibraryVersion
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, lIndex=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),lIndex
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0387-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"IsVersioningEnabled": (3, 2, (11, 0), (), "IsVersioningEnabled", None),
		"Parent": (2, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, lIndex=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),lIndex
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0387-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0387-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0387-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class DocumentProperties(DispatchBaseClass):
	CLSID = IID('{2DF8D04D-5BFA-101B-BDE5-00AA0044DE52}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743813, 2, (3, 0), ((16393, 10),), "Application", None),
		"Count": (4, 2, (3, 0), ((16387, 10),), "Count", None),
		"Creator": (1610743814, 2, (3, 0), ((16387, 10),), "Creator", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, None)
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),None)
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(4, 2, (3, 0), ((16387, 10),), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class DocumentProperty(DispatchBaseClass):
	CLSID = IID('{2DF8D04E-5BFA-101B-BDE5-00AA0044DE52}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743820, 2, (3, 0), ((16393, 10),), "Application", None),
		"Creator": (1610743821, 2, (3, 0), ((16387, 10),), "Creator", None),
		"LinkSource": (7, 2, (3, 0), ((16392, 10),), "LinkSource", None),
		"LinkToContent": (6, 2, (3, 0), ((16395, 10),), "LinkToContent", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
		"LinkSource": ((7, LCID, 4, 0),()),
		"LinkToContent": ((6, LCID, 4, 0),()),
	}

class FileDialog(DispatchBaseClass):
	CLSID = IID('{000C0362-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Execute(self):
		return self._oleobj_.InvokeTypes(1610809362, LCID, 1, (24, 0), (),)

	def Show(self):
		return self._oleobj_.InvokeTypes(1610809361, LCID, 1, (3, 0), (),)

	_prop_map_get_ = {
		"AllowMultiSelect": (1610809352, 2, (11, 0), (), "AllowMultiSelect", None),
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"ButtonName": (1610809350, 2, (8, 0), (), "ButtonName", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"DialogType": (1610809359, 2, (3, 0), (), "DialogType", None),
		"FilterIndex": (1610809346, 2, (3, 0), (), "FilterIndex", None),
		# Method 'Filters' returns object of type 'FileDialogFilters'
		"Filters": (1610809345, 2, (9, 0), (), "Filters", '{000C0365-0000-0000-C000-000000000046}'),
		"InitialFileName": (1610809356, 2, (8, 0), (), "InitialFileName", None),
		"InitialView": (1610809354, 2, (3, 0), (), "InitialView", None),
		"Item": (0, 2, (8, 0), (), "Item", None),
		"Parent": (1610809344, 2, (9, 0), (), "Parent", None),
		# Method 'SelectedItems' returns object of type 'FileDialogSelectedItems'
		"SelectedItems": (1610809358, 2, (9, 0), (), "SelectedItems", '{000C0363-0000-0000-C000-000000000046}'),
		"Title": (1610809348, 2, (8, 0), (), "Title", None),
	}
	_prop_map_put_ = {
		"AllowMultiSelect": ((1610809352, LCID, 4, 0),()),
		"ButtonName": ((1610809350, LCID, 4, 0),()),
		"FilterIndex": ((1610809346, LCID, 4, 0),()),
		"InitialFileName": ((1610809356, LCID, 4, 0),()),
		"InitialView": ((1610809354, LCID, 4, 0),()),
		"Title": ((1610809348, LCID, 4, 0),()),
	}
	# Default property for this class is 'Item'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "Item", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class FileDialogFilter(DispatchBaseClass):
	CLSID = IID('{000C0364-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Description": (1610809346, 2, (8, 0), (), "Description", None),
		"Extensions": (1610809345, 2, (8, 0), (), "Extensions", None),
		"Parent": (1610809344, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}

class FileDialogFilters(DispatchBaseClass):
	CLSID = IID('{000C0365-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type FileDialogFilter
	def Add(self, Description=defaultNamedNotOptArg, Extensions=defaultNamedNotOptArg, Position=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610809350, LCID, 1, (9, 0), ((8, 1), (8, 1), (12, 17)),Description
			, Extensions, Position)
		if ret is not None:
			ret = Dispatch(ret, u'Add', '{000C0364-0000-0000-C000-000000000046}')
		return ret

	def Clear(self):
		return self._oleobj_.InvokeTypes(1610809349, LCID, 1, (24, 0), (),)

	def Delete(self, filter=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1610809348, LCID, 1, (24, 0), ((12, 17),),filter
			)

	# Result is of type FileDialogFilter
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0364-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1610809346, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (1610809344, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0364-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0364-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0364-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1610809346, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class FileDialogSelectedItems(DispatchBaseClass):
	CLSID = IID('{000C0363-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Item(self, Index=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(0, LCID, 1, (8, 0), ((3, 1),),Index
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1610809346, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (1610809344, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(0, LCID, 1, (8, 0), ((3, 1),),Index
			)

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, None)
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),None)
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1610809346, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class FileSearch(DispatchBaseClass):
	CLSID = IID('{000C0332-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Execute(self, SortBy=1, SortOrder=1, AlwaysAccurate=True):
		return self._oleobj_.InvokeTypes(9, LCID, 1, (3, 0), ((3, 49), (3, 49), (11, 49)),SortBy
			, SortOrder, AlwaysAccurate)

	def NewSearch(self):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), (),)

	def RefreshScopes(self):
		return self._oleobj_.InvokeTypes(17, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"FileName": (4, 2, (8, 0), (), "FileName", None),
		"FileType": (5, 2, (3, 0), (), "FileType", None),
		# Method 'FileTypes' returns object of type 'FileTypes'
		"FileTypes": (16, 2, (9, 0), (), "FileTypes", '{000C036C-0000-0000-C000-000000000046}'),
		# Method 'FoundFiles' returns object of type 'FoundFiles'
		"FoundFiles": (11, 2, (9, 0), (), "FoundFiles", '{000C0331-0000-0000-C000-000000000046}'),
		"LastModified": (6, 2, (3, 0), (), "LastModified", None),
		"LookIn": (8, 2, (8, 0), (), "LookIn", None),
		"MatchAllWordForms": (3, 2, (11, 0), (), "MatchAllWordForms", None),
		"MatchTextExactly": (2, 2, (11, 0), (), "MatchTextExactly", None),
		# Method 'PropertyTests' returns object of type 'PropertyTests'
		"PropertyTests": (12, 2, (9, 0), (), "PropertyTests", '{000C0334-0000-0000-C000-000000000046}'),
		# Method 'SearchFolders' returns object of type 'SearchFolders'
		"SearchFolders": (14, 2, (9, 0), (), "SearchFolders", '{000C036A-0000-0000-C000-000000000046}'),
		# Method 'SearchScopes' returns object of type 'SearchScopes'
		"SearchScopes": (13, 2, (9, 0), (), "SearchScopes", '{000C0366-0000-0000-C000-000000000046}'),
		"SearchSubFolders": (1, 2, (11, 0), (), "SearchSubFolders", None),
		"TextOrProperty": (7, 2, (8, 0), (), "TextOrProperty", None),
	}
	_prop_map_put_ = {
		"FileName": ((4, LCID, 4, 0),()),
		"FileType": ((5, LCID, 4, 0),()),
		"LastModified": ((6, LCID, 4, 0),()),
		"LookIn": ((8, LCID, 4, 0),()),
		"MatchAllWordForms": ((3, LCID, 4, 0),()),
		"MatchTextExactly": ((2, LCID, 4, 0),()),
		"SearchSubFolders": ((1, LCID, 4, 0),()),
		"TextOrProperty": ((7, LCID, 4, 0),()),
	}

class FileTypes(DispatchBaseClass):
	CLSID = IID('{000C036C-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Add(self, FileType=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(3, LCID, 1, (24, 0), ((3, 1),),FileType
			)

	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(0, LCID, 2, (3, 0), ((3, 1),),Index
			)

	def Remove(self, Index=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (24, 0), ((3, 1),),Index
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (2, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(0, LCID, 2, (3, 0), ((3, 1),),Index
			)

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, None)
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),None)
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(2, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class FillFormat(DispatchBaseClass):
	CLSID = IID('{000C0314-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Background(self):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), (),)

	def OneColorGradient(self, Style=defaultNamedNotOptArg, Variant=defaultNamedNotOptArg, Degree=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), ((3, 1), (3, 1), (4, 1)),Style
			, Variant, Degree)

	def Patterned(self, Pattern=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(12, LCID, 1, (24, 0), ((3, 1),),Pattern
			)

	def PresetGradient(self, Style=defaultNamedNotOptArg, Variant=defaultNamedNotOptArg, PresetGradientType=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (24, 0), ((3, 1), (3, 1), (3, 1)),Style
			, Variant, PresetGradientType)

	def PresetTextured(self, PresetTexture=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (24, 0), ((3, 1),),PresetTexture
			)

	def Solid(self):
		return self._oleobj_.InvokeTypes(15, LCID, 1, (24, 0), (),)

	def TwoColorGradient(self, Style=defaultNamedNotOptArg, Variant=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(16, LCID, 1, (24, 0), ((3, 1), (3, 1)),Style
			, Variant)

	def UserPicture(self, PictureFile=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(17, LCID, 1, (24, 0), ((8, 1),),PictureFile
			)

	def UserTextured(self, TextureFile=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(18, LCID, 1, (24, 0), ((8, 1),),TextureFile
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		# Method 'BackColor' returns object of type 'ColorFormat'
		"BackColor": (100, 2, (9, 0), (), "BackColor", '{000C0312-0000-0000-C000-000000000046}'),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'ForeColor' returns object of type 'ColorFormat'
		"ForeColor": (101, 2, (9, 0), (), "ForeColor", '{000C0312-0000-0000-C000-000000000046}'),
		"GradientColorType": (102, 2, (3, 0), (), "GradientColorType", None),
		"GradientDegree": (103, 2, (4, 0), (), "GradientDegree", None),
		"GradientStyle": (104, 2, (3, 0), (), "GradientStyle", None),
		"GradientVariant": (105, 2, (3, 0), (), "GradientVariant", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		"Pattern": (106, 2, (3, 0), (), "Pattern", None),
		"PresetGradientType": (107, 2, (3, 0), (), "PresetGradientType", None),
		"PresetTexture": (108, 2, (3, 0), (), "PresetTexture", None),
		"TextureName": (109, 2, (8, 0), (), "TextureName", None),
		"TextureType": (110, 2, (3, 0), (), "TextureType", None),
		"Transparency": (111, 2, (4, 0), (), "Transparency", None),
		"Type": (112, 2, (3, 0), (), "Type", None),
		"Visible": (113, 2, (3, 0), (), "Visible", None),
	}
	_prop_map_put_ = {
		"BackColor": ((100, LCID, 4, 0),()),
		"ForeColor": ((101, LCID, 4, 0),()),
		"Transparency": ((111, LCID, 4, 0),()),
		"Visible": ((113, LCID, 4, 0),()),
	}

class FoundFiles(DispatchBaseClass):
	CLSID = IID('{000C0331-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(0, LCID, 2, (8, 0), ((3, 1),),Index
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (4, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(0, LCID, 2, (8, 0), ((3, 1),),Index
			)

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, None)
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),None)
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(4, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class FreeformBuilder(DispatchBaseClass):
	CLSID = IID('{000C0315-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def AddNodes(self, SegmentType=defaultNamedNotOptArg, EditingType=defaultNamedNotOptArg, X1=defaultNamedNotOptArg, Y1=defaultNamedNotOptArg
			, X2=0.0, Y2=0.0, X3=0.0, Y3=0.0):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), ((3, 1), (3, 1), (4, 1), (4, 1), (4, 49), (4, 49), (4, 49), (4, 49)),SegmentType
			, EditingType, X1, Y1, X2, Y2
			, X3, Y3)

	# Result is of type Shape
	def ConvertToShape(self):
		ret = self._oleobj_.InvokeTypes(11, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'ConvertToShape', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}

class GroupShapes(DispatchBaseClass):
	CLSID = IID('{000C0316-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type Shape
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type ShapeRange
	def Range(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(10, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Range', '{000C031D-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (2, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C031C-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C031C-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(2, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class HTMLProject(DispatchBaseClass):
	CLSID = IID('{000C0356-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Open(self, OpenKind=0):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (24, 0), ((3, 49),),OpenKind
			)

	def RefreshDocument(self, Refresh=True):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (24, 0), ((11, 49),),Refresh
			)

	def RefreshProject(self, Refresh=True):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), ((11, 49),),Refresh
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'HTMLProjectItems' returns object of type 'HTMLProjectItems'
		"HTMLProjectItems": (3, 2, (9, 0), (), "HTMLProjectItems", '{000C0357-0000-0000-C000-000000000046}'),
		"Parent": (4, 2, (9, 0), (), "Parent", None),
		"State": (0, 2, (3, 0), (), "State", None),
	}
	_prop_map_put_ = {
	}
	# Default property for this class is 'State'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (3, 0), (), "State", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class HTMLProjectItem(DispatchBaseClass):
	CLSID = IID('{000C0358-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def LoadFromFile(self, FileName=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (24, 0), ((8, 1),),FileName
			)

	def Open(self, OpenKind=0):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (24, 0), ((3, 49),),OpenKind
			)

	def SaveCopyAs(self, FileName=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(7, LCID, 1, (24, 0), ((8, 1),),FileName
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"IsOpen": (4, 2, (11, 0), (), "IsOpen", None),
		"Name": (0, 2, (8, 0), (), "Name", None),
		"Parent": (10, 2, (9, 0), (), "Parent", None),
		"Text": (8, 2, (8, 0), (), "Text", None),
	}
	_prop_map_put_ = {
		"Text": ((8, LCID, 4, 0),()),
	}
	# Default property for this class is 'Name'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "Name", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class HTMLProjectItems(DispatchBaseClass):
	CLSID = IID('{000C0357-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type HTMLProjectItem
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((16396, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0358-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (2, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((16396, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0358-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0358-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0358-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class IAccessible(DispatchBaseClass):
	CLSID = IID('{618736E0-3C3D-11CF-810C-00AA00389B71}')
	coclass_clsid = None

	# The method GetaccDefaultAction is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDefaultAction(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5013, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccDescription is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDescription(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5005, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelp is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelp(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5008, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelpTopic is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelpTopic(self, pszHelpFile=global_Missing, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5009, 2, (3, 0), ((16392, 2), (12, 17)), u'GetaccHelpTopic', None,pszHelpFile
			, varChild)

	# The method GetaccKeyboardShortcut is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccKeyboardShortcut(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5010, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccName(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5003, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccRole is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccRole(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5006, 2, (12, 0), ((12, 17),), u'GetaccRole', None,varChild
			)

	# The method GetaccState is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccState(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5007, 2, (12, 0), ((12, 17),), u'GetaccState', None,varChild
			)

	# The method GetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccValue(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5004, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method SetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccName(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5003, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method SetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccValue(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5004, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method accChild is actually a property, but must be used as a method to correctly pass the arguments
	def accChild(self, varChild=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(-5002, LCID, 2, (9, 0), ((12, 1),),varChild
			)
		if ret is not None:
			ret = Dispatch(ret, u'accChild', None)
		return ret

	def accDoDefaultAction(self, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5018, LCID, 1, (24, 0), ((12, 17),),varChild
			)

	def accHitTest(self, xLeft=defaultNamedNotOptArg, yTop=defaultNamedNotOptArg):
		return self._ApplyTypes_(-5017, 1, (12, 0), ((3, 1), (3, 1)), u'accHitTest', None,xLeft
			, yTop)

	def accLocation(self, pxLeft=global_Missing, pyTop=global_Missing, pcxWidth=global_Missing, pcyHeight=global_Missing
			, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5015, 1, (24, 0), ((16387, 2), (16387, 2), (16387, 2), (16387, 2), (12, 17)), u'accLocation', None,pxLeft
			, pyTop, pcxWidth, pcyHeight, varChild)

	def accNavigate(self, navDir=defaultNamedNotOptArg, varStart=defaultNamedOptArg):
		return self._ApplyTypes_(-5016, 1, (12, 0), ((3, 1), (12, 17)), u'accNavigate', None,navDir
			, varStart)

	def accSelect(self, flagsSelect=defaultNamedNotOptArg, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5014, LCID, 1, (24, 0), ((3, 1), (12, 17)),flagsSelect
			, varChild)

	_prop_map_get_ = {
		"accChildCount": (-5001, 2, (3, 0), (), "accChildCount", None),
		"accDefaultAction": (-5013, 2, (8, 0), ((12, 17),), "accDefaultAction", None),
		"accDescription": (-5005, 2, (8, 0), ((12, 17),), "accDescription", None),
		"accFocus": (-5011, 2, (12, 0), (), "accFocus", None),
		"accHelp": (-5008, 2, (8, 0), ((12, 17),), "accHelp", None),
		"accHelpTopic": (-5009, 2, (3, 0), ((16392, 2), (12, 17)), "accHelpTopic", None),
		"accKeyboardShortcut": (-5010, 2, (8, 0), ((12, 17),), "accKeyboardShortcut", None),
		"accName": (-5003, 2, (8, 0), ((12, 17),), "accName", None),
		"accParent": (-5000, 2, (9, 0), (), "accParent", None),
		"accRole": (-5006, 2, (12, 0), ((12, 17),), "accRole", None),
		"accSelection": (-5012, 2, (12, 0), (), "accSelection", None),
		"accState": (-5007, 2, (12, 0), ((12, 17),), "accState", None),
		"accValue": (-5004, 2, (8, 0), ((12, 17),), "accValue", None),
	}
	_prop_map_put_ = {
		"accName": ((-5003, LCID, 4, 0),()),
		"accValue": ((-5004, LCID, 4, 0),()),
	}

class ICommandBarButtonEvents(DispatchBaseClass):
	CLSID = IID('{55F88890-7708-11D1-ACEB-006008961DA5}')
	coclass_clsid = None

	def Click(self, Ctrl=defaultNamedNotOptArg, CancelDefault=defaultNamedNotOptArg):
		return self._ApplyTypes_(1, 1, (24, 0), ((13, 1), (16395, 3)), u'Click', None,Ctrl
			, CancelDefault)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}

class ICommandBarComboBoxEvents(DispatchBaseClass):
	CLSID = IID('{55F88896-7708-11D1-ACEB-006008961DA5}')
	coclass_clsid = None

	def Change(self, Ctrl=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), ((13, 1),),Ctrl
			)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}

class ICommandBarsEvents(DispatchBaseClass):
	CLSID = IID('{55F88892-7708-11D1-ACEB-006008961DA5}')
	coclass_clsid = None

	def OnUpdate(self):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}

class IFind(DispatchBaseClass):
	CLSID = IID('{000C0337-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Delete(self, bstrQueryName=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1610743853, LCID, 1, (24, 0), ((8, 1),),bstrQueryName
			)

	def Execute(self):
		return self._oleobj_.InvokeTypes(1610743850, LCID, 1, (24, 0), (),)

	def Load(self, bstrQueryName=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1610743851, LCID, 1, (24, 0), ((8, 1),),bstrQueryName
			)

	def Save(self, bstrQueryName=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1610743852, LCID, 1, (24, 0), ((8, 1),),bstrQueryName
			)

	def Show(self):
		return self._oleobj_.InvokeTypes(1610743829, LCID, 1, (3, 0), (),)

	_prop_map_get_ = {
		"Author": (1610743812, 2, (8, 0), (), "Author", None),
		"DateCreatedFrom": (1610743822, 2, (12, 0), (), "DateCreatedFrom", None),
		"DateCreatedTo": (1610743823, 2, (12, 0), (), "DateCreatedTo", None),
		"DateSavedFrom": (1610743819, 2, (12, 0), (), "DateSavedFrom", None),
		"DateSavedTo": (1610743820, 2, (12, 0), (), "DateSavedTo", None),
		"FileType": (1610743854, 2, (3, 0), (), "FileType", None),
		"Keywords": (1610743813, 2, (8, 0), (), "Keywords", None),
		"ListBy": (1610743826, 2, (3, 0), (), "ListBy", None),
		"MatchCase": (1610743816, 2, (11, 0), (), "MatchCase", None),
		"Name": (1610743809, 2, (8, 0), (), "Name", None),
		"Options": (1610743815, 2, (3, 0), (), "Options", None),
		"PatternMatch": (1610743818, 2, (11, 0), (), "PatternMatch", None),
		# Method 'Results' returns object of type 'IFoundFiles'
		"Results": (1610743828, 2, (9, 0), (), "Results", '{000C0338-0000-0000-C000-000000000046}'),
		"SavedBy": (1610743821, 2, (8, 0), (), "SavedBy", None),
		"SearchPath": (0, 2, (8, 0), (), "SearchPath", None),
		"SelectedFile": (1610743827, 2, (3, 0), (), "SelectedFile", None),
		"SortBy": (1610743825, 2, (3, 0), (), "SortBy", None),
		"SubDir": (1610743810, 2, (11, 0), (), "SubDir", None),
		"Subject": (1610743814, 2, (8, 0), (), "Subject", None),
		"Text": (1610743817, 2, (8, 0), (), "Text", None),
		"Title": (1610743811, 2, (8, 0), (), "Title", None),
		"View": (1610743824, 2, (3, 0), (), "View", None),
	}
	_prop_map_put_ = {
		"Author": ((1610743812, LCID, 4, 0),()),
		"DateCreatedFrom": ((1610743822, LCID, 4, 0),()),
		"DateCreatedTo": ((1610743823, LCID, 4, 0),()),
		"DateSavedFrom": ((1610743819, LCID, 4, 0),()),
		"DateSavedTo": ((1610743820, LCID, 4, 0),()),
		"FileType": ((1610743854, LCID, 4, 0),()),
		"Keywords": ((1610743813, LCID, 4, 0),()),
		"ListBy": ((1610743826, LCID, 4, 0),()),
		"MatchCase": ((1610743816, LCID, 4, 0),()),
		"Name": ((1610743809, LCID, 4, 0),()),
		"Options": ((1610743815, LCID, 4, 0),()),
		"PatternMatch": ((1610743818, LCID, 4, 0),()),
		"SavedBy": ((1610743821, LCID, 4, 0),()),
		"SearchPath": ((0, LCID, 4, 0),()),
		"SelectedFile": ((1610743827, LCID, 4, 0),()),
		"SortBy": ((1610743825, LCID, 4, 0),()),
		"SubDir": ((1610743810, LCID, 4, 0),()),
		"Subject": ((1610743814, LCID, 4, 0),()),
		"Text": ((1610743817, LCID, 4, 0),()),
		"Title": ((1610743811, LCID, 4, 0),()),
		"View": ((1610743824, LCID, 4, 0),()),
	}
	# Default property for this class is 'SearchPath'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "SearchPath", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class IFoundFiles(DispatchBaseClass):
	CLSID = IID('{000C0338-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(0, LCID, 2, (8, 0), ((3, 1),),Index
			)

	_prop_map_get_ = {
		"Count": (1610743809, 2, (3, 0), (), "Count", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(0, LCID, 2, (8, 0), ((3, 1),),Index
			)

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, None)
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),None)
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1610743809, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class ILicAgent(DispatchBaseClass):
	"""ILicAgent Interface"""
	CLSID = IID('{00194002-D9C3-11D3-8D59-0050048384E3}')
	coclass_clsid = None

	def AsyncProcessCCRenewalLicenseRequest(self):
		"""method AsyncProcessCCRenewalLicenseRequest"""
		return self._oleobj_.InvokeTypes(88, LCID, 1, (24, 0), (),)

	def AsyncProcessCCRenewalPriceRequest(self):
		"""method AsyncProcessCCRenewalPriceRequest"""
		return self._oleobj_.InvokeTypes(87, LCID, 1, (24, 0), (),)

	def AsyncProcessDroppedLicenseRequest(self):
		"""method AsyncProcessDroppedLicenseRequest"""
		return self._oleobj_.InvokeTypes(93, LCID, 1, (24, 0), (),)

	def AsyncProcessHandshakeRequest(self, bReviseCustInfo=defaultNamedNotOptArg):
		"""method AsyncProcessHandshakeRequest"""
		return self._oleobj_.InvokeTypes(82, LCID, 1, (24, 0), ((3, 1),),bReviseCustInfo
			)

	def AsyncProcessNewLicenseRequest(self):
		"""method AsyncProcessNewLicenseRequest"""
		return self._oleobj_.InvokeTypes(83, LCID, 1, (24, 0), (),)

	def AsyncProcessReissueLicenseRequest(self):
		"""method AsyncProcessReissueLicenseRequest"""
		return self._oleobj_.InvokeTypes(84, LCID, 1, (24, 0), (),)

	def AsyncProcessRetailRenewalLicenseRequest(self):
		"""method AsyncProcessRetailRenewalLicenseRequest"""
		return self._oleobj_.InvokeTypes(85, LCID, 1, (24, 0), (),)

	def AsyncProcessReviseCustInfoRequest(self):
		"""method AsyncProcessReviseCustInfoRequest"""
		return self._oleobj_.InvokeTypes(86, LCID, 1, (24, 0), (),)

	def CancelAsyncProcessRequest(self, bIsLicenseRequest=defaultNamedNotOptArg):
		"""method CancelAsyncProcessRequest"""
		return self._oleobj_.InvokeTypes(98, LCID, 1, (24, 0), ((3, 1),),bIsLicenseRequest
			)

	def CheckSystemClock(self):
		"""method CheckSystemClock"""
		return self._oleobj_.InvokeTypes(40, LCID, 1, (19, 0), (),)

	def DepositConfirmationId(self, bstrVal=defaultNamedNotOptArg):
		"""method DepositConfirmationId"""
		return self._oleobj_.InvokeTypes(95, LCID, 1, (19, 0), ((8, 1),),bstrVal
			)

	def DisplaySSLCert(self):
		"""method DisplaySSLCert"""
		return self._oleobj_.InvokeTypes(109, LCID, 1, (19, 0), (),)

	def GenerateInstallationId(self):
		"""method GenerateInstallationId"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(94, LCID, 1, (8, 0), (),)

	def GetAddress1(self):
		"""method GetAddress1"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(13, LCID, 1, (8, 0), (),)

	def GetAddress2(self):
		"""method GetAddress2"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(38, LCID, 1, (8, 0), (),)

	def GetAsyncProcessReturnCode(self):
		"""method GetAsyncProcessReturnCode"""
		return self._oleobj_.InvokeTypes(90, LCID, 1, (19, 0), (),)

	def GetBackendErrorMsg(self):
		"""method GetBackendErrorMsg"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(105, LCID, 1, (8, 0), (),)

	def GetBillingAddress1(self):
		"""method GetBillingAddress1"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(49, LCID, 1, (8, 0), (),)

	def GetBillingAddress2(self):
		"""method GetBillingAddress2"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(51, LCID, 1, (8, 0), (),)

	def GetBillingCity(self):
		"""method GetBillingCity"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(53, LCID, 1, (8, 0), (),)

	def GetBillingCountryCode(self):
		"""method GetBillingCountryCode"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(57, LCID, 1, (8, 0), (),)

	def GetBillingFirstName(self):
		"""method GetBillingFirstName"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(43, LCID, 1, (8, 0), (),)

	def GetBillingLastName(self):
		"""method GetBillingLastName"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(45, LCID, 1, (8, 0), (),)

	def GetBillingPhone(self):
		"""method GetBillingPhone"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(47, LCID, 1, (8, 0), (),)

	def GetBillingState(self):
		"""method GetBillingState"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(55, LCID, 1, (8, 0), (),)

	def GetBillingZip(self):
		"""method GetBillingZip"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(59, LCID, 1, (8, 0), (),)

	def GetCCRenewalExpiryDate(self):
		"""method GetCCRenewalExpiryDate"""
		return self._oleobj_.InvokeTypes(66, LCID, 1, (7, 0), (),)

	def GetCity(self):
		"""method GetCity"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(15, LCID, 1, (8, 0), (),)

	def GetCountryCode(self):
		"""method GetCountryCode"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(19, LCID, 1, (8, 0), (),)

	def GetCountryDesc(self):
		"""method GetCountryDesc"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(21, LCID, 1, (8, 0), (),)

	def GetCreditCardCode(self, dwIndex=defaultNamedNotOptArg):
		"""method GetCreditCardCode"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(73, LCID, 1, (8, 0), ((19, 1),),dwIndex
			)

	def GetCreditCardCount(self):
		"""method GetCreditCardCount"""
		return self._oleobj_.InvokeTypes(72, LCID, 1, (19, 0), (),)

	def GetCreditCardExpiryMonth(self):
		"""method GetCreditCardExpiryMonth"""
		return self._oleobj_.InvokeTypes(79, LCID, 1, (19, 0), (),)

	def GetCreditCardExpiryYear(self):
		"""method GetCreditCardExpiryYear"""
		return self._oleobj_.InvokeTypes(78, LCID, 1, (19, 0), (),)

	def GetCreditCardName(self, dwIndex=defaultNamedNotOptArg):
		"""method GetCreditCardName"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(74, LCID, 1, (8, 0), ((19, 1),),dwIndex
			)

	def GetCreditCardNumber(self):
		"""method GetCreditCardNumber"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(77, LCID, 1, (8, 0), (),)

	def GetCreditCardType(self):
		"""method GetCreditCardType"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(76, LCID, 1, (8, 0), (),)

	def GetCurrencyDescription(self, dwCurrencyIndex=defaultNamedNotOptArg):
		"""method GetCurrencyDescription"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(100, LCID, 1, (8, 0), ((19, 1),),dwCurrencyIndex
			)

	def GetCurrencyOption(self):
		"""method GetCurrencyOption"""
		return self._oleobj_.InvokeTypes(106, LCID, 1, (19, 0), (),)

	def GetCurrentExpiryDate(self):
		"""method GetCurrentExpiryDate"""
		return self._oleobj_.InvokeTypes(97, LCID, 1, (7, 0), (),)

	def GetDisconnectOption(self):
		"""method GetDisconnectOption"""
		return self._oleobj_.InvokeTypes(80, LCID, 1, (3, 0), (),)

	def GetEmail(self):
		"""method GetEmail"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(9, LCID, 1, (8, 0), (),)

	def GetEndOfLifeHtmlText(self):
		"""method GetEndOfLifeHtmlText"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(108, LCID, 1, (8, 0), (),)

	def GetExistingExpiryDate(self):
		"""method GetExistingExpiryDate"""
		return self._oleobj_.InvokeTypes(41, LCID, 1, (7, 0), (),)

	def GetFirstName(self):
		"""method GetFirstName"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(3, LCID, 1, (8, 0), (),)

	def GetInvoiceText(self):
		"""method GetInvoiceText"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(104, LCID, 1, (8, 0), (),)

	def GetIsoLanguage(self):
		"""method GetIsoLanguage"""
		return self._oleobj_.InvokeTypes(25, LCID, 1, (19, 0), (),)

	def GetLastName(self):
		"""method GetLastName"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(5, LCID, 1, (8, 0), (),)

	def GetMSOffer(self):
		"""method GetMSOffer"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(34, LCID, 1, (8, 0), (),)

	def GetMSUpdate(self):
		"""method GetMSUpdate"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(32, LCID, 1, (8, 0), (),)

	def GetNewExpiryDate(self):
		"""method GetNewExpiryDate"""
		return self._oleobj_.InvokeTypes(42, LCID, 1, (7, 0), (),)

	def GetOrgName(self):
		"""method GetOrgName"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(7, LCID, 1, (8, 0), (),)

	def GetOtherOffer(self):
		"""method GetOtherOffer"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(36, LCID, 1, (8, 0), (),)

	def GetPhone(self):
		"""method GetPhone"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(11, LCID, 1, (8, 0), (),)

	def GetPriceItemCount(self):
		"""method GetPriceItemCount"""
		return self._oleobj_.InvokeTypes(101, LCID, 1, (19, 0), (),)

	def GetPriceItemLabel(self, dwIndex=defaultNamedNotOptArg):
		"""method GetPriceItemLabel"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(102, LCID, 1, (8, 0), ((19, 1),),dwIndex
			)

	def GetPriceItemValue(self, dwCurrencyIndex=defaultNamedNotOptArg, dwIndex=defaultNamedNotOptArg):
		"""method GetPriceItemValue"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(103, LCID, 1, (8, 0), ((19, 1), (19, 1)),dwCurrencyIndex
			, dwIndex)

	def GetState(self):
		"""method GetState"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(17, LCID, 1, (8, 0), (),)

	def GetVATLabel(self, bstrCountryCode=defaultNamedNotOptArg):
		"""method GetVATLabel"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(65, LCID, 1, (8, 0), ((8, 1),),bstrCountryCode
			)

	def GetVATNumber(self):
		"""method GetVATNumber"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(75, LCID, 1, (8, 0), (),)

	def GetZip(self):
		"""method GetZip"""
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(23, LCID, 1, (8, 0), (),)

	def Initialize(self, dwBPC=defaultNamedNotOptArg, dwMode=defaultNamedNotOptArg, bstrLicSource=defaultNamedNotOptArg):
		"""method Initialize"""
		return self._oleobj_.InvokeTypes(1, LCID, 1, (19, 0), ((19, 1), (19, 1), (8, 1)),dwBPC
			, dwMode, bstrLicSource)

	def IsCCRenewalCountry(self, bstrCountryCode=defaultNamedNotOptArg):
		"""method IsCCRenewalCountry"""
		return self._oleobj_.InvokeTypes(64, LCID, 1, (3, 0), ((8, 1),),bstrCountryCode
			)

	def IsUpgradeAvailable(self):
		"""method IsUpgradeAvailable"""
		return self._oleobj_.InvokeTypes(91, LCID, 1, (3, 0), (),)

	def SaveBillingInfo(self, bSave=defaultNamedNotOptArg):
		"""method SaveBillingInfo"""
		return self._oleobj_.InvokeTypes(61, LCID, 1, (19, 0), ((3, 1),),bSave
			)

	def SetAddress1(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetAddress1"""
		return self._oleobj_.InvokeTypes(14, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetAddress2(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetAddress2"""
		return self._oleobj_.InvokeTypes(39, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetBillingAddress1(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetBillingAddress1"""
		return self._oleobj_.InvokeTypes(50, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetBillingAddress2(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetBillingAddress2"""
		return self._oleobj_.InvokeTypes(52, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetBillingCity(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetBillingCity"""
		return self._oleobj_.InvokeTypes(54, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetBillingCountryCode(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetBillingCountryCode"""
		return self._oleobj_.InvokeTypes(58, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetBillingFirstName(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetBillingFirstName"""
		return self._oleobj_.InvokeTypes(44, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetBillingLastName(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetBillingLastName"""
		return self._oleobj_.InvokeTypes(46, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetBillingPhone(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetBillingPhone"""
		return self._oleobj_.InvokeTypes(48, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetBillingState(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetBillingState"""
		return self._oleobj_.InvokeTypes(56, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetBillingZip(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetBillingZip"""
		return self._oleobj_.InvokeTypes(60, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetCity(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetCity"""
		return self._oleobj_.InvokeTypes(16, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetCountryCode(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetCountryCode"""
		return self._oleobj_.InvokeTypes(20, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetCountryDesc(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetCountryDesc"""
		return self._oleobj_.InvokeTypes(22, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetCreditCardExpiryMonth(self, dwCCMonth=defaultNamedNotOptArg):
		"""method SetCreditCardExpiryMonth"""
		return self._oleobj_.InvokeTypes(71, LCID, 1, (24, 0), ((19, 1),),dwCCMonth
			)

	def SetCreditCardExpiryYear(self, dwCCYear=defaultNamedNotOptArg):
		"""method SetCreditCardExpiryYear"""
		return self._oleobj_.InvokeTypes(70, LCID, 1, (24, 0), ((19, 1),),dwCCYear
			)

	def SetCreditCardNumber(self, bstrCCNumber=defaultNamedNotOptArg):
		"""method SetCreditCardNumber"""
		return self._oleobj_.InvokeTypes(69, LCID, 1, (24, 0), ((8, 1),),bstrCCNumber
			)

	def SetCreditCardType(self, bstrCCCode=defaultNamedNotOptArg):
		"""method SetCreditCardType"""
		return self._oleobj_.InvokeTypes(68, LCID, 1, (24, 0), ((8, 1),),bstrCCCode
			)

	def SetCurrencyOption(self, dwCurrencyOption=defaultNamedNotOptArg):
		"""method SetCurrencyOption"""
		return self._oleobj_.InvokeTypes(107, LCID, 1, (24, 0), ((19, 1),),dwCurrencyOption
			)

	def SetDisconnectOption(self, bNewVal=defaultNamedNotOptArg):
		"""method SetDisconnectOption"""
		return self._oleobj_.InvokeTypes(81, LCID, 1, (24, 0), ((3, 1),),bNewVal
			)

	def SetEmail(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetEmail"""
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetFirstName(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetFirstName"""
		return self._oleobj_.InvokeTypes(4, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetIsoLanguage(self, dwNewVal=defaultNamedNotOptArg):
		"""method SetIsoLanguage"""
		return self._oleobj_.InvokeTypes(26, LCID, 1, (24, 0), ((19, 1),),dwNewVal
			)

	def SetLastName(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetLastName"""
		return self._oleobj_.InvokeTypes(6, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetMSOffer(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetMSOffer"""
		return self._oleobj_.InvokeTypes(35, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetMSUpdate(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetMSUpdate"""
		return self._oleobj_.InvokeTypes(33, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetOrgName(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetOrgName"""
		return self._oleobj_.InvokeTypes(8, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetOtherOffer(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetOtherOffer"""
		return self._oleobj_.InvokeTypes(37, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetPhone(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetPhone"""
		return self._oleobj_.InvokeTypes(12, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetState(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetState"""
		return self._oleobj_.InvokeTypes(18, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def SetVATNumber(self, bstrVATNumber=defaultNamedNotOptArg):
		"""method SetVATNumber"""
		return self._oleobj_.InvokeTypes(67, LCID, 1, (24, 0), ((8, 1),),bstrVATNumber
			)

	def SetZip(self, bstrNewVal=defaultNamedNotOptArg):
		"""method SetZip"""
		return self._oleobj_.InvokeTypes(24, LCID, 1, (24, 0), ((8, 1),),bstrNewVal
			)

	def VerifyCheckDigits(self, bstrCIDIID=defaultNamedNotOptArg):
		"""method VerifyCheckDigits"""
		return self._oleobj_.InvokeTypes(96, LCID, 1, (3, 0), ((8, 1),),bstrCIDIID
			)

	def WantUpgrade(self, bWantUpgrade=defaultNamedNotOptArg):
		"""method WantUpgrade"""
		return self._oleobj_.InvokeTypes(92, LCID, 1, (24, 0), ((3, 1),),bWantUpgrade
			)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}

class ILicValidator(DispatchBaseClass):
	CLSID = IID('{919AA22C-B9AD-11D3-8D59-0050048384E3}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Products": (1, 2, (12, 0), (), "Products", None),
		"Selection": (2, 2, (3, 0), (), "Selection", None),
	}
	_prop_map_put_ = {
		"Selection": ((2, LCID, 4, 0),()),
	}

class ILicWizExternal(DispatchBaseClass):
	CLSID = IID('{4CAC6328-B9B0-11D3-8D59-0050048384E3}')
	coclass_clsid = None

	def DepositPidKey(self, bstrKey=defaultNamedNotOptArg, fMORW=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (3, 0), ((8, 1), (3, 1)),bstrKey
			, fMORW)

	def DisableVORWReminder(self, BPC=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (24, 0), ((3, 1),),BPC
			)

	def FormatDate(self, date=defaultNamedNotOptArg, pFormat=u''):
		return self._ApplyTypes_(3, 1, (8, 32), ((7, 1), (8, 49)), u'FormatDate', None,date
			, pFormat)

	def GetConnectedState(self):
		return self._oleobj_.InvokeTypes(18, LCID, 1, (3, 0), (),)

	def InternetDisconnect(self):
		return self._oleobj_.InvokeTypes(17, LCID, 1, (24, 0), (),)

	def InvokeDateTimeApplet(self):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (24, 0), (),)

	def MsoAlert(self, bstrText=defaultNamedNotOptArg, bstrButtons=defaultNamedNotOptArg, bstrIcon=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(9, LCID, 1, (3, 0), ((8, 1), (8, 1), (8, 1)),bstrText
			, bstrButtons, bstrIcon)

	def OpenInDefaultBrowser(self, bstrUrl=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (24, 0), ((8, 1),),bstrUrl
			)

	def PrintHtmlDocument(self, punkHtmlDoc=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), ((13, 1),),punkHtmlDoc
			)

	def ResetPID(self):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (24, 0), (),)

	def ResignDpc(self, bstrProductCode=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(12, LCID, 1, (24, 0), ((8, 1),),bstrProductCode
			)

	def SaveReceipt(self, bstrReceipt=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(7, LCID, 1, (8, 0), ((8, 1),),bstrReceipt
			)

	def SetDialogSize(self, dx=defaultNamedNotOptArg, dy=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (24, 0), ((3, 1), (3, 1)),dx
			, dy)

	def ShowHelp(self, pvarId=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (24, 0), ((16396, 17),),pvarId
			)

	def SortSelectOptions(self, pdispSelect=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(16, LCID, 1, (24, 0), ((9, 1),),pdispSelect
			)

	def Terminate(self):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (24, 0), (),)

	def VerifyClock(self, lMode=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(15, LCID, 1, (3, 0), ((3, 1),),lMode
			)

	def WriteLog(self, bstrMessage=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), ((8, 1),),bstrMessage
			)

	_prop_map_get_ = {
		"AnimationEnabled": (26, 2, (3, 0), (), "AnimationEnabled", None),
		"Context": (20, 2, (3, 0), (), "Context", None),
		"CountryInfo": (23, 2, (8, 0), (), "CountryInfo", None),
		"LicAgent": (22, 2, (9, 0), (), "LicAgent", None),
		"OfficeOnTheWebUrl": (28, 2, (8, 0), (), "OfficeOnTheWebUrl", None),
		"Validator": (21, 2, (9, 0), (), "Validator", None),
	}
	_prop_map_put_ = {
		"CurrentHelpId": ((27, LCID, 4, 0),()),
		"WizardTitle": ((25, LCID, 4, 0),()),
		"WizardVisible": ((24, LCID, 4, 0),()),
	}

class IMsoDiagram(DispatchBaseClass):
	CLSID = IID('{000C036D-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Convert(self, Type=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), ((3, 1),),Type
			)

	def FitText(self):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"AutoFormat": (105, 2, (3, 0), (), "AutoFormat", None),
		"AutoLayout": (103, 2, (3, 0), (), "AutoLayout", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'Nodes' returns object of type 'DiagramNodes'
		"Nodes": (101, 2, (9, 0), (), "Nodes", '{000C036E-0000-0000-C000-000000000046}'),
		"Parent": (100, 2, (9, 0), (), "Parent", None),
		"Reverse": (104, 2, (3, 0), (), "Reverse", None),
		"Type": (102, 2, (3, 0), (), "Type", None),
	}
	_prop_map_put_ = {
		"AutoFormat": ((105, LCID, 4, 0),()),
		"AutoLayout": ((103, LCID, 4, 0),()),
		"Reverse": ((104, LCID, 4, 0),()),
	}

class IMsoDispCagNotifySink(DispatchBaseClass):
	CLSID = IID('{000C0359-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def InsertClip(self, pClipMoniker=defaultNamedNotOptArg, pItemMoniker=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), ((13, 1), (13, 1)),pClipMoniker
			, pItemMoniker)

	def WindowIsClosing(self):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}

class IMsoEServicesDialog(DispatchBaseClass):
	CLSID = IID('{000C0372-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def AddTrustedDomain(self, Domain=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1610743809, LCID, 1, (24, 0), ((8, 1),),Domain
			)

	def Close(self, ApplyWebComponentChanges=False):
		return self._oleobj_.InvokeTypes(1610743808, LCID, 1, (24, 0), ((11, 49),),ApplyWebComponentChanges
			)

	_prop_map_get_ = {
		"Application": (1610743811, 2, (9, 0), (), "Application", None),
		"ApplicationName": (1610743810, 2, (8, 0), (), "ApplicationName", None),
		"ClipArt": (1610743813, 2, (9, 0), (), "ClipArt", None),
		"WebComponent": (1610743812, 2, (9, 0), (), "WebComponent", None),
	}
	_prop_map_put_ = {
	}

class IMsoEnvelopeVB(DispatchBaseClass):
	CLSID = IID('{000672AC-0000-0000-C000-000000000046}')
	coclass_clsid = IID('{0006F01A-0000-0000-C000-000000000046}')

	_prop_map_get_ = {
		"CommandBars": (4, 2, (9, 0), (), "CommandBars", None),
		"Introduction": (1, 2, (8, 0), (), "Introduction", None),
		"Item": (2, 2, (9, 0), (), "Item", None),
		"Parent": (3, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
		"Introduction": ((1, LCID, 4, 0),()),
	}
	#This class has Item property/method which may take args - allow indexed access
	def __getitem__(self, item):
		return self._get_good_object_(self._oleobj_.Invoke(*(2, LCID, 2, 1, item)), "Item")

class IMsoEnvelopeVBEvents:
	CLSID = CLSID_Sink = IID('{000672AD-0000-0000-C000-000000000046}')
	coclass_clsid = IID('{0006F01A-0000-0000-C000-000000000046}')
	_public_methods_ = [] # For COM Server support
	_dispid_to_func_ = {
		        2 : "OnEnvelopeHide",
		        1 : "OnEnvelopeShow",
		}

	def __init__(self, oobj = None):
		if oobj is None:
			self._olecp = None
		else:
			import win32com.server.util
			from win32com.server.policy import EventHandlerPolicy
			cpc=oobj._oleobj_.QueryInterface(global_IID_IConnectionPointContainer)
			cp=cpc.FindConnectionPoint(self.CLSID_Sink)
			cookie=cp.Advise(win32com.server.util.wrap(self, usePolicy=EventHandlerPolicy))
			self._olecp,self._olecp_cookie = cp,cookie
	def __del__(self):
		try:
			self.close()
		except global_com_error:
			pass
	def close(self):
		if self._olecp is not None:
			cp,cookie,self._olecp,self._olecp_cookie = self._olecp,self._olecp_cookie,None,None
			cp.Unadvise(cookie)
	def _query_interface_(self, iid):
		import win32com.server.util
		if iid==self.CLSID_Sink: return win32com.server.util.wrap(self)

	# Event Handlers
	# If you create handlers, they should have the following prototypes:
#	def OnEnvelopeHide(self):
#	def OnEnvelopeShow(self):


class LanguageSettings(DispatchBaseClass):
	CLSID = IID('{000C0353-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# The method LanguageID is actually a property, but must be used as a method to correctly pass the arguments
	def LanguageID(self, Id=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1, LCID, 2, (3, 0), ((3, 1),),Id
			)

	# The method LanguagePreferredForEditing is actually a property, but must be used as a method to correctly pass the arguments
	def LanguagePreferredForEditing(self, lid=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(2, LCID, 2, (11, 0), ((3, 1),),lid
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (3, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}

class LineFormat(DispatchBaseClass):
	CLSID = IID('{000C0317-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		# Method 'BackColor' returns object of type 'ColorFormat'
		"BackColor": (100, 2, (9, 0), (), "BackColor", '{000C0312-0000-0000-C000-000000000046}'),
		"BeginArrowheadLength": (101, 2, (3, 0), (), "BeginArrowheadLength", None),
		"BeginArrowheadStyle": (102, 2, (3, 0), (), "BeginArrowheadStyle", None),
		"BeginArrowheadWidth": (103, 2, (3, 0), (), "BeginArrowheadWidth", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"DashStyle": (104, 2, (3, 0), (), "DashStyle", None),
		"EndArrowheadLength": (105, 2, (3, 0), (), "EndArrowheadLength", None),
		"EndArrowheadStyle": (106, 2, (3, 0), (), "EndArrowheadStyle", None),
		"EndArrowheadWidth": (107, 2, (3, 0), (), "EndArrowheadWidth", None),
		# Method 'ForeColor' returns object of type 'ColorFormat'
		"ForeColor": (108, 2, (9, 0), (), "ForeColor", '{000C0312-0000-0000-C000-000000000046}'),
		"InsetPen": (114, 2, (3, 0), (), "InsetPen", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		"Pattern": (109, 2, (3, 0), (), "Pattern", None),
		"Style": (110, 2, (3, 0), (), "Style", None),
		"Transparency": (111, 2, (4, 0), (), "Transparency", None),
		"Visible": (112, 2, (3, 0), (), "Visible", None),
		"Weight": (113, 2, (4, 0), (), "Weight", None),
	}
	_prop_map_put_ = {
		"BackColor": ((100, LCID, 4, 0),()),
		"BeginArrowheadLength": ((101, LCID, 4, 0),()),
		"BeginArrowheadStyle": ((102, LCID, 4, 0),()),
		"BeginArrowheadWidth": ((103, LCID, 4, 0),()),
		"DashStyle": ((104, LCID, 4, 0),()),
		"EndArrowheadLength": ((105, LCID, 4, 0),()),
		"EndArrowheadStyle": ((106, LCID, 4, 0),()),
		"EndArrowheadWidth": ((107, LCID, 4, 0),()),
		"ForeColor": ((108, LCID, 4, 0),()),
		"InsetPen": ((114, LCID, 4, 0),()),
		"Pattern": ((109, LCID, 4, 0),()),
		"Style": ((110, LCID, 4, 0),()),
		"Transparency": ((111, LCID, 4, 0),()),
		"Visible": ((112, LCID, 4, 0),()),
		"Weight": ((113, LCID, 4, 0),()),
	}

class MsoDebugOptions(DispatchBaseClass):
	CLSID = IID('{000C035A-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"FeatureReports": (4, 2, (3, 0), (), "FeatureReports", None),
		"OutputToDebugger": (5, 2, (11, 0), (), "OutputToDebugger", None),
		"OutputToFile": (6, 2, (11, 0), (), "OutputToFile", None),
		"OutputToMessageBox": (7, 2, (11, 0), (), "OutputToMessageBox", None),
	}
	_prop_map_put_ = {
		"FeatureReports": ((4, LCID, 4, 0),()),
		"OutputToDebugger": ((5, LCID, 4, 0),()),
		"OutputToFile": ((6, LCID, 4, 0),()),
		"OutputToMessageBox": ((7, LCID, 4, 0),()),
	}

class NewFile(DispatchBaseClass):
	CLSID = IID('{000C0936-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Add(self, FileName=defaultNamedNotOptArg, Section=defaultNamedOptArg, DisplayName=defaultNamedOptArg, Action=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (11, 0), ((8, 1), (12, 17), (12, 17), (12, 17)),FileName
			, Section, DisplayName, Action)

	def Remove(self, FileName=defaultNamedNotOptArg, Section=defaultNamedOptArg, DisplayName=defaultNamedOptArg, Action=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (11, 0), ((8, 1), (12, 17), (12, 17), (12, 17)),FileName
			, Section, DisplayName, Action)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
	}
	_prop_map_put_ = {
	}

class ODSOColumn(DispatchBaseClass):
	CLSID = IID('{000C1531-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Index": (1, 2, (3, 0), (), "Index", None),
		"Name": (2, 2, (8, 0), (), "Name", None),
		"Parent": (3, 2, (9, 0), (), "Parent", None),
		"Value": (4, 2, (8, 0), (), "Value", None),
	}
	_prop_map_put_ = {
	}
	# Default property for this class is 'Value'
	def __call__(self):
		return self._ApplyTypes_(*(4, 2, (8, 0), (), "Value", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class ODSOColumns(DispatchBaseClass):
	CLSID = IID('{000C1532-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Item(self, varIndex=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(1610809346, LCID, 1, (9, 0), ((12, 1),),varIndex
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', None)
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (2, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	#This class has Item property/method which may take args - allow indexed access
	def __getitem__(self, item):
		return self._get_good_object_(self._oleobj_.Invoke(*(1610809346, LCID, 1, 1, item)), "Item")
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class ODSOFilter(DispatchBaseClass):
	CLSID = IID('{000C1533-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Column": (3, 2, (8, 0), (), "Column", None),
		"CompareTo": (5, 2, (8, 0), (), "CompareTo", None),
		"Comparison": (4, 2, (3, 0), (), "Comparison", None),
		"Conjunction": (6, 2, (3, 0), (), "Conjunction", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Index": (1, 2, (3, 0), (), "Index", None),
		"Parent": (2, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
		"Column": ((3, LCID, 4, 0),()),
		"CompareTo": ((5, LCID, 4, 0),()),
		"Comparison": ((4, LCID, 4, 0),()),
		"Conjunction": ((6, LCID, 4, 0),()),
	}

class ODSOFilters(DispatchBaseClass):
	CLSID = IID('{000C1534-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Add(self, Column=defaultNamedNotOptArg, Comparison=defaultNamedNotOptArg, Conjunction=defaultNamedNotOptArg, bstrCompareTo=u''
			, DeferUpdate=False):
		return self._ApplyTypes_(1610809347, 1, (24, 32), ((8, 1), (3, 1), (3, 1), (8, 49), (11, 49)), u'Add', None,Column
			, Comparison, Conjunction, bstrCompareTo, DeferUpdate)

	def Delete(self, Index=defaultNamedNotOptArg, DeferUpdate=False):
		return self._oleobj_.InvokeTypes(1610809348, LCID, 1, (24, 0), ((3, 1), (11, 49)),Index
			, DeferUpdate)

	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(1610809346, LCID, 1, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', None)
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (2, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	#This class has Item property/method which may take args - allow indexed access
	def __getitem__(self, item):
		return self._get_good_object_(self._oleobj_.Invoke(*(1610809346, LCID, 1, 1, item)), "Item")
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class OfficeDataSourceObject(DispatchBaseClass):
	CLSID = IID('{000C1530-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def ApplyFilter(self):
		return self._oleobj_.InvokeTypes(1610743820, LCID, 1, (24, 0), (),)

	def Move(self, MsoMoveRow=defaultNamedNotOptArg, RowNbr=1):
		return self._oleobj_.InvokeTypes(1610743817, LCID, 1, (3, 0), ((3, 1), (3, 49)),MsoMoveRow
			, RowNbr)

	def Open(self, bstrSrc=u'', bstrConnect=u'', bstrTable=u'', fOpenExclusive=0
			, fNeverPrompt=1):
		return self._ApplyTypes_(1610743818, 1, (24, 32), ((8, 49), (8, 49), (8, 49), (3, 49), (3, 49)), u'Open', None,bstrSrc
			, bstrConnect, bstrTable, fOpenExclusive, fNeverPrompt)

	def SetSortOrder(self, SortField1=defaultNamedNotOptArg, SortAscending1=True, SortField2=u'', SortAscending2=True
			, SortField3=u'', SortAscending3=True):
		return self._ApplyTypes_(1610743819, 1, (24, 32), ((8, 1), (11, 49), (8, 49), (11, 49), (8, 49), (11, 49)), u'SetSortOrder', None,SortField1
			, SortAscending1, SortField2, SortAscending2, SortField3, SortAscending3
			)

	_prop_map_get_ = {
		"Columns": (4, 2, (9, 0), (), "Columns", None),
		"ConnectString": (1, 2, (8, 0), (), "ConnectString", None),
		"DataSource": (3, 2, (8, 0), (), "DataSource", None),
		"Filters": (6, 2, (9, 0), (), "Filters", None),
		"RowCount": (5, 2, (3, 0), (), "RowCount", None),
		"Table": (2, 2, (8, 0), (), "Table", None),
	}
	_prop_map_put_ = {
		"ConnectString": ((1, LCID, 4, 0),()),
		"DataSource": ((3, LCID, 4, 0),()),
		"Table": ((2, LCID, 4, 0),()),
	}

class Permission(DispatchBaseClass):
	CLSID = IID('{000C0376-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type UserPermission
	def Add(self, UserId=defaultNamedNotOptArg, Permission=defaultNamedOptArg, ExpirationDate=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(3, LCID, 1, (9, 0), ((8, 1), (12, 17), (12, 17)),UserId
			, Permission, ExpirationDate)
		if ret is not None:
			ret = Dispatch(ret, u'Add', '{000C0375-0000-0000-C000-000000000046}')
		return ret

	def ApplyPolicy(self, FileName=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (24, 0), ((8, 1),),FileName
			)

	# Result is of type UserPermission
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0375-0000-0000-C000-000000000046}')
		return ret

	def RemoveAll(self):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"DocumentAuthor": (12, 2, (8, 0), (), "DocumentAuthor", None),
		"EnableTrustedBrowser": (2, 2, (11, 0), (), "EnableTrustedBrowser", None),
		"Enabled": (7, 2, (11, 0), (), "Enabled", None),
		"Parent": (5, 2, (9, 0), (), "Parent", None),
		"PermissionFromPolicy": (13, 2, (11, 0), (), "PermissionFromPolicy", None),
		"PolicyDescription": (10, 2, (8, 0), (), "PolicyDescription", None),
		"PolicyName": (9, 2, (8, 0), (), "PolicyName", None),
		"RequestPermissionURL": (8, 2, (8, 0), (), "RequestPermissionURL", None),
		"StoreLicenses": (11, 2, (11, 0), (), "StoreLicenses", None),
	}
	_prop_map_put_ = {
		"DocumentAuthor": ((12, LCID, 4, 0),()),
		"EnableTrustedBrowser": ((2, LCID, 4, 0),()),
		"Enabled": ((7, LCID, 4, 0),()),
		"RequestPermissionURL": ((8, LCID, 4, 0),()),
		"StoreLicenses": ((11, LCID, 4, 0),()),
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0375-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0375-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0375-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class PictureFormat(DispatchBaseClass):
	CLSID = IID('{000C031A-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def IncrementBrightness(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def IncrementContrast(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Brightness": (100, 2, (4, 0), (), "Brightness", None),
		"ColorType": (101, 2, (3, 0), (), "ColorType", None),
		"Contrast": (102, 2, (4, 0), (), "Contrast", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"CropBottom": (103, 2, (4, 0), (), "CropBottom", None),
		"CropLeft": (104, 2, (4, 0), (), "CropLeft", None),
		"CropRight": (105, 2, (4, 0), (), "CropRight", None),
		"CropTop": (106, 2, (4, 0), (), "CropTop", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		"TransparencyColor": (107, 2, (3, 0), (), "TransparencyColor", None),
		"TransparentBackground": (108, 2, (3, 0), (), "TransparentBackground", None),
	}
	_prop_map_put_ = {
		"Brightness": ((100, LCID, 4, 0),()),
		"ColorType": ((101, LCID, 4, 0),()),
		"Contrast": ((102, LCID, 4, 0),()),
		"CropBottom": ((103, LCID, 4, 0),()),
		"CropLeft": ((104, LCID, 4, 0),()),
		"CropRight": ((105, LCID, 4, 0),()),
		"CropTop": ((106, LCID, 4, 0),()),
		"TransparencyColor": ((107, LCID, 4, 0),()),
		"TransparentBackground": ((108, LCID, 4, 0),()),
	}

class PropertyTest(DispatchBaseClass):
	CLSID = IID('{000C0333-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Condition": (2, 2, (3, 0), (), "Condition", None),
		"Connector": (5, 2, (3, 0), (), "Connector", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Name": (0, 2, (8, 0), (), "Name", None),
		"SecondValue": (4, 2, (12, 0), (), "SecondValue", None),
		"Value": (3, 2, (12, 0), (), "Value", None),
	}
	_prop_map_put_ = {
	}
	# Default property for this class is 'Name'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "Name", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class PropertyTests(DispatchBaseClass):
	CLSID = IID('{000C0334-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Add(self, Name=defaultNamedNotOptArg, Condition=defaultNamedNotOptArg, Value=defaultNamedNotOptArg, SecondValue=defaultNamedNotOptArg
			, Connector=1):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (24, 0), ((8, 1), (3, 1), (12, 17), (12, 17), (3, 49)),Name
			, Condition, Value, SecondValue, Connector)

	# Result is of type PropertyTest
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0333-0000-0000-C000-000000000046}')
		return ret

	def Remove(self, Index=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (24, 0), ((3, 1),),Index
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (4, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0333-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0333-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0333-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(4, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class ScopeFolder(DispatchBaseClass):
	CLSID = IID('{000C0368-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def AddToSearchFolders(self):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Name": (0, 2, (8, 0), (), "Name", None),
		"Path": (2, 2, (8, 0), (), "Path", None),
		# Method 'ScopeFolders' returns object of type 'ScopeFolders'
		"ScopeFolders": (3, 2, (9, 0), (), "ScopeFolders", '{000C0369-0000-0000-C000-000000000046}'),
	}
	_prop_map_put_ = {
	}
	# Default property for this class is 'Name'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "Name", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class ScopeFolders(DispatchBaseClass):
	CLSID = IID('{000C0369-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type ScopeFolder
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0368-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (4, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0368-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0368-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0368-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(4, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class Script(DispatchBaseClass):
	CLSID = IID('{000C0341-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Delete(self):
		return self._oleobj_.InvokeTypes(1610809352, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Extended": (1610809345, 2, (8, 0), (), "Extended", None),
		"Id": (1610809347, 2, (8, 0), (), "Id", None),
		"Language": (1610809349, 2, (3, 0), (), "Language", None),
		"Location": (1610809351, 2, (3, 0), (), "Location", None),
		"Parent": (1610809344, 2, (9, 0), (), "Parent", None),
		"ScriptText": (0, 2, (8, 0), (), "ScriptText", None),
		"Shape": (1610809353, 2, (9, 0), (), "Shape", None),
	}
	_prop_map_put_ = {
		"Extended": ((1610809345, LCID, 4, 0),()),
		"Id": ((1610809347, LCID, 4, 0),()),
		"Language": ((1610809349, LCID, 4, 0),()),
		"ScriptText": ((0, LCID, 4, 0),()),
	}
	# Default property for this class is 'ScriptText'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "ScriptText", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class Scripts(DispatchBaseClass):
	CLSID = IID('{000C0340-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type Script
	def Add(self, Anchor=None, Location=2, Language=2, Id=u''
			, Extended=u'', ScriptText=u''):
		return self._ApplyTypes_(1610809348, 1, (9, 32), ((9, 49), (3, 49), (3, 49), (8, 49), (8, 49), (8, 49)), u'Add', '{000C0341-0000-0000-C000-000000000046}',Anchor
			, Location, Language, Id, Extended, ScriptText
			)

	def Delete(self):
		return self._oleobj_.InvokeTypes(1610809349, LCID, 1, (24, 0), (),)

	# Result is of type Script
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0341-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1610809345, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (1610809344, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0341-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0341-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0341-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1610809345, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class SearchFolders(DispatchBaseClass):
	CLSID = IID('{000C036A-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Add(self, ScopeFolder=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(3, LCID, 1, (24, 0), ((9, 1),),ScopeFolder
			)

	# Result is of type ScopeFolder
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0368-0000-0000-C000-000000000046}')
		return ret

	def Remove(self, Index=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (24, 0), ((3, 1),),Index
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (2, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0368-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0368-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0368-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(2, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class SearchScope(DispatchBaseClass):
	CLSID = IID('{000C0367-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'ScopeFolder' returns object of type 'ScopeFolder'
		"ScopeFolder": (1, 2, (9, 0), (), "ScopeFolder", '{000C0368-0000-0000-C000-000000000046}'),
		"Type": (0, 2, (3, 0), (), "Type", None),
	}
	_prop_map_put_ = {
	}
	# Default property for this class is 'Type'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (3, 0), (), "Type", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class SearchScopes(DispatchBaseClass):
	CLSID = IID('{000C0366-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type SearchScope
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0367-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (4, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0367-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0367-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0367-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(4, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class ShadowFormat(DispatchBaseClass):
	CLSID = IID('{000C031B-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def IncrementOffsetX(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def IncrementOffsetY(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'ForeColor' returns object of type 'ColorFormat'
		"ForeColor": (100, 2, (9, 0), (), "ForeColor", '{000C0312-0000-0000-C000-000000000046}'),
		"Obscured": (101, 2, (3, 0), (), "Obscured", None),
		"OffsetX": (102, 2, (4, 0), (), "OffsetX", None),
		"OffsetY": (103, 2, (4, 0), (), "OffsetY", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		"Transparency": (104, 2, (4, 0), (), "Transparency", None),
		"Type": (105, 2, (3, 0), (), "Type", None),
		"Visible": (106, 2, (3, 0), (), "Visible", None),
	}
	_prop_map_put_ = {
		"ForeColor": ((100, LCID, 4, 0),()),
		"Obscured": ((101, LCID, 4, 0),()),
		"OffsetX": ((102, LCID, 4, 0),()),
		"OffsetY": ((103, LCID, 4, 0),()),
		"Transparency": ((104, LCID, 4, 0),()),
		"Type": ((105, LCID, 4, 0),()),
		"Visible": ((106, LCID, 4, 0),()),
	}

class Shape(DispatchBaseClass):
	CLSID = IID('{000C031C-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Apply(self):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), (),)

	def CanvasCropBottom(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(143, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def CanvasCropLeft(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(140, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def CanvasCropRight(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(142, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def CanvasCropTop(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(141, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def Delete(self):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), (),)

	# Result is of type Shape
	def Duplicate(self):
		ret = self._oleobj_.InvokeTypes(12, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'Duplicate', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	def Flip(self, FlipCmd=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (24, 0), ((3, 1),),FlipCmd
			)

	def IncrementLeft(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def IncrementRotation(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(15, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def IncrementTop(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(16, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def PickUp(self):
		return self._oleobj_.InvokeTypes(17, LCID, 1, (24, 0), (),)

	def RerouteConnections(self):
		return self._oleobj_.InvokeTypes(18, LCID, 1, (24, 0), (),)

	def ScaleHeight(self, Factor=defaultNamedNotOptArg, RelativeToOriginalSize=defaultNamedNotOptArg, fScale=0):
		return self._oleobj_.InvokeTypes(19, LCID, 1, (24, 0), ((4, 1), (3, 1), (3, 49)),Factor
			, RelativeToOriginalSize, fScale)

	def ScaleWidth(self, Factor=defaultNamedNotOptArg, RelativeToOriginalSize=defaultNamedNotOptArg, fScale=0):
		return self._oleobj_.InvokeTypes(20, LCID, 1, (24, 0), ((4, 1), (3, 1), (3, 49)),Factor
			, RelativeToOriginalSize, fScale)

	def Select(self, Replace=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(21, LCID, 1, (24, 0), ((12, 17),),Replace
			)

	def SetShapesDefaultProperties(self):
		return self._oleobj_.InvokeTypes(22, LCID, 1, (24, 0), (),)

	# Result is of type ShapeRange
	def Ungroup(self):
		ret = self._oleobj_.InvokeTypes(23, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'Ungroup', '{000C031D-0000-0000-C000-000000000046}')
		return ret

	def ZOrder(self, ZOrderCmd=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(24, LCID, 1, (24, 0), ((3, 1),),ZOrderCmd
			)

	_prop_map_get_ = {
		# Method 'Adjustments' returns object of type 'Adjustments'
		"Adjustments": (100, 2, (9, 0), (), "Adjustments", '{000C0310-0000-0000-C000-000000000046}'),
		"AlternativeText": (131, 2, (8, 0), (), "AlternativeText", None),
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"AutoShapeType": (101, 2, (3, 0), (), "AutoShapeType", None),
		"BlackWhiteMode": (102, 2, (3, 0), (), "BlackWhiteMode", None),
		# Method 'Callout' returns object of type 'CalloutFormat'
		"Callout": (103, 2, (9, 0), (), "Callout", '{000C0311-0000-0000-C000-000000000046}'),
		# Method 'CanvasItems' returns object of type 'CanvasShapes'
		"CanvasItems": (138, 2, (9, 0), (), "CanvasItems", '{000C0371-0000-0000-C000-000000000046}'),
		"Child": (136, 2, (3, 0), (), "Child", None),
		"ConnectionSiteCount": (104, 2, (3, 0), (), "ConnectionSiteCount", None),
		"Connector": (105, 2, (3, 0), (), "Connector", None),
		# Method 'ConnectorFormat' returns object of type 'ConnectorFormat'
		"ConnectorFormat": (106, 2, (9, 0), (), "ConnectorFormat", '{000C0313-0000-0000-C000-000000000046}'),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'Diagram' returns object of type 'IMsoDiagram'
		"Diagram": (133, 2, (9, 0), (), "Diagram", '{000C036D-0000-0000-C000-000000000046}'),
		# Method 'DiagramNode' returns object of type 'DiagramNode'
		"DiagramNode": (135, 2, (9, 0), (), "DiagramNode", '{000C0370-0000-0000-C000-000000000046}'),
		# Method 'Fill' returns object of type 'FillFormat'
		"Fill": (107, 2, (9, 0), (), "Fill", '{000C0314-0000-0000-C000-000000000046}'),
		# Method 'GroupItems' returns object of type 'GroupShapes'
		"GroupItems": (108, 2, (9, 0), (), "GroupItems", '{000C0316-0000-0000-C000-000000000046}'),
		"HasDiagram": (132, 2, (3, 0), (), "HasDiagram", None),
		"HasDiagramNode": (134, 2, (3, 0), (), "HasDiagramNode", None),
		"Height": (109, 2, (4, 0), (), "Height", None),
		"HorizontalFlip": (110, 2, (3, 0), (), "HorizontalFlip", None),
		"Id": (139, 2, (3, 0), (), "Id", None),
		"Left": (111, 2, (4, 0), (), "Left", None),
		# Method 'Line' returns object of type 'LineFormat'
		"Line": (112, 2, (9, 0), (), "Line", '{000C0317-0000-0000-C000-000000000046}'),
		"LockAspectRatio": (113, 2, (3, 0), (), "LockAspectRatio", None),
		"Name": (115, 2, (8, 0), (), "Name", None),
		# Method 'Nodes' returns object of type 'ShapeNodes'
		"Nodes": (116, 2, (9, 0), (), "Nodes", '{000C0319-0000-0000-C000-000000000046}'),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		# Method 'ParentGroup' returns object of type 'Shape'
		"ParentGroup": (137, 2, (9, 0), (), "ParentGroup", '{000C031C-0000-0000-C000-000000000046}'),
		# Method 'PictureFormat' returns object of type 'PictureFormat'
		"PictureFormat": (118, 2, (9, 0), (), "PictureFormat", '{000C031A-0000-0000-C000-000000000046}'),
		"Rotation": (117, 2, (4, 0), (), "Rotation", None),
		# Method 'Script' returns object of type 'Script'
		"Script": (130, 2, (9, 0), (), "Script", '{000C0341-0000-0000-C000-000000000046}'),
		# Method 'Shadow' returns object of type 'ShadowFormat'
		"Shadow": (119, 2, (9, 0), (), "Shadow", '{000C031B-0000-0000-C000-000000000046}'),
		# Method 'TextEffect' returns object of type 'TextEffectFormat'
		"TextEffect": (120, 2, (9, 0), (), "TextEffect", '{000C031F-0000-0000-C000-000000000046}'),
		# Method 'TextFrame' returns object of type 'TextFrame'
		"TextFrame": (121, 2, (9, 0), (), "TextFrame", '{000C0320-0000-0000-C000-000000000046}'),
		# Method 'ThreeD' returns object of type 'ThreeDFormat'
		"ThreeD": (122, 2, (9, 0), (), "ThreeD", '{000C0321-0000-0000-C000-000000000046}'),
		"Top": (123, 2, (4, 0), (), "Top", None),
		"Type": (124, 2, (3, 0), (), "Type", None),
		"VerticalFlip": (125, 2, (3, 0), (), "VerticalFlip", None),
		"Vertices": (126, 2, (12, 0), (), "Vertices", None),
		"Visible": (127, 2, (3, 0), (), "Visible", None),
		"Width": (128, 2, (4, 0), (), "Width", None),
		"ZOrderPosition": (129, 2, (3, 0), (), "ZOrderPosition", None),
	}
	_prop_map_put_ = {
		"AlternativeText": ((131, LCID, 4, 0),()),
		"AutoShapeType": ((101, LCID, 4, 0),()),
		"BlackWhiteMode": ((102, LCID, 4, 0),()),
		"Height": ((109, LCID, 4, 0),()),
		"Left": ((111, LCID, 4, 0),()),
		"LockAspectRatio": ((113, LCID, 4, 0),()),
		"Name": ((115, LCID, 4, 0),()),
		"RTF": ((144, LCID, 4, 0),()),
		"Rotation": ((117, LCID, 4, 0),()),
		"Top": ((123, LCID, 4, 0),()),
		"Visible": ((127, LCID, 4, 0),()),
		"Width": ((128, LCID, 4, 0),()),
	}

class ShapeNode(DispatchBaseClass):
	CLSID = IID('{000C0318-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"EditingType": (100, 2, (3, 0), (), "EditingType", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		"Points": (101, 2, (12, 0), (), "Points", None),
		"SegmentType": (102, 2, (3, 0), (), "SegmentType", None),
	}
	_prop_map_put_ = {
	}

class ShapeNodes(DispatchBaseClass):
	CLSID = IID('{000C0319-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Delete(self, Index=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), ((3, 1),),Index
			)

	def Insert(self, Index=defaultNamedNotOptArg, SegmentType=defaultNamedNotOptArg, EditingType=defaultNamedNotOptArg, X1=defaultNamedNotOptArg
			, Y1=defaultNamedNotOptArg, X2=0.0, Y2=0.0, X3=0.0, Y3=0.0):
		return self._oleobj_.InvokeTypes(12, LCID, 1, (24, 0), ((3, 1), (3, 1), (3, 1), (4, 1), (4, 1), (4, 49), (4, 49), (4, 49), (4, 49)),Index
			, SegmentType, EditingType, X1, Y1, X2
			, Y2, X3, Y3)

	# Result is of type ShapeNode
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0318-0000-0000-C000-000000000046}')
		return ret

	def SetEditingType(self, Index=defaultNamedNotOptArg, EditingType=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (24, 0), ((3, 1), (3, 1)),Index
			, EditingType)

	def SetPosition(self, Index=defaultNamedNotOptArg, X1=defaultNamedNotOptArg, Y1=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (24, 0), ((3, 1), (4, 1), (4, 1)),Index
			, X1, Y1)

	def SetSegmentType(self, Index=defaultNamedNotOptArg, SegmentType=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(15, LCID, 1, (24, 0), ((3, 1), (3, 1)),Index
			, SegmentType)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (2, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0318-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0318-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0318-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(2, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class ShapeRange(DispatchBaseClass):
	CLSID = IID('{000C031D-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Align(self, AlignCmd=defaultNamedNotOptArg, RelativeTo=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), ((3, 1), (3, 1)),AlignCmd
			, RelativeTo)

	def Apply(self):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), (),)

	def CanvasCropBottom(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(143, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def CanvasCropLeft(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(140, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def CanvasCropRight(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(142, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def CanvasCropTop(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(141, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def Delete(self):
		return self._oleobj_.InvokeTypes(12, LCID, 1, (24, 0), (),)

	def Distribute(self, DistributeCmd=defaultNamedNotOptArg, RelativeTo=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (24, 0), ((3, 1), (3, 1)),DistributeCmd
			, RelativeTo)

	# Result is of type ShapeRange
	def Duplicate(self):
		ret = self._oleobj_.InvokeTypes(14, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'Duplicate', '{000C031D-0000-0000-C000-000000000046}')
		return ret

	def Flip(self, FlipCmd=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(15, LCID, 1, (24, 0), ((3, 1),),FlipCmd
			)

	# Result is of type Shape
	def Group(self):
		ret = self._oleobj_.InvokeTypes(19, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'Group', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	def IncrementLeft(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(16, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def IncrementRotation(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(17, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def IncrementTop(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(18, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	# Result is of type Shape
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	def PickUp(self):
		return self._oleobj_.InvokeTypes(20, LCID, 1, (24, 0), (),)

	# Result is of type Shape
	def Regroup(self):
		ret = self._oleobj_.InvokeTypes(21, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'Regroup', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	def RerouteConnections(self):
		return self._oleobj_.InvokeTypes(22, LCID, 1, (24, 0), (),)

	def ScaleHeight(self, Factor=defaultNamedNotOptArg, RelativeToOriginalSize=defaultNamedNotOptArg, fScale=0):
		return self._oleobj_.InvokeTypes(23, LCID, 1, (24, 0), ((4, 1), (3, 1), (3, 49)),Factor
			, RelativeToOriginalSize, fScale)

	def ScaleWidth(self, Factor=defaultNamedNotOptArg, RelativeToOriginalSize=defaultNamedNotOptArg, fScale=0):
		return self._oleobj_.InvokeTypes(24, LCID, 1, (24, 0), ((4, 1), (3, 1), (3, 49)),Factor
			, RelativeToOriginalSize, fScale)

	def Select(self, Replace=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(25, LCID, 1, (24, 0), ((12, 17),),Replace
			)

	def SetShapesDefaultProperties(self):
		return self._oleobj_.InvokeTypes(26, LCID, 1, (24, 0), (),)

	# Result is of type ShapeRange
	def Ungroup(self):
		ret = self._oleobj_.InvokeTypes(27, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'Ungroup', '{000C031D-0000-0000-C000-000000000046}')
		return ret

	def ZOrder(self, ZOrderCmd=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(28, LCID, 1, (24, 0), ((3, 1),),ZOrderCmd
			)

	_prop_map_get_ = {
		# Method 'Adjustments' returns object of type 'Adjustments'
		"Adjustments": (100, 2, (9, 0), (), "Adjustments", '{000C0310-0000-0000-C000-000000000046}'),
		"AlternativeText": (131, 2, (8, 0), (), "AlternativeText", None),
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"AutoShapeType": (101, 2, (3, 0), (), "AutoShapeType", None),
		"BlackWhiteMode": (102, 2, (3, 0), (), "BlackWhiteMode", None),
		# Method 'Callout' returns object of type 'CalloutFormat'
		"Callout": (103, 2, (9, 0), (), "Callout", '{000C0311-0000-0000-C000-000000000046}'),
		# Method 'CanvasItems' returns object of type 'CanvasShapes'
		"CanvasItems": (138, 2, (9, 0), (), "CanvasItems", '{000C0371-0000-0000-C000-000000000046}'),
		"Child": (136, 2, (3, 0), (), "Child", None),
		"ConnectionSiteCount": (104, 2, (3, 0), (), "ConnectionSiteCount", None),
		"Connector": (105, 2, (3, 0), (), "Connector", None),
		# Method 'ConnectorFormat' returns object of type 'ConnectorFormat'
		"ConnectorFormat": (106, 2, (9, 0), (), "ConnectorFormat", '{000C0313-0000-0000-C000-000000000046}'),
		"Count": (2, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'Diagram' returns object of type 'IMsoDiagram'
		"Diagram": (133, 2, (9, 0), (), "Diagram", '{000C036D-0000-0000-C000-000000000046}'),
		# Method 'DiagramNode' returns object of type 'DiagramNode'
		"DiagramNode": (135, 2, (9, 0), (), "DiagramNode", '{000C0370-0000-0000-C000-000000000046}'),
		# Method 'Fill' returns object of type 'FillFormat'
		"Fill": (107, 2, (9, 0), (), "Fill", '{000C0314-0000-0000-C000-000000000046}'),
		# Method 'GroupItems' returns object of type 'GroupShapes'
		"GroupItems": (108, 2, (9, 0), (), "GroupItems", '{000C0316-0000-0000-C000-000000000046}'),
		"HasDiagram": (132, 2, (3, 0), (), "HasDiagram", None),
		"HasDiagramNode": (134, 2, (3, 0), (), "HasDiagramNode", None),
		"Height": (109, 2, (4, 0), (), "Height", None),
		"HorizontalFlip": (110, 2, (3, 0), (), "HorizontalFlip", None),
		"Id": (139, 2, (3, 0), (), "Id", None),
		"Left": (111, 2, (4, 0), (), "Left", None),
		# Method 'Line' returns object of type 'LineFormat'
		"Line": (112, 2, (9, 0), (), "Line", '{000C0317-0000-0000-C000-000000000046}'),
		"LockAspectRatio": (113, 2, (3, 0), (), "LockAspectRatio", None),
		"Name": (115, 2, (8, 0), (), "Name", None),
		# Method 'Nodes' returns object of type 'ShapeNodes'
		"Nodes": (116, 2, (9, 0), (), "Nodes", '{000C0319-0000-0000-C000-000000000046}'),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		# Method 'ParentGroup' returns object of type 'Shape'
		"ParentGroup": (137, 2, (9, 0), (), "ParentGroup", '{000C031C-0000-0000-C000-000000000046}'),
		# Method 'PictureFormat' returns object of type 'PictureFormat'
		"PictureFormat": (118, 2, (9, 0), (), "PictureFormat", '{000C031A-0000-0000-C000-000000000046}'),
		"Rotation": (117, 2, (4, 0), (), "Rotation", None),
		# Method 'Script' returns object of type 'Script'
		"Script": (130, 2, (9, 0), (), "Script", '{000C0341-0000-0000-C000-000000000046}'),
		# Method 'Shadow' returns object of type 'ShadowFormat'
		"Shadow": (119, 2, (9, 0), (), "Shadow", '{000C031B-0000-0000-C000-000000000046}'),
		# Method 'TextEffect' returns object of type 'TextEffectFormat'
		"TextEffect": (120, 2, (9, 0), (), "TextEffect", '{000C031F-0000-0000-C000-000000000046}'),
		# Method 'TextFrame' returns object of type 'TextFrame'
		"TextFrame": (121, 2, (9, 0), (), "TextFrame", '{000C0320-0000-0000-C000-000000000046}'),
		# Method 'ThreeD' returns object of type 'ThreeDFormat'
		"ThreeD": (122, 2, (9, 0), (), "ThreeD", '{000C0321-0000-0000-C000-000000000046}'),
		"Top": (123, 2, (4, 0), (), "Top", None),
		"Type": (124, 2, (3, 0), (), "Type", None),
		"VerticalFlip": (125, 2, (3, 0), (), "VerticalFlip", None),
		"Vertices": (126, 2, (12, 0), (), "Vertices", None),
		"Visible": (127, 2, (3, 0), (), "Visible", None),
		"Width": (128, 2, (4, 0), (), "Width", None),
		"ZOrderPosition": (129, 2, (3, 0), (), "ZOrderPosition", None),
	}
	_prop_map_put_ = {
		"AlternativeText": ((131, LCID, 4, 0),()),
		"AutoShapeType": ((101, LCID, 4, 0),()),
		"BlackWhiteMode": ((102, LCID, 4, 0),()),
		"Height": ((109, LCID, 4, 0),()),
		"Left": ((111, LCID, 4, 0),()),
		"LockAspectRatio": ((113, LCID, 4, 0),()),
		"Name": ((115, LCID, 4, 0),()),
		"RTF": ((144, LCID, 4, 0),()),
		"Rotation": ((117, LCID, 4, 0),()),
		"Top": ((123, LCID, 4, 0),()),
		"Visible": ((127, LCID, 4, 0),()),
		"Width": ((128, LCID, 4, 0),()),
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C031C-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C031C-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(2, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class Shapes(DispatchBaseClass):
	CLSID = IID('{000C031E-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type Shape
	def AddCallout(self, Type=defaultNamedNotOptArg, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg, Width=defaultNamedNotOptArg
			, Height=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(10, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1), (4, 1), (4, 1)),Type
			, Left, Top, Width, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddCallout', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddCanvas(self, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg, Width=defaultNamedNotOptArg, Height=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(25, LCID, 1, (9, 0), ((4, 1), (4, 1), (4, 1), (4, 1)),Left
			, Top, Width, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddCanvas', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddConnector(self, Type=defaultNamedNotOptArg, BeginX=defaultNamedNotOptArg, BeginY=defaultNamedNotOptArg, EndX=defaultNamedNotOptArg
			, EndY=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(11, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1), (4, 1), (4, 1)),Type
			, BeginX, BeginY, EndX, EndY)
		if ret is not None:
			ret = Dispatch(ret, u'AddConnector', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddCurve(self, SafeArrayOfPoints=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(12, LCID, 1, (9, 0), ((12, 1),),SafeArrayOfPoints
			)
		if ret is not None:
			ret = Dispatch(ret, u'AddCurve', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddDiagram(self, Type=defaultNamedNotOptArg, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg, Width=defaultNamedNotOptArg
			, Height=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(23, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1), (4, 1), (4, 1)),Type
			, Left, Top, Width, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddDiagram', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddLabel(self, Orientation=defaultNamedNotOptArg, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg, Width=defaultNamedNotOptArg
			, Height=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(13, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1), (4, 1), (4, 1)),Orientation
			, Left, Top, Width, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddLabel', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddLine(self, BeginX=defaultNamedNotOptArg, BeginY=defaultNamedNotOptArg, EndX=defaultNamedNotOptArg, EndY=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(14, LCID, 1, (9, 0), ((4, 1), (4, 1), (4, 1), (4, 1)),BeginX
			, BeginY, EndX, EndY)
		if ret is not None:
			ret = Dispatch(ret, u'AddLine', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddPicture(self, FileName=defaultNamedNotOptArg, LinkToFile=defaultNamedNotOptArg, SaveWithDocument=defaultNamedNotOptArg, Left=defaultNamedNotOptArg
			, Top=defaultNamedNotOptArg, Width=-1.0, Height=-1.0):
		ret = self._oleobj_.InvokeTypes(15, LCID, 1, (9, 0), ((8, 1), (3, 1), (3, 1), (4, 1), (4, 1), (4, 49), (4, 49)),FileName
			, LinkToFile, SaveWithDocument, Left, Top, Width
			, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddPicture', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddPolyline(self, SafeArrayOfPoints=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(16, LCID, 1, (9, 0), ((12, 1),),SafeArrayOfPoints
			)
		if ret is not None:
			ret = Dispatch(ret, u'AddPolyline', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddShape(self, Type=defaultNamedNotOptArg, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg, Width=defaultNamedNotOptArg
			, Height=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(17, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1), (4, 1), (4, 1)),Type
			, Left, Top, Width, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddShape', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddTextEffect(self, PresetTextEffect=defaultNamedNotOptArg, Text=defaultNamedNotOptArg, FontName=defaultNamedNotOptArg, FontSize=defaultNamedNotOptArg
			, FontBold=defaultNamedNotOptArg, FontItalic=defaultNamedNotOptArg, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(18, LCID, 1, (9, 0), ((3, 1), (8, 1), (8, 1), (4, 1), (3, 1), (3, 1), (4, 1), (4, 1)),PresetTextEffect
			, Text, FontName, FontSize, FontBold, FontItalic
			, Left, Top)
		if ret is not None:
			ret = Dispatch(ret, u'AddTextEffect', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def AddTextbox(self, Orientation=defaultNamedNotOptArg, Left=defaultNamedNotOptArg, Top=defaultNamedNotOptArg, Width=defaultNamedNotOptArg
			, Height=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(19, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1), (4, 1), (4, 1)),Orientation
			, Left, Top, Width, Height)
		if ret is not None:
			ret = Dispatch(ret, u'AddTextbox', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type FreeformBuilder
	def BuildFreeform(self, EditingType=defaultNamedNotOptArg, X1=defaultNamedNotOptArg, Y1=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(20, LCID, 1, (9, 0), ((3, 1), (4, 1), (4, 1)),EditingType
			, X1, Y1)
		if ret is not None:
			ret = Dispatch(ret, u'BuildFreeform', '{000C0315-0000-0000-C000-000000000046}')
		return ret

	# Result is of type Shape
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# Result is of type ShapeRange
	def Range(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(21, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Range', '{000C031D-0000-0000-C000-000000000046}')
		return ret

	def SelectAll(self):
		return self._oleobj_.InvokeTypes(22, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		# Method 'Background' returns object of type 'Shape'
		"Background": (100, 2, (9, 0), (), "Background", '{000C031C-0000-0000-C000-000000000046}'),
		"Count": (2, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'Default' returns object of type 'Shape'
		"Default": (101, 2, (9, 0), (), "Default", '{000C031C-0000-0000-C000-000000000046}'),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 1, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C031C-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C031C-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C031C-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(2, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class SharedWorkspace(DispatchBaseClass):
	CLSID = IID('{000C0385-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def CreateNew(self, URL=defaultNamedOptArg, Name=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(7, LCID, 1, (24, 0), ((12, 17), (12, 17)),URL
			, Name)

	def Delete(self):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (24, 0), (),)

	def Disconnect(self):
		return self._oleobj_.InvokeTypes(15, LCID, 1, (24, 0), (),)

	def Refresh(self):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (24, 0), (),)

	def RemoveDocument(self):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Connected": (11, 2, (11, 0), (), "Connected", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		# Method 'Files' returns object of type 'SharedWorkspaceFiles'
		"Files": (3, 2, (9, 0), (), "Files", '{000C037C-0000-0000-C000-000000000046}'),
		# Method 'Folders' returns object of type 'SharedWorkspaceFolders'
		"Folders": (4, 2, (9, 0), (), "Folders", '{000C037E-0000-0000-C000-000000000046}'),
		"LastRefreshed": (12, 2, (12, 0), (), "LastRefreshed", None),
		# Method 'Links' returns object of type 'SharedWorkspaceLinks'
		"Links": (5, 2, (9, 0), (), "Links", '{000C0380-0000-0000-C000-000000000046}'),
		# Method 'Members' returns object of type 'SharedWorkspaceMembers'
		"Members": (1, 2, (9, 0), (), "Members", '{000C0382-0000-0000-C000-000000000046}'),
		"Name": (0, 2, (8, 0), (), "Name", None),
		"Parent": (9, 2, (9, 0), (), "Parent", None),
		"SourceURL": (13, 2, (8, 0), (), "SourceURL", None),
		# Method 'Tasks' returns object of type 'SharedWorkspaceTasks'
		"Tasks": (2, 2, (9, 0), (), "Tasks", '{000C037A-0000-0000-C000-000000000046}'),
		"URL": (10, 2, (8, 0), (), "URL", None),
	}
	_prop_map_put_ = {
		"Name": ((0, LCID, 4, 0),()),
		"SourceURL": ((13, LCID, 4, 0),()),
	}
	# Default property for this class is 'Name'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "Name", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class SharedWorkspaceFile(DispatchBaseClass):
	CLSID = IID('{000C037B-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Delete(self):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"CreatedBy": (1, 2, (8, 0), (), "CreatedBy", None),
		"CreatedDate": (2, 2, (12, 0), (), "CreatedDate", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"ModifiedBy": (3, 2, (8, 0), (), "ModifiedBy", None),
		"ModifiedDate": (4, 2, (12, 0), (), "ModifiedDate", None),
		"Parent": (6, 2, (9, 0), (), "Parent", None),
		"URL": (0, 2, (8, 0), (), "URL", None),
	}
	_prop_map_put_ = {
	}
	# Default property for this class is 'URL'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "URL", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class SharedWorkspaceFiles(DispatchBaseClass):
	CLSID = IID('{000C037C-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type SharedWorkspaceFile
	def Add(self, FileName=defaultNamedNotOptArg, ParentFolder=defaultNamedOptArg, OverwriteIfFileAlreadyExists=defaultNamedOptArg, KeepInSync=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(2, LCID, 1, (9, 0), ((8, 1), (12, 17), (12, 17), (12, 17)),FileName
			, ParentFolder, OverwriteIfFileAlreadyExists, KeepInSync)
		if ret is not None:
			ret = Dispatch(ret, u'Add', '{000C037B-0000-0000-C000-000000000046}')
		return ret

	# Result is of type SharedWorkspaceFile
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C037B-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"ItemCountExceeded": (4, 2, (11, 0), (), "ItemCountExceeded", None),
		"Parent": (3, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C037B-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C037B-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C037B-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class SharedWorkspaceFolder(DispatchBaseClass):
	CLSID = IID('{000C037D-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Delete(self, DeleteEventIfFolderContainsFiles=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), ((12, 17),),DeleteEventIfFolderContainsFiles
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"FolderName": (0, 2, (8, 0), (), "FolderName", None),
		"Parent": (2, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default property for this class is 'FolderName'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "FolderName", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class SharedWorkspaceFolders(DispatchBaseClass):
	CLSID = IID('{000C037E-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type SharedWorkspaceFolder
	def Add(self, FolderName=defaultNamedNotOptArg, ParentFolder=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(2, LCID, 1, (9, 0), ((8, 1), (12, 17)),FolderName
			, ParentFolder)
		if ret is not None:
			ret = Dispatch(ret, u'Add', '{000C037D-0000-0000-C000-000000000046}')
		return ret

	# Result is of type SharedWorkspaceFolder
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C037D-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"ItemCountExceeded": (4, 2, (11, 0), (), "ItemCountExceeded", None),
		"Parent": (3, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C037D-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C037D-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C037D-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class SharedWorkspaceLink(DispatchBaseClass):
	CLSID = IID('{000C037F-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Delete(self):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (24, 0), (),)

	def Save(self):
		return self._oleobj_.InvokeTypes(7, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"CreatedBy": (3, 2, (8, 0), (), "CreatedBy", None),
		"CreatedDate": (4, 2, (12, 0), (), "CreatedDate", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Description": (1, 2, (8, 0), (), "Description", None),
		"ModifiedBy": (5, 2, (8, 0), (), "ModifiedBy", None),
		"ModifiedDate": (6, 2, (12, 0), (), "ModifiedDate", None),
		"Notes": (2, 2, (8, 0), (), "Notes", None),
		"Parent": (9, 2, (9, 0), (), "Parent", None),
		"URL": (0, 2, (8, 0), (), "URL", None),
	}
	_prop_map_put_ = {
		"Description": ((1, LCID, 4, 0),()),
		"Notes": ((2, LCID, 4, 0),()),
		"URL": ((0, LCID, 4, 0),()),
	}
	# Default property for this class is 'URL'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "URL", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class SharedWorkspaceLinks(DispatchBaseClass):
	CLSID = IID('{000C0380-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type SharedWorkspaceLink
	def Add(self, URL=defaultNamedNotOptArg, Description=defaultNamedOptArg, Notes=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(2, LCID, 1, (9, 0), ((8, 1), (12, 17), (12, 17)),URL
			, Description, Notes)
		if ret is not None:
			ret = Dispatch(ret, u'Add', '{000C037F-0000-0000-C000-000000000046}')
		return ret

	# Result is of type SharedWorkspaceLink
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C037F-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"ItemCountExceeded": (4, 2, (11, 0), (), "ItemCountExceeded", None),
		"Parent": (3, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C037F-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C037F-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C037F-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class SharedWorkspaceMember(DispatchBaseClass):
	CLSID = IID('{000C0381-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Delete(self):
		return self._oleobj_.InvokeTypes(3, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"DomainName": (0, 2, (8, 0), (), "DomainName", None),
		"Email": (2, 2, (8, 0), (), "Email", None),
		"Id": (4, 2, (8, 0), (), "Id", None),
		"Name": (1, 2, (8, 0), (), "Name", None),
		"Parent": (5, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default property for this class is 'DomainName'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "DomainName", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class SharedWorkspaceMembers(DispatchBaseClass):
	CLSID = IID('{000C0382-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type SharedWorkspaceMember
	def Add(self, Email=defaultNamedNotOptArg, DomainName=defaultNamedNotOptArg, DisplayName=defaultNamedNotOptArg, Role=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(2, LCID, 1, (9, 0), ((8, 1), (8, 1), (8, 1), (12, 17)),Email
			, DomainName, DisplayName, Role)
		if ret is not None:
			ret = Dispatch(ret, u'Add', '{000C0381-0000-0000-C000-000000000046}')
		return ret

	# Result is of type SharedWorkspaceMember
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0381-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"ItemCountExceeded": (4, 2, (11, 0), (), "ItemCountExceeded", None),
		"Parent": (3, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0381-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0381-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0381-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class SharedWorkspaceTask(DispatchBaseClass):
	CLSID = IID('{000C0379-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Delete(self):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), (),)

	def Save(self):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"AssignedTo": (1, 2, (8, 0), (), "AssignedTo", None),
		"CreatedBy": (6, 2, (8, 0), (), "CreatedBy", None),
		"CreatedDate": (7, 2, (12, 0), (), "CreatedDate", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Description": (4, 2, (8, 0), (), "Description", None),
		"DueDate": (5, 2, (12, 0), (), "DueDate", None),
		"ModifiedBy": (8, 2, (8, 0), (), "ModifiedBy", None),
		"ModifiedDate": (9, 2, (12, 0), (), "ModifiedDate", None),
		"Parent": (12, 2, (9, 0), (), "Parent", None),
		"Priority": (3, 2, (3, 0), (), "Priority", None),
		"Status": (2, 2, (3, 0), (), "Status", None),
		"Title": (0, 2, (8, 0), (), "Title", None),
	}
	_prop_map_put_ = {
		"AssignedTo": ((1, LCID, 4, 0),()),
		"Description": ((4, LCID, 4, 0),()),
		"DueDate": ((5, LCID, 4, 0),()),
		"Priority": ((3, LCID, 4, 0),()),
		"Status": ((2, LCID, 4, 0),()),
		"Title": ((0, LCID, 4, 0),()),
	}
	# Default property for this class is 'Title'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "Title", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class SharedWorkspaceTasks(DispatchBaseClass):
	CLSID = IID('{000C037A-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type SharedWorkspaceTask
	def Add(self, Title=defaultNamedNotOptArg, Status=defaultNamedOptArg, Priority=defaultNamedOptArg, Assignee=defaultNamedOptArg
			, Description=defaultNamedOptArg, DueDate=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(2, LCID, 1, (9, 0), ((8, 1), (12, 17), (12, 17), (12, 17), (12, 17), (12, 17)),Title
			, Status, Priority, Assignee, Description, DueDate
			)
		if ret is not None:
			ret = Dispatch(ret, u'Add', '{000C0379-0000-0000-C000-000000000046}')
		return ret

	# Result is of type SharedWorkspaceTask
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0379-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"ItemCountExceeded": (4, 2, (11, 0), (), "ItemCountExceeded", None),
		"Parent": (3, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0379-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0379-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0379-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class Signature(DispatchBaseClass):
	CLSID = IID('{000C0411-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Delete(self):
		return self._oleobj_.InvokeTypes(1610809350, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"AttachCertificate": (1610809348, 2, (11, 0), (), "AttachCertificate", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"ExpireDate": (1610809346, 2, (12, 0), (), "ExpireDate", None),
		"IsCertificateExpired": (1610809352, 2, (11, 0), (), "IsCertificateExpired", None),
		"IsCertificateRevoked": (1610809353, 2, (11, 0), (), "IsCertificateRevoked", None),
		"IsValid": (1610809347, 2, (11, 0), (), "IsValid", None),
		"Issuer": (1610809345, 2, (8, 0), (), "Issuer", None),
		"Parent": (1610809351, 2, (9, 0), (), "Parent", None),
		"SignDate": (1610809354, 2, (12, 0), (), "SignDate", None),
		"Signer": (1610809344, 2, (8, 0), (), "Signer", None),
	}
	_prop_map_put_ = {
		"AttachCertificate": ((1610809348, LCID, 4, 0),()),
	}

class SignatureSet(DispatchBaseClass):
	CLSID = IID('{000C0410-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type Signature
	def Add(self):
		ret = self._oleobj_.InvokeTypes(1610809347, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'Add', '{000C0411-0000-0000-C000-000000000046}')
		return ret

	def Commit(self):
		return self._oleobj_.InvokeTypes(1610809348, LCID, 1, (24, 0), (),)

	# Result is of type Signature
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, iSig=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),iSig
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0411-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1610809345, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Parent": (1610809349, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, iSig=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),iSig
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0411-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0411-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0411-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1610809345, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class SmartDocument(DispatchBaseClass):
	CLSID = IID('{000C0377-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def PickSolution(self, ConsiderAllSchemas=False):
		return self._oleobj_.InvokeTypes(3, LCID, 1, (24, 0), ((11, 49),),ConsiderAllSchemas
			)

	def RefreshPane(self):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"SolutionID": (1, 2, (8, 0), (), "SolutionID", None),
		"SolutionURL": (2, 2, (8, 0), (), "SolutionURL", None),
	}
	_prop_map_put_ = {
		"SolutionID": ((1, LCID, 4, 0),()),
		"SolutionURL": ((2, LCID, 4, 0),()),
	}

class Sync(DispatchBaseClass):
	CLSID = IID('{000C0386-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def GetUpdate(self):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (24, 0), (),)

	def OpenVersion(self, SyncVersionType=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (24, 0), ((3, 1),),SyncVersionType
			)

	def PutUpdate(self):
		return self._oleobj_.InvokeTypes(7, LCID, 1, (24, 0), (),)

	def ResolveConflict(self, SyncConflictResolution=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(9, LCID, 1, (24, 0), ((3, 1),),SyncConflictResolution
			)

	def Unsuspend(self):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"ErrorType": (4, 2, (3, 0), (), "ErrorType", None),
		"LastSyncTime": (2, 2, (12, 0), (), "LastSyncTime", None),
		"Parent": (14, 2, (9, 0), (), "Parent", None),
		"Status": (0, 2, (3, 0), (), "Status", None),
		"WorkspaceLastChangedBy": (1, 2, (8, 0), (), "WorkspaceLastChangedBy", None),
	}
	_prop_map_put_ = {
	}
	# Default property for this class is 'Status'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (3, 0), (), "Status", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class TextEffectFormat(DispatchBaseClass):
	CLSID = IID('{000C031F-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def ToggleVerticalText(self):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Alignment": (100, 2, (3, 0), (), "Alignment", None),
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"FontBold": (101, 2, (3, 0), (), "FontBold", None),
		"FontItalic": (102, 2, (3, 0), (), "FontItalic", None),
		"FontName": (103, 2, (8, 0), (), "FontName", None),
		"FontSize": (104, 2, (4, 0), (), "FontSize", None),
		"KernedPairs": (105, 2, (3, 0), (), "KernedPairs", None),
		"NormalizedHeight": (106, 2, (3, 0), (), "NormalizedHeight", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		"PresetShape": (107, 2, (3, 0), (), "PresetShape", None),
		"PresetTextEffect": (108, 2, (3, 0), (), "PresetTextEffect", None),
		"RotatedChars": (109, 2, (3, 0), (), "RotatedChars", None),
		"Text": (110, 2, (8, 0), (), "Text", None),
		"Tracking": (111, 2, (4, 0), (), "Tracking", None),
	}
	_prop_map_put_ = {
		"Alignment": ((100, LCID, 4, 0),()),
		"FontBold": ((101, LCID, 4, 0),()),
		"FontItalic": ((102, LCID, 4, 0),()),
		"FontName": ((103, LCID, 4, 0),()),
		"FontSize": ((104, LCID, 4, 0),()),
		"KernedPairs": ((105, LCID, 4, 0),()),
		"NormalizedHeight": ((106, LCID, 4, 0),()),
		"PresetShape": ((107, LCID, 4, 0),()),
		"PresetTextEffect": ((108, LCID, 4, 0),()),
		"RotatedChars": ((109, LCID, 4, 0),()),
		"Text": ((110, LCID, 4, 0),()),
		"Tracking": ((111, LCID, 4, 0),()),
	}

class TextFrame(DispatchBaseClass):
	CLSID = IID('{000C0320-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"MarginBottom": (100, 2, (4, 0), (), "MarginBottom", None),
		"MarginLeft": (101, 2, (4, 0), (), "MarginLeft", None),
		"MarginRight": (102, 2, (4, 0), (), "MarginRight", None),
		"MarginTop": (103, 2, (4, 0), (), "MarginTop", None),
		"Orientation": (104, 2, (3, 0), (), "Orientation", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
		"MarginBottom": ((100, LCID, 4, 0),()),
		"MarginLeft": ((101, LCID, 4, 0),()),
		"MarginRight": ((102, LCID, 4, 0),()),
		"MarginTop": ((103, LCID, 4, 0),()),
		"Orientation": ((104, LCID, 4, 0),()),
	}

class ThreeDFormat(DispatchBaseClass):
	CLSID = IID('{000C0321-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def IncrementRotationX(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def IncrementRotationY(self, Increment=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (24, 0), ((4, 1),),Increment
			)

	def ResetRotation(self):
		return self._oleobj_.InvokeTypes(12, LCID, 1, (24, 0), (),)

	def SetExtrusionDirection(self, PresetExtrusionDirection=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (24, 0), ((3, 1),),PresetExtrusionDirection
			)

	def SetThreeDFormat(self, PresetThreeDFormat=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (24, 0), ((3, 1),),PresetThreeDFormat
			)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"Depth": (100, 2, (4, 0), (), "Depth", None),
		# Method 'ExtrusionColor' returns object of type 'ColorFormat'
		"ExtrusionColor": (101, 2, (9, 0), (), "ExtrusionColor", '{000C0312-0000-0000-C000-000000000046}'),
		"ExtrusionColorType": (102, 2, (3, 0), (), "ExtrusionColorType", None),
		"Parent": (1, 2, (9, 0), (), "Parent", None),
		"Perspective": (103, 2, (3, 0), (), "Perspective", None),
		"PresetExtrusionDirection": (104, 2, (3, 0), (), "PresetExtrusionDirection", None),
		"PresetLightingDirection": (105, 2, (3, 0), (), "PresetLightingDirection", None),
		"PresetLightingSoftness": (106, 2, (3, 0), (), "PresetLightingSoftness", None),
		"PresetMaterial": (107, 2, (3, 0), (), "PresetMaterial", None),
		"PresetThreeDFormat": (108, 2, (3, 0), (), "PresetThreeDFormat", None),
		"RotationX": (109, 2, (4, 0), (), "RotationX", None),
		"RotationY": (110, 2, (4, 0), (), "RotationY", None),
		"Visible": (111, 2, (3, 0), (), "Visible", None),
	}
	_prop_map_put_ = {
		"Depth": ((100, LCID, 4, 0),()),
		"ExtrusionColorType": ((102, LCID, 4, 0),()),
		"Perspective": ((103, LCID, 4, 0),()),
		"PresetLightingDirection": ((105, LCID, 4, 0),()),
		"PresetLightingSoftness": ((106, LCID, 4, 0),()),
		"PresetMaterial": ((107, LCID, 4, 0),()),
		"RotationX": ((109, LCID, 4, 0),()),
		"RotationY": ((110, LCID, 4, 0),()),
		"Visible": ((111, LCID, 4, 0),()),
	}

class UserPermission(DispatchBaseClass):
	CLSID = IID('{000C0375-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Remove(self):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"ExpirationDate": (2, 2, (12, 0), (), "ExpirationDate", None),
		"Parent": (3, 2, (9, 0), (), "Parent", None),
		"Permission": (1, 2, (3, 0), (), "Permission", None),
		"UserId": (0, 2, (8, 0), (), "UserId", None),
	}
	_prop_map_put_ = {
		"ExpirationDate": ((2, LCID, 4, 0),()),
		"Permission": ((1, LCID, 4, 0),()),
	}
	# Default property for this class is 'UserId'
	def __call__(self):
		return self._ApplyTypes_(*(0, 2, (8, 0), (), "UserId", None))
	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))

class WebComponent(DispatchBaseClass):
	CLSID = IID('{000CD100-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def Commit(self):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (24, 0), (),)

	def Revert(self):
		return self._oleobj_.InvokeTypes(9, LCID, 1, (24, 0), (),)

	def SetPlaceHolderGraphic(self, PlaceHolderGraphic=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(7, LCID, 1, (24, 0), ((8, 1),),PlaceHolderGraphic
			)

	_prop_map_get_ = {
		"HTML": (3, 2, (8, 0), (), "HTML", None),
		"Height": (6, 2, (3, 0), (), "Height", None),
		"Name": (4, 2, (8, 0), (), "Name", None),
		"Shape": (1, 2, (9, 0), (), "Shape", None),
		"URL": (2, 2, (8, 0), (), "URL", None),
		"Width": (5, 2, (3, 0), (), "Width", None),
	}
	_prop_map_put_ = {
		"HTML": ((3, LCID, 4, 0),()),
		"Height": ((6, LCID, 4, 0),()),
		"Name": ((4, LCID, 4, 0),()),
		"URL": ((2, LCID, 4, 0),()),
		"Width": ((5, LCID, 4, 0),()),
	}

class WebComponentFormat(DispatchBaseClass):
	CLSID = IID('{000CD102-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def LaunchPropertiesWindow(self):
		return self._oleobj_.InvokeTypes(9, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (1, 2, (9, 0), (), "Application", None),
		"HTML": (4, 2, (8, 0), (), "HTML", None),
		"Height": (7, 2, (3, 0), (), "Height", None),
		"Name": (5, 2, (8, 0), (), "Name", None),
		"Parent": (2, 2, (9, 0), (), "Parent", None),
		"PreviewGraphic": (8, 2, (8, 0), (), "PreviewGraphic", None),
		"URL": (3, 2, (8, 0), (), "URL", None),
		"Width": (6, 2, (3, 0), (), "Width", None),
	}
	_prop_map_put_ = {
		"HTML": ((4, LCID, 4, 0),()),
		"Height": ((7, LCID, 4, 0),()),
		"Name": ((5, LCID, 4, 0),()),
		"PreviewGraphic": ((8, LCID, 4, 0),()),
		"URL": ((3, LCID, 4, 0),()),
		"Width": ((6, LCID, 4, 0),()),
	}

class WebComponentProperties(DispatchBaseClass):
	CLSID = IID('{000C0373-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"HTML": (4, 2, (8, 0), (), "HTML", None),
		"Height": (8, 2, (3, 0), (), "Height", None),
		"Name": (2, 2, (8, 0), (), "Name", None),
		"PreviewGraphic": (5, 2, (8, 0), (), "PreviewGraphic", None),
		"PreviewHTML": (6, 2, (8, 0), (), "PreviewHTML", None),
		"Shape": (1, 2, (9, 0), (), "Shape", None),
		"Tag": (9, 2, (8, 0), (), "Tag", None),
		"URL": (3, 2, (8, 0), (), "URL", None),
		"Width": (7, 2, (3, 0), (), "Width", None),
	}
	_prop_map_put_ = {
		"HTML": ((4, LCID, 4, 0),()),
		"Height": ((8, LCID, 4, 0),()),
		"Name": ((2, LCID, 4, 0),()),
		"PreviewGraphic": ((5, LCID, 4, 0),()),
		"PreviewHTML": ((6, LCID, 4, 0),()),
		"Tag": ((9, LCID, 4, 0),()),
		"URL": ((3, LCID, 4, 0),()),
		"Width": ((7, LCID, 4, 0),()),
	}

class WebComponentWindowExternal(DispatchBaseClass):
	CLSID = IID('{000CD101-0000-0000-C000-000000000046}')
	coclass_clsid = None

	def CloseWindow(self):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Application": (4, 2, (9, 0), (), "Application", None),
		"ApplicationName": (2, 2, (8, 0), (), "ApplicationName", None),
		"ApplicationVersion": (3, 2, (3, 0), (), "ApplicationVersion", None),
		"InterfaceVersion": (1, 2, (3, 0), (), "InterfaceVersion", None),
		# Method 'WebComponent' returns object of type 'WebComponent'
		"WebComponent": (6, 2, (9, 0), (), "WebComponent", '{000CD100-0000-0000-C000-000000000046}'),
	}
	_prop_map_put_ = {
	}

class WebPageFont(DispatchBaseClass):
	CLSID = IID('{000C0913-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"FixedWidthFont": (12, 2, (8, 0), (), "FixedWidthFont", None),
		"FixedWidthFontSize": (13, 2, (4, 0), (), "FixedWidthFontSize", None),
		"ProportionalFont": (10, 2, (8, 0), (), "ProportionalFont", None),
		"ProportionalFontSize": (11, 2, (4, 0), (), "ProportionalFontSize", None),
	}
	_prop_map_put_ = {
		"FixedWidthFont": ((12, LCID, 4, 0),()),
		"FixedWidthFontSize": ((13, LCID, 4, 0),()),
		"ProportionalFont": ((10, LCID, 4, 0),()),
		"ProportionalFontSize": ((11, LCID, 4, 0),()),
	}

class WebPageFonts(DispatchBaseClass):
	CLSID = IID('{000C0914-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type WebPageFont
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0913-0000-0000-C000-000000000046}')
		return ret

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((3, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0913-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0913-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0913-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class _CommandBarActiveX(DispatchBaseClass):
	CLSID = IID('{000C030D-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# Result is of type CommandBarControl
	def Copy(self, Bar=defaultNamedOptArg, Before=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610874886, LCID, 1, (9, 0), ((12, 17), (12, 17)),Bar
			, Before)
		if ret is not None:
			ret = Dispatch(ret, u'Copy', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	def Delete(self, Temporary=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1610874887, LCID, 1, (24, 0), ((12, 17),),Temporary
			)

	def EnsureControl(self):
		return self._oleobj_.InvokeTypes(1610940420, LCID, 1, (24, 0), (),)

	def Execute(self):
		return self._oleobj_.InvokeTypes(1610874892, LCID, 1, (24, 0), (),)

	# The method GetaccDefaultAction is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDefaultAction(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5013, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccDescription is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDescription(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5005, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelp is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelp(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5008, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelpTopic is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelpTopic(self, pszHelpFile=global_Missing, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5009, 2, (3, 0), ((16392, 2), (12, 17)), u'GetaccHelpTopic', None,pszHelpFile
			, varChild)

	# The method GetaccKeyboardShortcut is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccKeyboardShortcut(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5010, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccName(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5003, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccRole is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccRole(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5006, 2, (12, 0), ((12, 17),), u'GetaccRole', None,varChild
			)

	# The method GetaccState is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccState(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5007, 2, (12, 0), ((12, 17),), u'GetaccState', None,varChild
			)

	# The method GetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccValue(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5004, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# Result is of type CommandBarControl
	def Move(self, Bar=defaultNamedOptArg, Before=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610874902, LCID, 1, (9, 0), ((12, 17), (12, 17)),Bar
			, Before)
		if ret is not None:
			ret = Dispatch(ret, u'Move', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	# The method QueryControlInterface is actually a property, but must be used as a method to correctly pass the arguments
	def QueryControlInterface(self, bstrIid=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(1610940418, LCID, 2, (13, 0), ((8, 1),),bstrIid
			)
		if ret is not None:
			# See if this IUnknown is really an IDispatch
			try:
				ret = ret.QueryInterface(global_IID_IDispatch)
			except global_error:
				return ret
			ret = Dispatch(ret, u'QueryControlInterface', None)
		return ret

	def Reserved1(self):
		return self._oleobj_.InvokeTypes(1610874926, LCID, 1, (24, 0), (),)

	def Reserved2(self):
		return self._oleobj_.InvokeTypes(1610874927, LCID, 1, (24, 0), (),)

	def Reserved3(self):
		return self._oleobj_.InvokeTypes(1610874928, LCID, 1, (24, 0), (),)

	def Reserved4(self):
		return self._oleobj_.InvokeTypes(1610874929, LCID, 1, (24, 0), (),)

	def Reserved5(self):
		return self._oleobj_.InvokeTypes(1610874930, LCID, 1, (24, 0), (),)

	def Reserved6(self):
		return self._oleobj_.InvokeTypes(1610874931, LCID, 1, (24, 0), (),)

	def Reserved7(self):
		return self._oleobj_.InvokeTypes(1610874932, LCID, 1, (24, 0), (),)

	def Reset(self):
		return self._oleobj_.InvokeTypes(1610874913, LCID, 1, (24, 0), (),)

	def SetFocus(self):
		return self._oleobj_.InvokeTypes(1610874914, LCID, 1, (24, 0), (),)

	def SetInnerObjectFactory(self, pUnk=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1610940419, LCID, 1, (24, 0), ((13, 1),),pUnk
			)

	# The method SetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccName(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5003, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method SetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccValue(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5004, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method accChild is actually a property, but must be used as a method to correctly pass the arguments
	def accChild(self, varChild=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(-5002, LCID, 2, (9, 0), ((12, 1),),varChild
			)
		if ret is not None:
			ret = Dispatch(ret, u'accChild', None)
		return ret

	def accDoDefaultAction(self, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5018, LCID, 1, (24, 0), ((12, 17),),varChild
			)

	def accHitTest(self, xLeft=defaultNamedNotOptArg, yTop=defaultNamedNotOptArg):
		return self._ApplyTypes_(-5017, 1, (12, 0), ((3, 1), (3, 1)), u'accHitTest', None,xLeft
			, yTop)

	def accLocation(self, pxLeft=global_Missing, pyTop=global_Missing, pcxWidth=global_Missing, pcyHeight=global_Missing
			, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5015, 1, (24, 0), ((16387, 2), (16387, 2), (16387, 2), (16387, 2), (12, 17)), u'accLocation', None,pxLeft
			, pyTop, pcxWidth, pcyHeight, varChild)

	def accNavigate(self, navDir=defaultNamedNotOptArg, varStart=defaultNamedOptArg):
		return self._ApplyTypes_(-5016, 1, (12, 0), ((3, 1), (12, 17)), u'accNavigate', None,navDir
			, varStart)

	def accSelect(self, flagsSelect=defaultNamedNotOptArg, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5014, LCID, 1, (24, 0), ((3, 1), (12, 17)),flagsSelect
			, varChild)

	_prop_map_get_ = {
		"Application": (1610809344, 2, (9, 0), (), "Application", None),
		"BeginGroup": (1610874880, 2, (11, 0), (), "BeginGroup", None),
		"BuiltIn": (1610874882, 2, (11, 0), (), "BuiltIn", None),
		"Caption": (1610874883, 2, (8, 0), (), "Caption", None),
		"Control": (1610874885, 2, (9, 0), (), "Control", None),
		"ControlCLSID": (1610940416, 2, (8, 0), (), "ControlCLSID", None),
		"Creator": (1610809345, 2, (3, 0), (), "Creator", None),
		"DescriptionText": (1610874888, 2, (8, 0), (), "DescriptionText", None),
		"Enabled": (1610874890, 2, (11, 0), (), "Enabled", None),
		"Height": (1610874893, 2, (3, 0), (), "Height", None),
		"HelpContextId": (1610874895, 2, (3, 0), (), "HelpContextId", None),
		"HelpFile": (1610874897, 2, (8, 0), (), "HelpFile", None),
		"Id": (1610874899, 2, (3, 0), (), "Id", None),
		"Index": (1610874900, 2, (3, 0), (), "Index", None),
		"InstanceId": (1610874901, 2, (3, 0), (), "InstanceId", None),
		"IsPriorityDropped": (1610874925, 2, (11, 0), (), "IsPriorityDropped", None),
		"Left": (1610874903, 2, (3, 0), (), "Left", None),
		"OLEUsage": (1610874904, 2, (3, 0), (), "OLEUsage", None),
		"OnAction": (1610874906, 2, (8, 0), (), "OnAction", None),
		"Parameter": (1610874909, 2, (8, 0), (), "Parameter", None),
		# Method 'Parent' returns object of type 'CommandBar'
		"Parent": (1610874908, 2, (9, 0), (), "Parent", '{000C0304-0000-0000-C000-000000000046}'),
		"Priority": (1610874911, 2, (3, 0), (), "Priority", None),
		"Tag": (1610874915, 2, (8, 0), (), "Tag", None),
		"TooltipText": (1610874917, 2, (8, 0), (), "TooltipText", None),
		"Top": (1610874919, 2, (3, 0), (), "Top", None),
		"Type": (1610874920, 2, (3, 0), (), "Type", None),
		"Visible": (1610874921, 2, (11, 0), (), "Visible", None),
		"Width": (1610874923, 2, (3, 0), (), "Width", None),
		"accChildCount": (-5001, 2, (3, 0), (), "accChildCount", None),
		"accDefaultAction": (-5013, 2, (8, 0), ((12, 17),), "accDefaultAction", None),
		"accDescription": (-5005, 2, (8, 0), ((12, 17),), "accDescription", None),
		"accFocus": (-5011, 2, (12, 0), (), "accFocus", None),
		"accHelp": (-5008, 2, (8, 0), ((12, 17),), "accHelp", None),
		"accHelpTopic": (-5009, 2, (3, 0), ((16392, 2), (12, 17)), "accHelpTopic", None),
		"accKeyboardShortcut": (-5010, 2, (8, 0), ((12, 17),), "accKeyboardShortcut", None),
		"accName": (-5003, 2, (8, 0), ((12, 17),), "accName", None),
		"accParent": (-5000, 2, (9, 0), (), "accParent", None),
		"accRole": (-5006, 2, (12, 0), ((12, 17),), "accRole", None),
		"accSelection": (-5012, 2, (12, 0), (), "accSelection", None),
		"accState": (-5007, 2, (12, 0), ((12, 17),), "accState", None),
		"accValue": (-5004, 2, (8, 0), ((12, 17),), "accValue", None),
	}
	_prop_map_put_ = {
		"BeginGroup": ((1610874880, LCID, 4, 0),()),
		"Caption": ((1610874883, LCID, 4, 0),()),
		"ControlCLSID": ((1610940416, LCID, 4, 0),()),
		"DescriptionText": ((1610874888, LCID, 4, 0),()),
		"Enabled": ((1610874890, LCID, 4, 0),()),
		"Height": ((1610874893, LCID, 4, 0),()),
		"HelpContextId": ((1610874895, LCID, 4, 0),()),
		"HelpFile": ((1610874897, LCID, 4, 0),()),
		"InitWith": ((1610940421, LCID, 4, 0),()),
		"OLEUsage": ((1610874904, LCID, 4, 0),()),
		"OnAction": ((1610874906, LCID, 4, 0),()),
		"Parameter": ((1610874909, LCID, 4, 0),()),
		"Priority": ((1610874911, LCID, 4, 0),()),
		"Tag": ((1610874915, LCID, 4, 0),()),
		"TooltipText": ((1610874917, LCID, 4, 0),()),
		"Visible": ((1610874921, LCID, 4, 0),()),
		"Width": ((1610874923, LCID, 4, 0),()),
		"accName": ((-5003, LCID, 4, 0),()),
		"accValue": ((-5004, LCID, 4, 0),()),
	}

class _CommandBarButton(DispatchBaseClass):
	CLSID = IID('{000C030E-0000-0000-C000-000000000046}')
	coclass_clsid = IID('{55F88891-7708-11D1-ACEB-006008961DA5}')

	# Result is of type CommandBarControl
	def Copy(self, Bar=defaultNamedOptArg, Before=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610874886, LCID, 1, (9, 0), ((12, 17), (12, 17)),Bar
			, Before)
		if ret is not None:
			ret = Dispatch(ret, u'Copy', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	def CopyFace(self):
		return self._oleobj_.InvokeTypes(1610940418, LCID, 1, (24, 0), (),)

	def Delete(self, Temporary=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1610874887, LCID, 1, (24, 0), ((12, 17),),Temporary
			)

	def Execute(self):
		return self._oleobj_.InvokeTypes(1610874892, LCID, 1, (24, 0), (),)

	# The method GetaccDefaultAction is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDefaultAction(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5013, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccDescription is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDescription(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5005, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelp is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelp(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5008, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelpTopic is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelpTopic(self, pszHelpFile=global_Missing, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5009, 2, (3, 0), ((16392, 2), (12, 17)), u'GetaccHelpTopic', None,pszHelpFile
			, varChild)

	# The method GetaccKeyboardShortcut is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccKeyboardShortcut(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5010, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccName(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5003, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccRole is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccRole(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5006, 2, (12, 0), ((12, 17),), u'GetaccRole', None,varChild
			)

	# The method GetaccState is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccState(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5007, 2, (12, 0), ((12, 17),), u'GetaccState', None,varChild
			)

	# The method GetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccValue(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5004, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# Result is of type CommandBarControl
	def Move(self, Bar=defaultNamedOptArg, Before=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610874902, LCID, 1, (9, 0), ((12, 17), (12, 17)),Bar
			, Before)
		if ret is not None:
			ret = Dispatch(ret, u'Move', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	def PasteFace(self):
		return self._oleobj_.InvokeTypes(1610940421, LCID, 1, (24, 0), (),)

	def Reserved1(self):
		return self._oleobj_.InvokeTypes(1610874926, LCID, 1, (24, 0), (),)

	def Reserved2(self):
		return self._oleobj_.InvokeTypes(1610874927, LCID, 1, (24, 0), (),)

	def Reserved3(self):
		return self._oleobj_.InvokeTypes(1610874928, LCID, 1, (24, 0), (),)

	def Reserved4(self):
		return self._oleobj_.InvokeTypes(1610874929, LCID, 1, (24, 0), (),)

	def Reserved5(self):
		return self._oleobj_.InvokeTypes(1610874930, LCID, 1, (24, 0), (),)

	def Reserved6(self):
		return self._oleobj_.InvokeTypes(1610874931, LCID, 1, (24, 0), (),)

	def Reserved7(self):
		return self._oleobj_.InvokeTypes(1610874932, LCID, 1, (24, 0), (),)

	def Reset(self):
		return self._oleobj_.InvokeTypes(1610874913, LCID, 1, (24, 0), (),)

	def SetFocus(self):
		return self._oleobj_.InvokeTypes(1610874914, LCID, 1, (24, 0), (),)

	# The method SetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccName(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5003, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method SetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccValue(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5004, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method accChild is actually a property, but must be used as a method to correctly pass the arguments
	def accChild(self, varChild=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(-5002, LCID, 2, (9, 0), ((12, 1),),varChild
			)
		if ret is not None:
			ret = Dispatch(ret, u'accChild', None)
		return ret

	def accDoDefaultAction(self, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5018, LCID, 1, (24, 0), ((12, 17),),varChild
			)

	def accHitTest(self, xLeft=defaultNamedNotOptArg, yTop=defaultNamedNotOptArg):
		return self._ApplyTypes_(-5017, 1, (12, 0), ((3, 1), (3, 1)), u'accHitTest', None,xLeft
			, yTop)

	def accLocation(self, pxLeft=global_Missing, pyTop=global_Missing, pcxWidth=global_Missing, pcyHeight=global_Missing
			, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5015, 1, (24, 0), ((16387, 2), (16387, 2), (16387, 2), (16387, 2), (12, 17)), u'accLocation', None,pxLeft
			, pyTop, pcxWidth, pcyHeight, varChild)

	def accNavigate(self, navDir=defaultNamedNotOptArg, varStart=defaultNamedOptArg):
		return self._ApplyTypes_(-5016, 1, (12, 0), ((3, 1), (12, 17)), u'accNavigate', None,navDir
			, varStart)

	def accSelect(self, flagsSelect=defaultNamedNotOptArg, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5014, LCID, 1, (24, 0), ((3, 1), (12, 17)),flagsSelect
			, varChild)

	_prop_map_get_ = {
		"Application": (1610809344, 2, (9, 0), (), "Application", None),
		"BeginGroup": (1610874880, 2, (11, 0), (), "BeginGroup", None),
		"BuiltIn": (1610874882, 2, (11, 0), (), "BuiltIn", None),
		"BuiltInFace": (1610940416, 2, (11, 0), (), "BuiltInFace", None),
		"Caption": (1610874883, 2, (8, 0), (), "Caption", None),
		"Control": (1610874885, 2, (9, 0), (), "Control", None),
		"Creator": (1610809345, 2, (3, 0), (), "Creator", None),
		"DescriptionText": (1610874888, 2, (8, 0), (), "DescriptionText", None),
		"Enabled": (1610874890, 2, (11, 0), (), "Enabled", None),
		"FaceId": (1610940419, 2, (3, 0), (), "FaceId", None),
		"Height": (1610874893, 2, (3, 0), (), "Height", None),
		"HelpContextId": (1610874895, 2, (3, 0), (), "HelpContextId", None),
		"HelpFile": (1610874897, 2, (8, 0), (), "HelpFile", None),
		"HyperlinkType": (1610940428, 2, (3, 0), (), "HyperlinkType", None),
		"Id": (1610874899, 2, (3, 0), (), "Id", None),
		"Index": (1610874900, 2, (3, 0), (), "Index", None),
		"InstanceId": (1610874901, 2, (3, 0), (), "InstanceId", None),
		"IsPriorityDropped": (1610874925, 2, (11, 0), (), "IsPriorityDropped", None),
		"Left": (1610874903, 2, (3, 0), (), "Left", None),
		# Method 'Mask' returns object of type 'Picture'
		"Mask": (1610940432, 2, (9, 0), (), "Mask", '{7BF80981-BF32-101A-8BBB-00AA00300CAB}'),
		"OLEUsage": (1610874904, 2, (3, 0), (), "OLEUsage", None),
		"OnAction": (1610874906, 2, (8, 0), (), "OnAction", None),
		"Parameter": (1610874909, 2, (8, 0), (), "Parameter", None),
		# Method 'Parent' returns object of type 'CommandBar'
		"Parent": (1610874908, 2, (9, 0), (), "Parent", '{000C0304-0000-0000-C000-000000000046}'),
		# Method 'Picture' returns object of type 'Picture'
		"Picture": (1610940430, 2, (9, 0), (), "Picture", '{7BF80981-BF32-101A-8BBB-00AA00300CAB}'),
		"Priority": (1610874911, 2, (3, 0), (), "Priority", None),
		"ShortcutText": (1610940422, 2, (8, 0), (), "ShortcutText", None),
		"State": (1610940424, 2, (3, 0), (), "State", None),
		"Style": (1610940426, 2, (3, 0), (), "Style", None),
		"Tag": (1610874915, 2, (8, 0), (), "Tag", None),
		"TooltipText": (1610874917, 2, (8, 0), (), "TooltipText", None),
		"Top": (1610874919, 2, (3, 0), (), "Top", None),
		"Type": (1610874920, 2, (3, 0), (), "Type", None),
		"Visible": (1610874921, 2, (11, 0), (), "Visible", None),
		"Width": (1610874923, 2, (3, 0), (), "Width", None),
		"accChildCount": (-5001, 2, (3, 0), (), "accChildCount", None),
		"accDefaultAction": (-5013, 2, (8, 0), ((12, 17),), "accDefaultAction", None),
		"accDescription": (-5005, 2, (8, 0), ((12, 17),), "accDescription", None),
		"accFocus": (-5011, 2, (12, 0), (), "accFocus", None),
		"accHelp": (-5008, 2, (8, 0), ((12, 17),), "accHelp", None),
		"accHelpTopic": (-5009, 2, (3, 0), ((16392, 2), (12, 17)), "accHelpTopic", None),
		"accKeyboardShortcut": (-5010, 2, (8, 0), ((12, 17),), "accKeyboardShortcut", None),
		"accName": (-5003, 2, (8, 0), ((12, 17),), "accName", None),
		"accParent": (-5000, 2, (9, 0), (), "accParent", None),
		"accRole": (-5006, 2, (12, 0), ((12, 17),), "accRole", None),
		"accSelection": (-5012, 2, (12, 0), (), "accSelection", None),
		"accState": (-5007, 2, (12, 0), ((12, 17),), "accState", None),
		"accValue": (-5004, 2, (8, 0), ((12, 17),), "accValue", None),
	}
	_prop_map_put_ = {
		"BeginGroup": ((1610874880, LCID, 4, 0),()),
		"BuiltInFace": ((1610940416, LCID, 4, 0),()),
		"Caption": ((1610874883, LCID, 4, 0),()),
		"DescriptionText": ((1610874888, LCID, 4, 0),()),
		"Enabled": ((1610874890, LCID, 4, 0),()),
		"FaceId": ((1610940419, LCID, 4, 0),()),
		"Height": ((1610874893, LCID, 4, 0),()),
		"HelpContextId": ((1610874895, LCID, 4, 0),()),
		"HelpFile": ((1610874897, LCID, 4, 0),()),
		"HyperlinkType": ((1610940428, LCID, 4, 0),()),
		"Mask": ((1610940432, LCID, 4, 0),()),
		"OLEUsage": ((1610874904, LCID, 4, 0),()),
		"OnAction": ((1610874906, LCID, 4, 0),()),
		"Parameter": ((1610874909, LCID, 4, 0),()),
		"Picture": ((1610940430, LCID, 4, 0),()),
		"Priority": ((1610874911, LCID, 4, 0),()),
		"ShortcutText": ((1610940422, LCID, 4, 0),()),
		"State": ((1610940424, LCID, 4, 0),()),
		"Style": ((1610940426, LCID, 4, 0),()),
		"Tag": ((1610874915, LCID, 4, 0),()),
		"TooltipText": ((1610874917, LCID, 4, 0),()),
		"Visible": ((1610874921, LCID, 4, 0),()),
		"Width": ((1610874923, LCID, 4, 0),()),
		"accName": ((-5003, LCID, 4, 0),()),
		"accValue": ((-5004, LCID, 4, 0),()),
	}

class _CommandBarButtonEvents:
	CLSID = CLSID_Sink = IID('{000C0351-0000-0000-C000-000000000046}')
	coclass_clsid = IID('{55F88891-7708-11D1-ACEB-006008961DA5}')
	_public_methods_ = [] # For COM Server support
	_dispid_to_func_ = {
		1610678275 : "OnInvoke",
		1610678273 : "OnGetTypeInfo",
		        1 : "OnClick",
		1610612737 : "OnAddRef",
		1610612736 : "OnQueryInterface",
		1610612738 : "OnRelease",
		1610678274 : "OnGetIDsOfNames",
		1610678272 : "OnGetTypeInfoCount",
		}

	def __init__(self, oobj = None):
		if oobj is None:
			self._olecp = None
		else:
			import win32com.server.util
			from win32com.server.policy import EventHandlerPolicy
			cpc=oobj._oleobj_.QueryInterface(global_IID_IConnectionPointContainer)
			cp=cpc.FindConnectionPoint(self.CLSID_Sink)
			cookie=cp.Advise(win32com.server.util.wrap(self, usePolicy=EventHandlerPolicy))
			self._olecp,self._olecp_cookie = cp,cookie
	def __del__(self):
		try:
			self.close()
		except global_com_error:
			pass
	def close(self):
		if self._olecp is not None:
			cp,cookie,self._olecp,self._olecp_cookie = self._olecp,self._olecp_cookie,None,None
			cp.Unadvise(cookie)
	def _query_interface_(self, iid):
		import win32com.server.util
		if iid==self.CLSID_Sink: return win32com.server.util.wrap(self)

	# Event Handlers
	# If you create handlers, they should have the following prototypes:
#	def OnInvoke(self, dispidMember=defaultNamedNotOptArg, riid=defaultNamedNotOptArg, lcid=defaultNamedNotOptArg, wFlags=defaultNamedNotOptArg
#			, pdispparams=defaultNamedNotOptArg, pvarResult=global_Missing, pexcepinfo=global_Missing, puArgErr=global_Missing):
#	def OnGetTypeInfo(self, itinfo=defaultNamedNotOptArg, lcid=defaultNamedNotOptArg, pptinfo=global_Missing):
#	def OnClick(self, Ctrl=defaultNamedNotOptArg, CancelDefault=defaultNamedNotOptArg):
#	def OnAddRef(self):
#	def OnQueryInterface(self, riid=defaultNamedNotOptArg, ppvObj=global_Missing):
#	def OnRelease(self):
#	def OnGetIDsOfNames(self, riid=defaultNamedNotOptArg, rgszNames=defaultNamedNotOptArg, cNames=defaultNamedNotOptArg, lcid=defaultNamedNotOptArg
#			, rgdispid=global_Missing):
#	def OnGetTypeInfoCount(self, pctinfo=global_Missing):


class _CommandBarComboBox(DispatchBaseClass):
	CLSID = IID('{000C030C-0000-0000-C000-000000000046}')
	coclass_clsid = IID('{55F88897-7708-11D1-ACEB-006008961DA5}')

	def AddItem(self, Text=defaultNamedNotOptArg, Index=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1610940416, LCID, 1, (24, 0), ((8, 1), (12, 17)),Text
			, Index)

	def Clear(self):
		return self._oleobj_.InvokeTypes(1610940417, LCID, 1, (24, 0), (),)

	# Result is of type CommandBarControl
	def Copy(self, Bar=defaultNamedOptArg, Before=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610874886, LCID, 1, (9, 0), ((12, 17), (12, 17)),Bar
			, Before)
		if ret is not None:
			ret = Dispatch(ret, u'Copy', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	def Delete(self, Temporary=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(1610874887, LCID, 1, (24, 0), ((12, 17),),Temporary
			)

	def Execute(self):
		return self._oleobj_.InvokeTypes(1610874892, LCID, 1, (24, 0), (),)

	# The method GetaccDefaultAction is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDefaultAction(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5013, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccDescription is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDescription(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5005, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelp is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelp(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5008, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelpTopic is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelpTopic(self, pszHelpFile=global_Missing, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5009, 2, (3, 0), ((16392, 2), (12, 17)), u'GetaccHelpTopic', None,pszHelpFile
			, varChild)

	# The method GetaccKeyboardShortcut is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccKeyboardShortcut(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5010, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccName(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5003, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccRole is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccRole(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5006, 2, (12, 0), ((12, 17),), u'GetaccRole', None,varChild
			)

	# The method GetaccState is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccState(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5007, 2, (12, 0), ((12, 17),), u'GetaccState', None,varChild
			)

	# The method GetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccValue(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5004, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method List is actually a property, but must be used as a method to correctly pass the arguments
	def List(self, Index=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(1610940422, LCID, 2, (8, 0), ((3, 1),),Index
			)

	# Result is of type CommandBarControl
	def Move(self, Bar=defaultNamedOptArg, Before=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610874902, LCID, 1, (9, 0), ((12, 17), (12, 17)),Bar
			, Before)
		if ret is not None:
			ret = Dispatch(ret, u'Move', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	def RemoveItem(self, Index=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1610940429, LCID, 1, (24, 0), ((3, 1),),Index
			)

	def Reserved1(self):
		return self._oleobj_.InvokeTypes(1610874926, LCID, 1, (24, 0), (),)

	def Reserved2(self):
		return self._oleobj_.InvokeTypes(1610874927, LCID, 1, (24, 0), (),)

	def Reserved3(self):
		return self._oleobj_.InvokeTypes(1610874928, LCID, 1, (24, 0), (),)

	def Reserved4(self):
		return self._oleobj_.InvokeTypes(1610874929, LCID, 1, (24, 0), (),)

	def Reserved5(self):
		return self._oleobj_.InvokeTypes(1610874930, LCID, 1, (24, 0), (),)

	def Reserved6(self):
		return self._oleobj_.InvokeTypes(1610874931, LCID, 1, (24, 0), (),)

	def Reserved7(self):
		return self._oleobj_.InvokeTypes(1610874932, LCID, 1, (24, 0), (),)

	def Reset(self):
		return self._oleobj_.InvokeTypes(1610874913, LCID, 1, (24, 0), (),)

	def SetFocus(self):
		return self._oleobj_.InvokeTypes(1610874914, LCID, 1, (24, 0), (),)

	# The method SetList is actually a property, but must be used as a method to correctly pass the arguments
	def SetList(self, Index=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(1610940422, LCID, 4, (24, 0), ((3, 1), (8, 1)),Index
			, arg1)

	# The method SetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccName(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5003, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method SetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccValue(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5004, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method accChild is actually a property, but must be used as a method to correctly pass the arguments
	def accChild(self, varChild=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(-5002, LCID, 2, (9, 0), ((12, 1),),varChild
			)
		if ret is not None:
			ret = Dispatch(ret, u'accChild', None)
		return ret

	def accDoDefaultAction(self, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5018, LCID, 1, (24, 0), ((12, 17),),varChild
			)

	def accHitTest(self, xLeft=defaultNamedNotOptArg, yTop=defaultNamedNotOptArg):
		return self._ApplyTypes_(-5017, 1, (12, 0), ((3, 1), (3, 1)), u'accHitTest', None,xLeft
			, yTop)

	def accLocation(self, pxLeft=global_Missing, pyTop=global_Missing, pcxWidth=global_Missing, pcyHeight=global_Missing
			, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5015, 1, (24, 0), ((16387, 2), (16387, 2), (16387, 2), (16387, 2), (12, 17)), u'accLocation', None,pxLeft
			, pyTop, pcxWidth, pcyHeight, varChild)

	def accNavigate(self, navDir=defaultNamedNotOptArg, varStart=defaultNamedOptArg):
		return self._ApplyTypes_(-5016, 1, (12, 0), ((3, 1), (12, 17)), u'accNavigate', None,navDir
			, varStart)

	def accSelect(self, flagsSelect=defaultNamedNotOptArg, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5014, LCID, 1, (24, 0), ((3, 1), (12, 17)),flagsSelect
			, varChild)

	_prop_map_get_ = {
		"Application": (1610809344, 2, (9, 0), (), "Application", None),
		"BeginGroup": (1610874880, 2, (11, 0), (), "BeginGroup", None),
		"BuiltIn": (1610874882, 2, (11, 0), (), "BuiltIn", None),
		"Caption": (1610874883, 2, (8, 0), (), "Caption", None),
		"Control": (1610874885, 2, (9, 0), (), "Control", None),
		"Creator": (1610809345, 2, (3, 0), (), "Creator", None),
		"DescriptionText": (1610874888, 2, (8, 0), (), "DescriptionText", None),
		"DropDownLines": (1610940418, 2, (3, 0), (), "DropDownLines", None),
		"DropDownWidth": (1610940420, 2, (3, 0), (), "DropDownWidth", None),
		"Enabled": (1610874890, 2, (11, 0), (), "Enabled", None),
		"Height": (1610874893, 2, (3, 0), (), "Height", None),
		"HelpContextId": (1610874895, 2, (3, 0), (), "HelpContextId", None),
		"HelpFile": (1610874897, 2, (8, 0), (), "HelpFile", None),
		"Id": (1610874899, 2, (3, 0), (), "Id", None),
		"Index": (1610874900, 2, (3, 0), (), "Index", None),
		"InstanceId": (1610874901, 2, (3, 0), (), "InstanceId", None),
		"IsPriorityDropped": (1610874925, 2, (11, 0), (), "IsPriorityDropped", None),
		"Left": (1610874903, 2, (3, 0), (), "Left", None),
		"ListCount": (1610940424, 2, (3, 0), (), "ListCount", None),
		"ListHeaderCount": (1610940425, 2, (3, 0), (), "ListHeaderCount", None),
		"ListIndex": (1610940427, 2, (3, 0), (), "ListIndex", None),
		"OLEUsage": (1610874904, 2, (3, 0), (), "OLEUsage", None),
		"OnAction": (1610874906, 2, (8, 0), (), "OnAction", None),
		"Parameter": (1610874909, 2, (8, 0), (), "Parameter", None),
		# Method 'Parent' returns object of type 'CommandBar'
		"Parent": (1610874908, 2, (9, 0), (), "Parent", '{000C0304-0000-0000-C000-000000000046}'),
		"Priority": (1610874911, 2, (3, 0), (), "Priority", None),
		"Style": (1610940430, 2, (3, 0), (), "Style", None),
		"Tag": (1610874915, 2, (8, 0), (), "Tag", None),
		"Text": (1610940432, 2, (8, 0), (), "Text", None),
		"TooltipText": (1610874917, 2, (8, 0), (), "TooltipText", None),
		"Top": (1610874919, 2, (3, 0), (), "Top", None),
		"Type": (1610874920, 2, (3, 0), (), "Type", None),
		"Visible": (1610874921, 2, (11, 0), (), "Visible", None),
		"Width": (1610874923, 2, (3, 0), (), "Width", None),
		"accChildCount": (-5001, 2, (3, 0), (), "accChildCount", None),
		"accDefaultAction": (-5013, 2, (8, 0), ((12, 17),), "accDefaultAction", None),
		"accDescription": (-5005, 2, (8, 0), ((12, 17),), "accDescription", None),
		"accFocus": (-5011, 2, (12, 0), (), "accFocus", None),
		"accHelp": (-5008, 2, (8, 0), ((12, 17),), "accHelp", None),
		"accHelpTopic": (-5009, 2, (3, 0), ((16392, 2), (12, 17)), "accHelpTopic", None),
		"accKeyboardShortcut": (-5010, 2, (8, 0), ((12, 17),), "accKeyboardShortcut", None),
		"accName": (-5003, 2, (8, 0), ((12, 17),), "accName", None),
		"accParent": (-5000, 2, (9, 0), (), "accParent", None),
		"accRole": (-5006, 2, (12, 0), ((12, 17),), "accRole", None),
		"accSelection": (-5012, 2, (12, 0), (), "accSelection", None),
		"accState": (-5007, 2, (12, 0), ((12, 17),), "accState", None),
		"accValue": (-5004, 2, (8, 0), ((12, 17),), "accValue", None),
	}
	_prop_map_put_ = {
		"BeginGroup": ((1610874880, LCID, 4, 0),()),
		"Caption": ((1610874883, LCID, 4, 0),()),
		"DescriptionText": ((1610874888, LCID, 4, 0),()),
		"DropDownLines": ((1610940418, LCID, 4, 0),()),
		"DropDownWidth": ((1610940420, LCID, 4, 0),()),
		"Enabled": ((1610874890, LCID, 4, 0),()),
		"Height": ((1610874893, LCID, 4, 0),()),
		"HelpContextId": ((1610874895, LCID, 4, 0),()),
		"HelpFile": ((1610874897, LCID, 4, 0),()),
		"ListHeaderCount": ((1610940425, LCID, 4, 0),()),
		"ListIndex": ((1610940427, LCID, 4, 0),()),
		"OLEUsage": ((1610874904, LCID, 4, 0),()),
		"OnAction": ((1610874906, LCID, 4, 0),()),
		"Parameter": ((1610874909, LCID, 4, 0),()),
		"Priority": ((1610874911, LCID, 4, 0),()),
		"Style": ((1610940430, LCID, 4, 0),()),
		"Tag": ((1610874915, LCID, 4, 0),()),
		"Text": ((1610940432, LCID, 4, 0),()),
		"TooltipText": ((1610874917, LCID, 4, 0),()),
		"Visible": ((1610874921, LCID, 4, 0),()),
		"Width": ((1610874923, LCID, 4, 0),()),
		"accName": ((-5003, LCID, 4, 0),()),
		"accValue": ((-5004, LCID, 4, 0),()),
	}

class _CommandBarComboBoxEvents:
	CLSID = CLSID_Sink = IID('{000C0354-0000-0000-C000-000000000046}')
	coclass_clsid = IID('{55F88897-7708-11D1-ACEB-006008961DA5}')
	_public_methods_ = [] # For COM Server support
	_dispid_to_func_ = {
		1610678275 : "OnInvoke",
		1610678273 : "OnGetTypeInfo",
		1610612737 : "OnAddRef",
		1610612736 : "OnQueryInterface",
		1610612738 : "OnRelease",
		1610678274 : "OnGetIDsOfNames",
		1610678272 : "OnGetTypeInfoCount",
		        1 : "OnChange",
		}

	def __init__(self, oobj = None):
		if oobj is None:
			self._olecp = None
		else:
			import win32com.server.util
			from win32com.server.policy import EventHandlerPolicy
			cpc=oobj._oleobj_.QueryInterface(global_IID_IConnectionPointContainer)
			cp=cpc.FindConnectionPoint(self.CLSID_Sink)
			cookie=cp.Advise(win32com.server.util.wrap(self, usePolicy=EventHandlerPolicy))
			self._olecp,self._olecp_cookie = cp,cookie
	def __del__(self):
		try:
			self.close()
		except global_com_error:
			pass
	def close(self):
		if self._olecp is not None:
			cp,cookie,self._olecp,self._olecp_cookie = self._olecp,self._olecp_cookie,None,None
			cp.Unadvise(cookie)
	def _query_interface_(self, iid):
		import win32com.server.util
		if iid==self.CLSID_Sink: return win32com.server.util.wrap(self)

	# Event Handlers
	# If you create handlers, they should have the following prototypes:
#	def OnInvoke(self, dispidMember=defaultNamedNotOptArg, riid=defaultNamedNotOptArg, lcid=defaultNamedNotOptArg, wFlags=defaultNamedNotOptArg
#			, pdispparams=defaultNamedNotOptArg, pvarResult=global_Missing, pexcepinfo=global_Missing, puArgErr=global_Missing):
#	def OnGetTypeInfo(self, itinfo=defaultNamedNotOptArg, lcid=defaultNamedNotOptArg, pptinfo=global_Missing):
#	def OnAddRef(self):
#	def OnQueryInterface(self, riid=defaultNamedNotOptArg, ppvObj=global_Missing):
#	def OnRelease(self):
#	def OnGetIDsOfNames(self, riid=defaultNamedNotOptArg, rgszNames=defaultNamedNotOptArg, cNames=defaultNamedNotOptArg, lcid=defaultNamedNotOptArg
#			, rgdispid=global_Missing):
#	def OnGetTypeInfoCount(self, pctinfo=global_Missing):
#	def OnChange(self, Ctrl=defaultNamedNotOptArg):


class _CommandBars(DispatchBaseClass):
	CLSID = IID('{000C0302-0000-0000-C000-000000000046}')
	coclass_clsid = IID('{55F88893-7708-11D1-ACEB-006008961DA5}')

	# Result is of type CommandBar
	def Add(self, Name=defaultNamedOptArg, Position=defaultNamedOptArg, MenuBar=defaultNamedOptArg, Temporary=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610809346, LCID, 1, (9, 0), ((12, 17), (12, 17), (12, 17), (12, 17)),Name
			, Position, MenuBar, Temporary)
		if ret is not None:
			ret = Dispatch(ret, u'Add', '{000C0304-0000-0000-C000-000000000046}')
		return ret

	# Result is of type CommandBar
	def AddEx(self, TbidOrName=defaultNamedOptArg, Position=defaultNamedOptArg, MenuBar=defaultNamedOptArg, Temporary=defaultNamedOptArg
			, TbtrProtection=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610809366, LCID, 1, (9, 0), ((12, 17), (12, 17), (12, 17), (12, 17), (12, 17)),TbidOrName
			, Position, MenuBar, Temporary, TbtrProtection)
		if ret is not None:
			ret = Dispatch(ret, u'AddEx', '{000C0304-0000-0000-C000-000000000046}')
		return ret

	# Result is of type CommandBarControl
	def FindControl(self, Type=defaultNamedOptArg, Id=defaultNamedOptArg, Tag=defaultNamedOptArg, Visible=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610809352, LCID, 1, (9, 0), ((12, 17), (12, 17), (12, 17), (12, 17)),Type
			, Id, Tag, Visible)
		if ret is not None:
			ret = Dispatch(ret, u'FindControl', '{000C0308-0000-0000-C000-000000000046}')
		return ret

	# Result is of type CommandBarControls
	def FindControls(self, Type=defaultNamedOptArg, Id=defaultNamedOptArg, Tag=defaultNamedOptArg, Visible=defaultNamedOptArg):
		ret = self._oleobj_.InvokeTypes(1610809365, LCID, 1, (9, 0), ((12, 17), (12, 17), (12, 17), (12, 17)),Type
			, Id, Tag, Visible)
		if ret is not None:
			ret = Dispatch(ret, u'FindControls', '{000C0306-0000-0000-C000-000000000046}')
		return ret

	# The method IdsString is actually a property, but must be used as a method to correctly pass the arguments
	def IdsString(self, ids=defaultNamedNotOptArg, pbstrName=global_Missing):
		return self._ApplyTypes_(1610809361, 2, (3, 0), ((3, 1), (16392, 2)), u'IdsString', None,ids
			, pbstrName)

	# Result is of type CommandBar
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{000C0304-0000-0000-C000-000000000046}')
		return ret

	def ReleaseFocus(self):
		return self._oleobj_.InvokeTypes(1610809360, LCID, 1, (24, 0), (),)

	# The method TmcGetName is actually a property, but must be used as a method to correctly pass the arguments
	def TmcGetName(self, tmc=defaultNamedNotOptArg, pbstrName=global_Missing):
		return self._ApplyTypes_(1610809362, 2, (3, 0), ((3, 1), (16392, 2)), u'TmcGetName', None,tmc
			, pbstrName)

	_prop_map_get_ = {
		# Method 'ActionControl' returns object of type 'CommandBarControl'
		"ActionControl": (1610809344, 2, (9, 0), (), "ActionControl", '{000C0308-0000-0000-C000-000000000046}'),
		# Method 'ActiveMenuBar' returns object of type 'CommandBar'
		"ActiveMenuBar": (1610809345, 2, (9, 0), (), "ActiveMenuBar", '{000C0304-0000-0000-C000-000000000046}'),
		"AdaptiveMenus": (1610809363, 2, (11, 0), (), "AdaptiveMenus", None),
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Count": (1610809347, 2, (3, 0), (), "Count", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
		"DisableAskAQuestionDropdown": (1610809371, 2, (11, 0), (), "DisableAskAQuestionDropdown", None),
		"DisableCustomize": (1610809369, 2, (11, 0), (), "DisableCustomize", None),
		"DisplayFonts": (1610809367, 2, (11, 0), (), "DisplayFonts", None),
		"DisplayKeysInTooltips": (1610809350, 2, (11, 0), (), "DisplayKeysInTooltips", None),
		"DisplayTooltips": (1610809348, 2, (11, 0), (), "DisplayTooltips", None),
		"LargeButtons": (1610809354, 2, (11, 0), (), "LargeButtons", None),
		"MenuAnimationStyle": (1610809356, 2, (3, 0), (), "MenuAnimationStyle", None),
		"Parent": (1610809359, 2, (9, 0), (), "Parent", None),
	}
	_prop_map_put_ = {
		"AdaptiveMenus": ((1610809363, LCID, 4, 0),()),
		"DisableAskAQuestionDropdown": ((1610809371, LCID, 4, 0),()),
		"DisableCustomize": ((1610809369, LCID, 4, 0),()),
		"DisplayFonts": ((1610809367, LCID, 4, 0),()),
		"DisplayKeysInTooltips": ((1610809350, LCID, 4, 0),()),
		"DisplayTooltips": ((1610809348, LCID, 4, 0),()),
		"LargeButtons": ((1610809354, LCID, 4, 0),()),
		"MenuAnimationStyle": ((1610809356, LCID, 4, 0),()),
	}
	# Default method for this class is 'Item'
	def __call__(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{000C0304-0000-0000-C000-000000000046}')
		return ret

	# str(ob) and int(ob) will use __call__
	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except global_com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		return win32com.client.util.Iterator(ob, '{000C0304-0000-0000-C000-000000000046}')
	def _NewEnum(self):
		"Create an enumerator from this object"
		return win32com.client.util.WrapEnum(self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),()),'{000C0304-0000-0000-C000-000000000046}')
	def __getitem__(self, index):
		"Allow this class to be accessed as a collection"
		if '_enum_' not in self.__dict__:
			self.__dict__['_enum_'] = self._NewEnum()
		return self._enum_.__getitem__(index)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1610809347, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class _CommandBarsEvents:
	CLSID = CLSID_Sink = IID('{000C0352-0000-0000-C000-000000000046}')
	coclass_clsid = IID('{55F88893-7708-11D1-ACEB-006008961DA5}')
	_public_methods_ = [] # For COM Server support
	_dispid_to_func_ = {
		1610678275 : "OnInvoke",
		1610678273 : "OnGetTypeInfo",
		        1 : "OnUpdate",
		1610612737 : "OnAddRef",
		1610612736 : "OnQueryInterface",
		1610612738 : "OnRelease",
		1610678274 : "OnGetIDsOfNames",
		1610678272 : "OnGetTypeInfoCount",
		}

	def __init__(self, oobj = None):
		if oobj is None:
			self._olecp = None
		else:
			import win32com.server.util
			from win32com.server.policy import EventHandlerPolicy
			cpc=oobj._oleobj_.QueryInterface(global_IID_IConnectionPointContainer)
			cp=cpc.FindConnectionPoint(self.CLSID_Sink)
			cookie=cp.Advise(win32com.server.util.wrap(self, usePolicy=EventHandlerPolicy))
			self._olecp,self._olecp_cookie = cp,cookie
	def __del__(self):
		try:
			self.close()
		except global_com_error:
			pass
	def close(self):
		if self._olecp is not None:
			cp,cookie,self._olecp,self._olecp_cookie = self._olecp,self._olecp_cookie,None,None
			cp.Unadvise(cookie)
	def _query_interface_(self, iid):
		import win32com.server.util
		if iid==self.CLSID_Sink: return win32com.server.util.wrap(self)

	# Event Handlers
	# If you create handlers, they should have the following prototypes:
#	def OnInvoke(self, dispidMember=defaultNamedNotOptArg, riid=defaultNamedNotOptArg, lcid=defaultNamedNotOptArg, wFlags=defaultNamedNotOptArg
#			, pdispparams=defaultNamedNotOptArg, pvarResult=global_Missing, pexcepinfo=global_Missing, puArgErr=global_Missing):
#	def OnGetTypeInfo(self, itinfo=defaultNamedNotOptArg, lcid=defaultNamedNotOptArg, pptinfo=global_Missing):
#	def OnUpdate(self):
#	def OnAddRef(self):
#	def OnQueryInterface(self, riid=defaultNamedNotOptArg, ppvObj=global_Missing):
#	def OnRelease(self):
#	def OnGetIDsOfNames(self, riid=defaultNamedNotOptArg, rgszNames=defaultNamedNotOptArg, cNames=defaultNamedNotOptArg, lcid=defaultNamedNotOptArg
#			, rgdispid=global_Missing):
#	def OnGetTypeInfoCount(self, pctinfo=global_Missing):


class _IMsoDispObj(DispatchBaseClass):
	CLSID = IID('{000C0300-0000-0000-C000-000000000046}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Application": (1610743808, 2, (9, 0), (), "Application", None),
		"Creator": (1610743809, 2, (3, 0), (), "Creator", None),
	}
	_prop_map_put_ = {
	}

class _IMsoOleAccDispObj(DispatchBaseClass):
	CLSID = IID('{000C0301-0000-0000-C000-000000000046}')
	coclass_clsid = None

	# The method GetaccDefaultAction is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDefaultAction(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5013, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccDescription is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccDescription(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5005, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelp is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelp(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5008, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccHelpTopic is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccHelpTopic(self, pszHelpFile=global_Missing, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5009, 2, (3, 0), ((16392, 2), (12, 17)), u'GetaccHelpTopic', None,pszHelpFile
			, varChild)

	# The method GetaccKeyboardShortcut is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccKeyboardShortcut(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5010, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccName(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5003, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method GetaccRole is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccRole(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5006, 2, (12, 0), ((12, 17),), u'GetaccRole', None,varChild
			)

	# The method GetaccState is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccState(self, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5007, 2, (12, 0), ((12, 17),), u'GetaccState', None,varChild
			)

	# The method GetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def GetaccValue(self, varChild=defaultNamedOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(-5004, LCID, 2, (8, 0), ((12, 17),),varChild
			)

	# The method SetaccName is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccName(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5003, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method SetaccValue is actually a property, but must be used as a method to correctly pass the arguments
	def SetaccValue(self, varChild=defaultNamedNotOptArg, arg1=defaultUnnamedArg):
		return self._oleobj_.InvokeTypes(-5004, LCID, 4, (24, 0), ((12, 17), (8, 1)),varChild
			, arg1)

	# The method accChild is actually a property, but must be used as a method to correctly pass the arguments
	def accChild(self, varChild=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(-5002, LCID, 2, (9, 0), ((12, 1),),varChild
			)
		if ret is not None:
			ret = Dispatch(ret, u'accChild', None)
		return ret

	def accDoDefaultAction(self, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5018, LCID, 1, (24, 0), ((12, 17),),varChild
			)

	def accHitTest(self, xLeft=defaultNamedNotOptArg, yTop=defaultNamedNotOptArg):
		return self._ApplyTypes_(-5017, 1, (12, 0), ((3, 1), (3, 1)), u'accHitTest', None,xLeft
			, yTop)

	def accLocation(self, pxLeft=global_Missing, pyTop=global_Missing, pcxWidth=global_Missing, pcyHeight=global_Missing
			, varChild=defaultNamedOptArg):
		return self._ApplyTypes_(-5015, 1, (24, 0), ((16387, 2), (16387, 2), (16387, 2), (16387, 2), (12, 17)), u'accLocation', None,pxLeft
			, pyTop, pcxWidth, pcyHeight, varChild)

	def accNavigate(self, navDir=defaultNamedNotOptArg, varStart=defaultNamedOptArg):
		return self._ApplyTypes_(-5016, 1, (12, 0), ((3, 1), (12, 17)), u'accNavigate', None,navDir
			, varStart)

	def accSelect(self, flagsSelect=defaultNamedNotOptArg, varChild=defaultNamedOptArg):
		return self._oleobj_.InvokeTypes(-5014, LCID, 1, (24, 0), ((3, 1), (12, 17)),flagsSelect
			, varChild)

	_prop_map_get_ = {
		"Application": (1610809344, 2, (9, 0), (), "Application", None),
		"Creator": (1610809345, 2, (3, 0), (), "Creator", None),
		"accChildCount": (-5001, 2, (3, 0), (), "accChildCount", None),
		"accDefaultAction": (-5013, 2, (8, 0), ((12, 17),), "accDefaultAction", None),
		"accDescription": (-5005, 2, (8, 0), ((12, 17),), "accDescription", None),
		"accFocus": (-5011, 2, (12, 0), (), "accFocus", None),
		"accHelp": (-5008, 2, (8, 0), ((12, 17),), "accHelp", None),
		"accHelpTopic": (-5009, 2, (3, 0), ((16392, 2), (12, 17)), "accHelpTopic", None),
		"accKeyboardShortcut": (-5010, 2, (8, 0), ((12, 17),), "accKeyboardShortcut", None),
		"accName": (-5003, 2, (8, 0), ((12, 17),), "accName", None),
		"accParent": (-5000, 2, (9, 0), (), "accParent", None),
		"accRole": (-5006, 2, (12, 0), ((12, 17),), "accRole", None),
		"accSelection": (-5012, 2, (12, 0), (), "accSelection", None),
		"accState": (-5007, 2, (12, 0), ((12, 17),), "accState", None),
		"accValue": (-5004, 2, (8, 0), ((12, 17),), "accValue", None),
	}
	_prop_map_put_ = {
		"accName": ((-5003, LCID, 4, 0),()),
		"accValue": ((-5004, LCID, 4, 0),()),
	}

from win32com.client import CoClassBaseClass
class CommandBarButton(CoClassBaseClass): # A CoClass
	CLSID = IID('{55F88891-7708-11D1-ACEB-006008961DA5}')
	coclass_sources = [
		_CommandBarButtonEvents,
	]
	default_source = _CommandBarButtonEvents
	coclass_interfaces = [
		_CommandBarButton,
	]
	default_interface = _CommandBarButton

class CommandBarComboBox(CoClassBaseClass): # A CoClass
	CLSID = IID('{55F88897-7708-11D1-ACEB-006008961DA5}')
	coclass_sources = [
		_CommandBarComboBoxEvents,
	]
	default_source = _CommandBarComboBoxEvents
	coclass_interfaces = [
		_CommandBarComboBox,
	]
	default_interface = _CommandBarComboBox

class CommandBars(CoClassBaseClass): # A CoClass
	CLSID = IID('{55F88893-7708-11D1-ACEB-006008961DA5}')
	coclass_sources = [
		_CommandBarsEvents,
	]
	default_source = _CommandBarsEvents
	coclass_interfaces = [
		_CommandBars,
	]
	default_interface = _CommandBars

class MsoEnvelope(CoClassBaseClass): # A CoClass
	CLSID = IID('{0006F01A-0000-0000-C000-000000000046}')
	coclass_sources = [
		IMsoEnvelopeVBEvents,
	]
	default_source = IMsoEnvelopeVBEvents
	coclass_interfaces = [
		IMsoEnvelopeVB,
	]
	default_interface = IMsoEnvelopeVB

Adjustments_vtables_dispatch_ = 1
Adjustments_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'Count' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'Val' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'Val' , ), 0, (0, (), [ (3, 1, None, None) ,
			(4, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
]

AnswerWizard_vtables_dispatch_ = 1
AnswerWizard_vtables_ = [
	(( u'Parent' , u'ppidisp' , ), 1610809344, (1610809344, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Files' , u'Files' , ), 1610809345, (1610809345, (), [ (16393, 10, None, "IID('{000C0361-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'ClearFileList' , ), 1610809346, (1610809346, (), [ ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ResetFileList' , ), 1610809347, (1610809347, (), [ ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
]

AnswerWizardFiles_vtables_dispatch_ = 1
AnswerWizardFiles_vtables_ = [
	(( u'Parent' , u'ppidisp' , ), 1610809344, (1610809344, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'pbstr' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pCount' , ), 1610809346, (1610809346, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'FileName' , ), 1610809347, (1610809347, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , u'FileName' , ), 1610809348, (1610809348, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

Assistant_vtables_dispatch_ = 1
Assistant_vtables_ = [
	(( u'Parent' , u'ppidisp' , ), 1610809344, (1610809344, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Move' , u'xLeft' , u'yTop' , ), 1610809345, (1610809345, (), [ (3, 1, None, None) ,
			(3, 1, None, None) , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Top' , u'pyTop' , ), 1610809346, (1610809346, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Top' , u'pyTop' , ), 1610809346, (1610809346, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Left' , u'pxLeft' , ), 1610809348, (1610809348, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Left' , u'pxLeft' , ), 1610809348, (1610809348, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Help' , ), 1610809350, (1610809350, (), [ ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'StartWizard' , u'On' , u'Callback' , u'PrivateX' , u'Animation' ,
			u'CustomTeaser' , u'Top' , u'Left' , u'Bottom' , u'Right' ,
			u'plWizID' , ), 1610809351, (1610809351, (), [ (11, 1, None, None) , (8, 1, None, None) , (3, 1, None, None) ,
			(12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) ,
			(12, 17, None, None) , (16387, 10, None, None) , ], 1 , 1 , 4 , 6 , 64 , (3, 0, None, None) , 0 , )),
	(( u'EndWizard' , u'WizardID' , u'varfSuccess' , u'Animation' , ), 1610809352, (1610809352, (), [
			(3, 1, None, None) , (11, 1, None, None) , (12, 17, None, None) , ], 1 , 1 , 4 , 1 , 68 , (3, 0, None, None) , 0 , )),
	(( u'ActivateWizard' , u'WizardID' , u'act' , u'Animation' , ), 1610809353, (1610809353, (), [
			(3, 1, None, None) , (3, 1, None, None) , (12, 17, None, None) , ], 1 , 1 , 4 , 1 , 72 , (3, 0, None, None) , 0 , )),
	(( u'ResetTips' , ), 1610809354, (1610809354, (), [ ], 1 , 1 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'NewBalloon' , u'ppibal' , ), 1610809355, (1610809355, (), [ (16393, 10, None, "IID('{000C0324-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'BalloonError' , u'pbne' , ), 1610809356, (1610809356, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'pvarfVisible' , ), 1610809357, (1610809357, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'pvarfVisible' , ), 1610809357, (1610809357, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'Animation' , u'pfca' , ), 1610809359, (1610809359, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'Animation' , u'pfca' , ), 1610809359, (1610809359, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'Reduced' , u'pvarfReduced' , ), 1610809361, (1610809361, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'Reduced' , u'pvarfReduced' , ), 1610809361, (1610809361, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'AssistWithHelp' , u'pvarfAssistWithHelp' , ), 1610809363, (1610809363, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'AssistWithHelp' , u'pvarfAssistWithHelp' , ), 1610809363, (1610809363, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'AssistWithWizards' , u'pvarfAssistWithWizards' , ), 1610809365, (1610809365, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'AssistWithWizards' , u'pvarfAssistWithWizards' , ), 1610809365, (1610809365, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'AssistWithAlerts' , u'pvarfAssistWithAlerts' , ), 1610809367, (1610809367, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'AssistWithAlerts' , u'pvarfAssistWithAlerts' , ), 1610809367, (1610809367, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'MoveWhenInTheWay' , u'pvarfMove' , ), 1610809369, (1610809369, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'MoveWhenInTheWay' , u'pvarfMove' , ), 1610809369, (1610809369, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
	(( u'Sounds' , u'pvarfSounds' , ), 1610809371, (1610809371, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 144 , (3, 0, None, None) , 0 , )),
	(( u'Sounds' , u'pvarfSounds' , ), 1610809371, (1610809371, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 148 , (3, 0, None, None) , 0 , )),
	(( u'FeatureTips' , u'pvarfFeatures' , ), 1610809373, (1610809373, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 152 , (3, 0, None, None) , 0 , )),
	(( u'FeatureTips' , u'pvarfFeatures' , ), 1610809373, (1610809373, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 156 , (3, 0, None, None) , 0 , )),
	(( u'MouseTips' , u'pvarfMouse' , ), 1610809375, (1610809375, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 160 , (3, 0, None, None) , 0 , )),
	(( u'MouseTips' , u'pvarfMouse' , ), 1610809375, (1610809375, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 164 , (3, 0, None, None) , 0 , )),
	(( u'KeyboardShortcutTips' , u'pvarfKeyboardShortcuts' , ), 1610809377, (1610809377, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 168 , (3, 0, None, None) , 0 , )),
	(( u'KeyboardShortcutTips' , u'pvarfKeyboardShortcuts' , ), 1610809377, (1610809377, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 172 , (3, 0, None, None) , 0 , )),
	(( u'HighPriorityTips' , u'pvarfHighPriorityTips' , ), 1610809379, (1610809379, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 176 , (3, 0, None, None) , 0 , )),
	(( u'HighPriorityTips' , u'pvarfHighPriorityTips' , ), 1610809379, (1610809379, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 180 , (3, 0, None, None) , 0 , )),
	(( u'TipOfDay' , u'pvarfTipOfDay' , ), 1610809381, (1610809381, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 184 , (3, 0, None, None) , 0 , )),
	(( u'TipOfDay' , u'pvarfTipOfDay' , ), 1610809381, (1610809381, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 188 , (3, 0, None, None) , 0 , )),
	(( u'GuessHelp' , u'pvarfGuessHelp' , ), 1610809383, (1610809383, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 192 , (3, 0, None, None) , 0 , )),
	(( u'GuessHelp' , u'pvarfGuessHelp' , ), 1610809383, (1610809383, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 196 , (3, 0, None, None) , 0 , )),
	(( u'SearchWhenProgramming' , u'pvarfSearchInProgram' , ), 1610809385, (1610809385, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 200 , (3, 0, None, None) , 0 , )),
	(( u'SearchWhenProgramming' , u'pvarfSearchInProgram' , ), 1610809385, (1610809385, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 204 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'pbstrName' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 208 , (3, 0, None, None) , 0 , )),
	(( u'FileName' , u'pbstr' , ), 1610809388, (1610809388, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 212 , (3, 0, None, None) , 0 , )),
	(( u'FileName' , u'pbstr' , ), 1610809388, (1610809388, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 216 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'pbstrName' , ), 1610809390, (1610809390, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 220 , (3, 0, None, None) , 0 , )),
	(( u'On' , u'pvarfOn' , ), 1610809391, (1610809391, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 224 , (3, 0, None, None) , 0 , )),
	(( u'On' , u'pvarfOn' , ), 1610809391, (1610809391, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 228 , (3, 0, None, None) , 0 , )),
	(( u'DoAlert' , u'bstrAlertTitle' , u'bstrAlertText' , u'alb' , u'alc' ,
			u'ald' , u'alq' , u'varfSysAlert' , u'pibtn' , ), 1610809393, (1610809393, (), [
			(8, 1, None, None) , (8, 1, None, None) , (3, 1, None, None) , (3, 1, None, None) , (3, 1, None, None) ,
			(3, 1, None, None) , (11, 1, None, None) , (16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 232 , (3, 0, None, None) , 0 , )),
]

Balloon_vtables_dispatch_ = 1
Balloon_vtables_ = [
	(( u'Parent' , u'ppidisp' , ), 1610809344, (1610809344, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Checkboxes' , u'ppidisp' , ), 1610809345, (1610809345, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Labels' , u'ppidisp' , ), 1610809346, (1610809346, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'BalloonType' , u'pbty' , ), 1610809347, (1610809347, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'BalloonType' , u'pbty' , ), 1610809347, (1610809347, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Icon' , u'picn' , ), 1610809349, (1610809349, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Icon' , u'picn' , ), 1610809349, (1610809349, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Heading' , u'pbstr' , ), 1610809351, (1610809351, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Heading' , u'pbstr' , ), 1610809351, (1610809351, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'pbstr' , ), 1610809353, (1610809353, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'pbstr' , ), 1610809353, (1610809353, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'Mode' , u'pmd' , ), 1610809355, (1610809355, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Mode' , u'pmd' , ), 1610809355, (1610809355, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'Animation' , u'pfca' , ), 1610809357, (1610809357, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'Animation' , u'pfca' , ), 1610809357, (1610809357, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'Button' , u'psbs' , ), 1610809359, (1610809359, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'Button' , u'psbs' , ), 1610809359, (1610809359, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'Callback' , u'pbstr' , ), 1610809361, (1610809361, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'Callback' , u'pbstr' , ), 1610809361, (1610809361, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'Private' , u'plPrivate' , ), 1610809363, (1610809363, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'Private' , u'plPrivate' , ), 1610809363, (1610809363, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'SetAvoidRectangle' , u'Left' , u'Top' , u'Right' , u'Bottom' ,
			), 1610809365, (1610809365, (), [ (3, 1, None, None) , (3, 1, None, None) , (3, 1, None, None) , (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'pbstrName' , ), 1610809366, (1610809366, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'Show' , u'pibtn' , ), 1610809367, (1610809367, (), [ (16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'Close' , ), 1610809368, (1610809368, (), [ ], 1 , 1 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
]

BalloonCheckbox_vtables_dispatch_ = 1
BalloonCheckbox_vtables_ = [
	(( u'Item' , u'pbstrName' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'pbstrName' , ), 1610809345, (1610809345, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 1610809346, (1610809346, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Checked' , u'pvarfChecked' , ), 1610809347, (1610809347, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Checked' , u'pvarfChecked' , ), 1610809347, (1610809347, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'pbstr' , ), 1610809349, (1610809349, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'pbstr' , ), 1610809349, (1610809349, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
]

BalloonCheckboxes_vtables_dispatch_ = 1
BalloonCheckboxes_vtables_ = [
	(( u'Name' , u'pbstrName' , ), 1610809344, (1610809344, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 1610809345, (1610809345, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'ppidisp' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pccbx' , ), 1610809347, (1610809347, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pccbx' , ), 1610809347, (1610809347, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 1024 , )),
]

BalloonLabel_vtables_dispatch_ = 1
BalloonLabel_vtables_ = [
	(( u'Item' , u'pbstrName' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'pbstrName' , ), 1610809345, (1610809345, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 1610809346, (1610809346, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'pbstr' , ), 1610809347, (1610809347, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'pbstr' , ), 1610809347, (1610809347, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

BalloonLabels_vtables_dispatch_ = 1
BalloonLabels_vtables_ = [
	(( u'Name' , u'pbstrName' , ), 1610809344, (1610809344, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 1610809345, (1610809345, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'ppidisp' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pcwz' , ), 1610809347, (1610809347, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pcwz' , ), 1610809347, (1610809347, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 1024 , )),
]

COMAddIn_vtables_dispatch_ = 1
COMAddIn_vtables_ = [
	(( u'Description' , u'RetValue' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Description' , u'RetValue' , ), 0, (0, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'ProgId' , u'RetValue' , ), 3, (3, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Guid' , u'RetValue' , ), 4, (4, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Connect' , u'RetValue' , ), 6, (6, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Connect' , u'RetValue' , ), 6, (6, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Object' , u'RetValue' , ), 7, (7, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Object' , u'RetValue' , ), 7, (7, (), [ (9, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'retval' , ), 8, (8, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
]

COMAddIns_vtables_dispatch_ = 1
COMAddIns_vtables_ = [
	(( u'Item' , u'Index' , u'RetValue' , ), 0, (0, (), [ (16396, 1, None, None) ,
			(16393, 10, None, "IID('{000C033A-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'RetValue' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'RetValue' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 1025 , )),
	(( u'Update' , ), 2, (2, (), [ ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 3, (3, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'SetAppModal' , u'varfModal' , ), 4, (4, (), [ (11, 1, None, None) , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 1088 , )),
]

CalloutFormat_vtables_dispatch_ = 1
CalloutFormat_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'AutomaticLength' , ), 10, (10, (), [ ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'CustomDrop' , u'Drop' , ), 11, (11, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'CustomLength' , u'Length' , ), 12, (12, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'PresetDrop' , u'DropType' , ), 13, (13, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Accent' , u'Accent' , ), 100, (100, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Accent' , u'Accent' , ), 100, (100, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Angle' , u'Angle' , ), 101, (101, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Angle' , u'Angle' , ), 101, (101, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'AutoAttach' , u'AutoAttach' , ), 102, (102, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'AutoAttach' , u'AutoAttach' , ), 102, (102, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'AutoLength' , u'AutoLength' , ), 103, (103, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Border' , u'Border' , ), 104, (104, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'Border' , u'Border' , ), 104, (104, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'Drop' , u'Drop' , ), 105, (105, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'DropType' , u'DropType' , ), 106, (106, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'Gap' , u'Gap' , ), 107, (107, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'Gap' , u'Gap' , ), 107, (107, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'Length' , u'Length' , ), 108, (108, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'Type' , ), 109, (109, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'Type' , ), 109, (109, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
]

CanvasShapes_vtables_dispatch_ = 1
CanvasShapes_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'Count' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'Item' , ), 0, (0, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'_NewEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 1024 , )),
	(( u'AddCallout' , u'Type' , u'Left' , u'Top' , u'Width' ,
			u'Height' , u'Callout' , ), 10, (10, (), [ (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'AddConnector' , u'Type' , u'BeginX' , u'BeginY' , u'EndX' ,
			u'EndY' , u'Connector' , ), 11, (11, (), [ (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'AddCurve' , u'SafeArrayOfPoints' , u'Curve' , ), 12, (12, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'AddLabel' , u'Orientation' , u'Left' , u'Top' , u'Width' ,
			u'Height' , u'Label' , ), 13, (13, (), [ (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'AddLine' , u'BeginX' , u'BeginY' , u'EndX' , u'EndY' ,
			u'Line' , ), 14, (14, (), [ (4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'AddPicture' , u'FileName' , u'LinkToFile' , u'SaveWithDocument' , u'Left' ,
			u'Top' , u'Width' , u'Height' , u'Picture' , ), 15, (15, (), [
			(8, 1, None, None) , (3, 1, None, None) , (3, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) ,
			(4, 49, '-1.0', None) , (4, 49, '-1.0', None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'AddPolyline' , u'SafeArrayOfPoints' , u'Polyline' , ), 16, (16, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'AddShape' , u'Type' , u'Left' , u'Top' , u'Width' ,
			u'Height' , u'Shape' , ), 17, (17, (), [ (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'AddTextEffect' , u'PresetTextEffect' , u'Text' , u'FontName' , u'FontSize' ,
			u'FontBold' , u'FontItalic' , u'Left' , u'Top' , u'TextEffect' ,
			), 18, (18, (), [ (3, 1, None, None) , (8, 1, None, None) , (8, 1, None, None) , (4, 1, None, None) ,
			(3, 1, None, None) , (3, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'AddTextbox' , u'Orientation' , u'Left' , u'Top' , u'Width' ,
			u'Height' , u'Textbox' , ), 19, (19, (), [ (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'BuildFreeform' , u'EditingType' , u'X1' , u'Y1' , u'FreeformBuilder' ,
			), 20, (20, (), [ (3, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C0315-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'Range' , u'Index' , u'Range' , ), 21, (21, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C031D-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'SelectAll' , ), 22, (22, (), [ ], 1 , 1 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'Background' , u'Background' , ), 100, (100, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
]

ColorFormat_vtables_dispatch_ = 1
ColorFormat_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'RGB' , u'RGB' , ), 0, (0, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'RGB' , u'RGB' , ), 0, (0, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'SchemeColor' , u'SchemeColor' , ), 100, (100, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'SchemeColor' , u'SchemeColor' , ), 100, (100, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'Type' , ), 101, (101, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'TintAndShade' , u'pValue' , ), 103, (103, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'TintAndShade' , u'pValue' , ), 103, (103, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
]

CommandBar_vtables_dispatch_ = 1
CommandBar_vtables_ = [
	(( u'BuiltIn' , u'pvarfBuiltIn' , ), 1610874880, (1610874880, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'Context' , u'pbstrContext' , ), 1610874881, (1610874881, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'Context' , u'pbstrContext' , ), 1610874881, (1610874881, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'Controls' , u'ppcbcs' , ), 1610874883, (1610874883, (), [ (16393, 10, None, "IID('{000C0306-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 1610874884, (1610874884, (), [ ], 1 , 1 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'Enabled' , u'pvarfEnabled' , ), 1610874885, (1610874885, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
	(( u'Enabled' , u'pvarfEnabled' , ), 1610874885, (1610874885, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 144 , (3, 0, None, None) , 0 , )),
	(( u'FindControl' , u'Type' , u'Id' , u'Tag' , u'Visible' ,
			u'Recursive' , u'ppcbc' , ), 1610874887, (1610874887, (), [ (12, 17, None, None) , (12, 17, None, None) ,
			(12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) , (16393, 10, None, "IID('{000C0308-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 5 , 148 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'pdy' , ), 1610874888, (1610874888, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 152 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'pdy' , ), 1610874888, (1610874888, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 156 , (3, 0, None, None) , 0 , )),
	(( u'Index' , u'pi' , ), 1610874890, (1610874890, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 160 , (3, 0, None, None) , 0 , )),
	(( u'InstanceId' , u'pid' , ), 1610874891, (1610874891, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 164 , (3, 0, None, None) , 64 , )),
	(( u'Left' , u'pxpLeft' , ), 1610874892, (1610874892, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 168 , (3, 0, None, None) , 0 , )),
	(( u'Left' , u'pxpLeft' , ), 1610874892, (1610874892, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 172 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'pbstrName' , ), 1610874894, (1610874894, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 176 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'pbstrName' , ), 1610874894, (1610874894, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 180 , (3, 0, None, None) , 0 , )),
	(( u'NameLocal' , u'pbstrNameLocal' , ), 1610874896, (1610874896, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 184 , (3, 0, None, None) , 0 , )),
	(( u'NameLocal' , u'pbstrNameLocal' , ), 1610874896, (1610874896, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 188 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 1610874898, (1610874898, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 192 , (3, 0, None, None) , 0 , )),
	(( u'Position' , u'ppos' , ), 1610874899, (1610874899, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 196 , (3, 0, None, None) , 0 , )),
	(( u'Position' , u'ppos' , ), 1610874899, (1610874899, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 200 , (3, 0, None, None) , 0 , )),
	(( u'RowIndex' , u'piRow' , ), 1610874901, (1610874901, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 204 , (3, 0, None, None) , 0 , )),
	(( u'RowIndex' , u'piRow' , ), 1610874901, (1610874901, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 208 , (3, 0, None, None) , 0 , )),
	(( u'Protection' , u'pprot' , ), 1610874903, (1610874903, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 212 , (3, 0, None, None) , 0 , )),
	(( u'Protection' , u'pprot' , ), 1610874903, (1610874903, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 216 , (3, 0, None, None) , 0 , )),
	(( u'Reset' , ), 1610874905, (1610874905, (), [ ], 1 , 1 , 4 , 0 , 220 , (3, 0, None, None) , 0 , )),
	(( u'ShowPopup' , u'x' , u'y' , ), 1610874906, (1610874906, (), [ (12, 17, None, None) ,
			(12, 17, None, None) , ], 1 , 1 , 4 , 2 , 224 , (3, 0, None, None) , 0 , )),
	(( u'Top' , u'pypTop' , ), 1610874907, (1610874907, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 228 , (3, 0, None, None) , 0 , )),
	(( u'Top' , u'pypTop' , ), 1610874907, (1610874907, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 232 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'ptype' , ), 1610874909, (1610874909, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 236 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'pvarfVisible' , ), 1610874910, (1610874910, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 240 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'pvarfVisible' , ), 1610874910, (1610874910, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 244 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'pdx' , ), 1610874912, (1610874912, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 248 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'pdx' , ), 1610874912, (1610874912, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 252 , (3, 0, None, None) , 0 , )),
	(( u'AdaptiveMenu' , u'pvarfAdaptiveMenu' , ), 1610874914, (1610874914, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 256 , (3, 0, None, None) , 0 , )),
	(( u'AdaptiveMenu' , u'pvarfAdaptiveMenu' , ), 1610874914, (1610874914, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 260 , (3, 0, None, None) , 0 , )),
	(( u'Id' , u'pid' , ), 1610874916, (1610874916, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 264 , (3, 0, None, None) , 64 , )),
]

CommandBarControl_vtables_dispatch_ = 1
CommandBarControl_vtables_ = [
	(( u'BeginGroup' , u'pvarfBeginGroup' , ), 1610874880, (1610874880, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'BeginGroup' , u'pvarfBeginGroup' , ), 1610874880, (1610874880, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'BuiltIn' , u'pvarfBuiltIn' , ), 1610874882, (1610874882, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'Caption' , u'pbstrCaption' , ), 1610874883, (1610874883, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'Caption' , u'pbstrCaption' , ), 1610874883, (1610874883, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'Control' , u'ppidisp' , ), 1610874885, (1610874885, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 140 , (3, 0, None, None) , 64 , )),
	(( u'Copy' , u'Bar' , u'Before' , u'ppcbc' , ), 1610874886, (1610874886, (), [
			(12, 17, None, None) , (12, 17, None, None) , (16393, 10, None, "IID('{000C0308-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 2 , 144 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , u'Temporary' , ), 1610874887, (1610874887, (), [ (12, 17, None, None) , ], 1 , 1 , 4 , 1 , 148 , (3, 0, None, None) , 0 , )),
	(( u'DescriptionText' , u'pbstrText' , ), 1610874888, (1610874888, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 152 , (3, 0, None, None) , 0 , )),
	(( u'DescriptionText' , u'pbstrText' , ), 1610874888, (1610874888, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 156 , (3, 0, None, None) , 0 , )),
	(( u'Enabled' , u'pvarfEnabled' , ), 1610874890, (1610874890, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 160 , (3, 0, None, None) , 0 , )),
	(( u'Enabled' , u'pvarfEnabled' , ), 1610874890, (1610874890, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 164 , (3, 0, None, None) , 0 , )),
	(( u'Execute' , ), 1610874892, (1610874892, (), [ ], 1 , 1 , 4 , 0 , 168 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'pdy' , ), 1610874893, (1610874893, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 172 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'pdy' , ), 1610874893, (1610874893, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 176 , (3, 0, None, None) , 0 , )),
	(( u'HelpContextId' , u'pid' , ), 1610874895, (1610874895, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 180 , (3, 0, None, None) , 0 , )),
	(( u'HelpContextId' , u'pid' , ), 1610874895, (1610874895, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 184 , (3, 0, None, None) , 0 , )),
	(( u'HelpFile' , u'pbstrFilename' , ), 1610874897, (1610874897, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 188 , (3, 0, None, None) , 0 , )),
	(( u'HelpFile' , u'pbstrFilename' , ), 1610874897, (1610874897, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 192 , (3, 0, None, None) , 0 , )),
	(( u'Id' , u'pid' , ), 1610874899, (1610874899, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 196 , (3, 0, None, None) , 0 , )),
	(( u'Index' , u'pi' , ), 1610874900, (1610874900, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 200 , (3, 0, None, None) , 0 , )),
	(( u'InstanceId' , u'pid' , ), 1610874901, (1610874901, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 204 , (3, 0, None, None) , 64 , )),
	(( u'Move' , u'Bar' , u'Before' , u'ppcbc' , ), 1610874902, (1610874902, (), [
			(12, 17, None, None) , (12, 17, None, None) , (16393, 10, None, "IID('{000C0308-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 2 , 208 , (3, 0, None, None) , 0 , )),
	(( u'Left' , u'px' , ), 1610874903, (1610874903, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 212 , (3, 0, None, None) , 0 , )),
	(( u'OLEUsage' , u'pcou' , ), 1610874904, (1610874904, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 216 , (3, 0, None, None) , 0 , )),
	(( u'OLEUsage' , u'pcou' , ), 1610874904, (1610874904, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 220 , (3, 0, None, None) , 0 , )),
	(( u'OnAction' , u'pbstrOnAction' , ), 1610874906, (1610874906, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 224 , (3, 0, None, None) , 0 , )),
	(( u'OnAction' , u'pbstrOnAction' , ), 1610874906, (1610874906, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 228 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppcb' , ), 1610874908, (1610874908, (), [ (16393, 10, None, "IID('{000C0304-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 232 , (3, 0, None, None) , 0 , )),
	(( u'Parameter' , u'pbstrParam' , ), 1610874909, (1610874909, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 236 , (3, 0, None, None) , 0 , )),
	(( u'Parameter' , u'pbstrParam' , ), 1610874909, (1610874909, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 240 , (3, 0, None, None) , 0 , )),
	(( u'Priority' , u'pnPri' , ), 1610874911, (1610874911, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 244 , (3, 0, None, None) , 0 , )),
	(( u'Priority' , u'pnPri' , ), 1610874911, (1610874911, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 248 , (3, 0, None, None) , 0 , )),
	(( u'Reset' , ), 1610874913, (1610874913, (), [ ], 1 , 1 , 4 , 0 , 252 , (3, 0, None, None) , 0 , )),
	(( u'SetFocus' , ), 1610874914, (1610874914, (), [ ], 1 , 1 , 4 , 0 , 256 , (3, 0, None, None) , 0 , )),
	(( u'Tag' , u'pbstrTag' , ), 1610874915, (1610874915, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 260 , (3, 0, None, None) , 0 , )),
	(( u'Tag' , u'pbstrTag' , ), 1610874915, (1610874915, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 264 , (3, 0, None, None) , 0 , )),
	(( u'TooltipText' , u'pbstrTooltip' , ), 1610874917, (1610874917, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 268 , (3, 0, None, None) , 0 , )),
	(( u'TooltipText' , u'pbstrTooltip' , ), 1610874917, (1610874917, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 272 , (3, 0, None, None) , 0 , )),
	(( u'Top' , u'py' , ), 1610874919, (1610874919, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 276 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'ptype' , ), 1610874920, (1610874920, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 280 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'pvarfVisible' , ), 1610874921, (1610874921, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 284 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'pvarfVisible' , ), 1610874921, (1610874921, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 288 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'pdx' , ), 1610874923, (1610874923, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 292 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'pdx' , ), 1610874923, (1610874923, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 296 , (3, 0, None, None) , 0 , )),
	(( u'IsPriorityDropped' , u'pvarfDropped' , ), 1610874925, (1610874925, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 300 , (3, 0, None, None) , 0 , )),
	(( u'Reserved1' , ), 1610874926, (1610874926, (), [ ], 1 , 1 , 4 , 0 , 304 , (3, 0, None, None) , 64 , )),
	(( u'Reserved2' , ), 1610874927, (1610874927, (), [ ], 1 , 1 , 4 , 0 , 308 , (3, 0, None, None) , 64 , )),
	(( u'Reserved3' , ), 1610874928, (1610874928, (), [ ], 1 , 1 , 4 , 0 , 312 , (3, 0, None, None) , 64 , )),
	(( u'Reserved4' , ), 1610874929, (1610874929, (), [ ], 1 , 1 , 4 , 0 , 316 , (3, 0, None, None) , 64 , )),
	(( u'Reserved5' , ), 1610874930, (1610874930, (), [ ], 1 , 1 , 4 , 0 , 320 , (3, 0, None, None) , 64 , )),
	(( u'Reserved6' , ), 1610874931, (1610874931, (), [ ], 1 , 1 , 4 , 0 , 324 , (3, 0, None, None) , 64 , )),
	(( u'Reserved7' , ), 1610874932, (1610874932, (), [ ], 1 , 1 , 4 , 0 , 328 , (3, 0, None, None) , 64 , )),
]

CommandBarControls_vtables_dispatch_ = 1
CommandBarControls_vtables_ = [
	(( u'Add' , u'Type' , u'Id' , u'Parameter' , u'Before' ,
			u'Temporary' , u'ppcbc' , ), 1610809344, (1610809344, (), [ (12, 17, None, None) , (12, 17, None, None) ,
			(12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) , (16393, 10, None, "IID('{000C0308-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 5 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pcToolbarControls' , ), 1610809345, (1610809345, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'ppcbc' , ), 0, (0, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C0308-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 1024 , )),
	(( u'Parent' , u'ppcb' , ), 1610809348, (1610809348, (), [ (16393, 10, None, "IID('{000C0304-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

CommandBarPopup_vtables_dispatch_ = 1
CommandBarPopup_vtables_ = [
	(( u'CommandBar' , u'ppcb' , ), 1610940416, (1610940416, (), [ (16393, 10, None, "IID('{000C0304-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 332 , (3, 0, None, None) , 0 , )),
	(( u'Controls' , u'ppcbcs' , ), 1610940417, (1610940417, (), [ (16393, 10, None, "IID('{000C0306-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 336 , (3, 0, None, None) , 0 , )),
	(( u'OLEMenuGroup' , u'pomg' , ), 1610940418, (1610940418, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 340 , (3, 0, None, None) , 0 , )),
	(( u'OLEMenuGroup' , u'pomg' , ), 1610940418, (1610940418, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 344 , (3, 0, None, None) , 0 , )),
]

ConnectorFormat_vtables_dispatch_ = 1
ConnectorFormat_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'BeginConnect' , u'ConnectedShape' , u'ConnectionSite' , ), 10, (10, (), [ (9, 1, None, "IID('{000C031C-0000-0000-C000-000000000046}')") ,
			(3, 1, None, None) , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'BeginDisconnect' , ), 11, (11, (), [ ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'EndConnect' , u'ConnectedShape' , u'ConnectionSite' , ), 12, (12, (), [ (9, 1, None, "IID('{000C031C-0000-0000-C000-000000000046}')") ,
			(3, 1, None, None) , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'EndDisconnect' , ), 13, (13, (), [ ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'BeginConnected' , u'BeginConnected' , ), 100, (100, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'BeginConnectedShape' , u'BeginConnectedShape' , ), 101, (101, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'BeginConnectionSite' , u'BeginConnectionSite' , ), 102, (102, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'EndConnected' , u'EndConnected' , ), 103, (103, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'EndConnectedShape' , u'EndConnectedShape' , ), 104, (104, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'EndConnectionSite' , u'EndConnectionSite' , ), 105, (105, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'Type' , ), 106, (106, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'Type' , ), 106, (106, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
]

DiagramNode_vtables_dispatch_ = 1
DiagramNode_vtables_ = [
	(( u'AddNode' , u'Pos' , u'NodeType' , u'NewNode' , ), 10, (10, (), [
			(3, 49, '2', None) , (3, 49, '1', None) , (16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 11, (11, (), [ ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'MoveNode' , u'TargetNode' , u'Pos' , ), 12, (12, (), [ (9, 1, None, "IID('{000C0370-0000-0000-C000-000000000046}')") ,
			(3, 1, None, None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ReplaceNode' , u'TargetNode' , ), 13, (13, (), [ (9, 1, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'SwapNode' , u'TargetNode' , u'SwapChildren' , ), 14, (14, (), [ (9, 1, None, "IID('{000C0370-0000-0000-C000-000000000046}')") ,
			(11, 49, 'True', None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'CloneNode' , u'CopyChildren' , u'TargetNode' , u'Pos' , u'Node' ,
			), 15, (15, (), [ (11, 1, None, None) , (9, 1, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , (3, 49, '2', None) , (16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'TransferChildren' , u'ReceivingNode' , ), 16, (16, (), [ (9, 1, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'NextNode' , u'NextNode' , ), 17, (17, (), [ (16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'PrevNode' , u'PrevNode' , ), 18, (18, (), [ (16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'Parent' , ), 100, (100, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'Children' , u'Children' , ), 101, (101, (), [ (16393, 10, None, "IID('{000C036F-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'Shape' , u'Shape' , ), 102, (102, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Root' , u'Root' , ), 103, (103, (), [ (16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'Diagram' , u'Diagram' , ), 104, (104, (), [ (16393, 10, None, "IID('{000C036D-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'Layout' , u'Type' , ), 105, (105, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'Layout' , u'Type' , ), 105, (105, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'TextShape' , u'Shape' , ), 106, (106, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
]

DiagramNodeChildren_vtables_dispatch_ = 1
DiagramNodeChildren_vtables_ = [
	(( u'_NewEnum' , u'ppunkEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 1024 , )),
	(( u'Item' , u'Index' , u'Node' , ), 0, (0, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'AddNode' , u'Index' , u'NodeType' , u'NewNode' , ), 10, (10, (), [
			(12, 49, '-1', None) , (3, 49, '1', None) , (16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'SelectAll' , ), 11, (11, (), [ ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'Parent' , ), 100, (100, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'iDiagramNodes' , ), 101, (101, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'FirstChild' , u'First' , ), 103, (103, (), [ (16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'LastChild' , u'Last' , ), 104, (104, (), [ (16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
]

DiagramNodes_vtables_dispatch_ = 1
DiagramNodes_vtables_ = [
	(( u'_NewEnum' , u'ppunkEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 1024 , )),
	(( u'Item' , u'Index' , u'ppdn' , ), 0, (0, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'SelectAll' , ), 10, (10, (), [ ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'Parent' , ), 100, (100, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'iDiagramNodes' , ), 101, (101, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

DocumentLibraryVersion_vtables_dispatch_ = 1
DocumentLibraryVersion_vtables_ = [
	(( u'Modified' , u'pvarDate' , ), 0, (0, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Index' , u'lIndex' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 2, (2, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ModifiedBy' , u'userName' , ), 3, (3, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Comments' , u'Comments' , ), 4, (4, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 5, (5, (), [ ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Open' , u'ppdispOpened' , ), 6, (6, (), [ (16393, 10, None, None) , ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Restore' , u'ppdispOpened' , ), 7, (7, (), [ (16393, 10, None, None) , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
]

DocumentLibraryVersions_vtables_dispatch_ = 1
DocumentLibraryVersions_vtables_ = [
	(( u'Item' , u'lIndex' , u'ppidisp' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C0387-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'lCount' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 2, (2, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'IsVersioningEnabled' , u'pvarfVersioningOn' , ), 3, (3, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 1024 , )),
]

FileDialog_vtables_dispatch_ = 1
FileDialog_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1610809344, (1610809344, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Filters' , u'Filters' , ), 1610809345, (1610809345, (), [ (16393, 10, None, "IID('{000C0365-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'FilterIndex' , u'FilterIndex' , ), 1610809346, (1610809346, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'FilterIndex' , u'FilterIndex' , ), 1610809346, (1610809346, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Title' , u'Title' , ), 1610809348, (1610809348, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Title' , u'Title' , ), 1610809348, (1610809348, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'ButtonName' , u'ButtonName' , ), 1610809350, (1610809350, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'ButtonName' , u'ButtonName' , ), 1610809350, (1610809350, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'AllowMultiSelect' , u'pvarfAllowMultiSelect' , ), 1610809352, (1610809352, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'AllowMultiSelect' , u'pvarfAllowMultiSelect' , ), 1610809352, (1610809352, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'InitialView' , u'pinitialview' , ), 1610809354, (1610809354, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'InitialView' , u'pinitialview' , ), 1610809354, (1610809354, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'InitialFileName' , u'InitialFileName' , ), 1610809356, (1610809356, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'InitialFileName' , u'InitialFileName' , ), 1610809356, (1610809356, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'SelectedItems' , u'Files' , ), 1610809358, (1610809358, (), [ (16393, 10, None, "IID('{000C0363-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'DialogType' , u'pdialogtype' , ), 1610809359, (1610809359, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Name' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'Show' , u'rval' , ), 1610809361, (1610809361, (), [ (16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'Execute' , ), 1610809362, (1610809362, (), [ ], 1 , 1 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
]

FileDialogFilter_vtables_dispatch_ = 1
FileDialogFilter_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1610809344, (1610809344, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Extensions' , u'Extensions' , ), 1610809345, (1610809345, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Description' , u'Description' , ), 1610809346, (1610809346, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
]

FileDialogFilters_vtables_dispatch_ = 1
FileDialogFilters_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1610809344, (1610809344, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 1024 , )),
	(( u'Count' , u'pcFilters' , ), 1610809346, (1610809346, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'Item' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C0364-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , u'filter' , ), 1610809348, (1610809348, (), [ (12, 17, None, None) , ], 1 , 1 , 4 , 1 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Clear' , ), 1610809349, (1610809349, (), [ ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'Description' , u'Extensions' , u'Position' , u'Add' ,
			), 1610809350, (1610809350, (), [ (8, 1, None, None) , (8, 1, None, None) , (12, 17, None, None) , (16393, 10, None, "IID('{000C0364-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 1 , 60 , (3, 0, None, None) , 0 , )),
]

FileDialogSelectedItems_vtables_dispatch_ = 1
FileDialogSelectedItems_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1610809344, (1610809344, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 1024 , )),
	(( u'Count' , u'pcFiles' , ), 1610809346, (1610809346, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'Item' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
]

FileSearch_vtables_dispatch_ = 1
FileSearch_vtables_ = [
	(( u'SearchSubFolders' , u'SearchSubFoldersRetVal' , ), 1, (1, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'SearchSubFolders' , u'SearchSubFoldersRetVal' , ), 1, (1, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'MatchTextExactly' , u'MatchTextRetVal' , ), 2, (2, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'MatchTextExactly' , u'MatchTextRetVal' , ), 2, (2, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'MatchAllWordForms' , u'MatchAllWordFormsRetVal' , ), 3, (3, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'MatchAllWordForms' , u'MatchAllWordFormsRetVal' , ), 3, (3, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'FileName' , u'FileNameRetVal' , ), 4, (4, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'FileName' , u'FileNameRetVal' , ), 4, (4, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'FileType' , u'FileTypeRetVal' , ), 5, (5, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'FileType' , u'FileTypeRetVal' , ), 5, (5, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'LastModified' , u'LastModifiedRetVal' , ), 6, (6, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'LastModified' , u'LastModifiedRetVal' , ), 6, (6, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'TextOrProperty' , u'TextOrProperty' , ), 7, (7, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'TextOrProperty' , u'TextOrProperty' , ), 7, (7, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'LookIn' , u'LookInRetVal' , ), 8, (8, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'LookIn' , u'LookInRetVal' , ), 8, (8, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'Execute' , u'SortBy' , u'SortOrder' , u'AlwaysAccurate' , u'pRet' ,
			), 9, (9, (), [ (3, 49, '1', None) , (3, 49, '1', None) , (11, 49, 'True', None) , (16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'NewSearch' , ), 10, (10, (), [ ], 1 , 1 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'FoundFiles' , u'FoundFilesRet' , ), 11, (11, (), [ (16393, 10, None, "IID('{000C0331-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'PropertyTests' , u'PropTestsRet' , ), 12, (12, (), [ (16393, 10, None, "IID('{000C0334-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'SearchScopes' , u'SearchScopesRet' , ), 13, (13, (), [ (16393, 10, None, "IID('{000C0366-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'SearchFolders' , u'SearchFoldersRet' , ), 14, (14, (), [ (16393, 10, None, "IID('{000C036A-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'FileTypes' , u'FileTypesRet' , ), 16, (16, (), [ (16393, 10, None, "IID('{000C036C-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'RefreshScopes' , ), 17, (17, (), [ ], 1 , 1 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
]

FileTypes_vtables_dispatch_ = 1
FileTypes_vtables_ = [
	(( u'Item' , u'Index' , u'MsoFileTypeRet' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'iCountRetVal' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'FileType' , ), 3, (3, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Remove' , u'Index' , ), 4, (4, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppunkEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 1024 , )),
]

FillFormat_vtables_dispatch_ = 1
FillFormat_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Background' , ), 10, (10, (), [ ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'OneColorGradient' , u'Style' , u'Variant' , u'Degree' , ), 11, (11, (), [
			(3, 1, None, None) , (3, 1, None, None) , (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Patterned' , u'Pattern' , ), 12, (12, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'PresetGradient' , u'Style' , u'Variant' , u'PresetGradientType' , ), 13, (13, (), [
			(3, 1, None, None) , (3, 1, None, None) , (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'PresetTextured' , u'PresetTexture' , ), 14, (14, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Solid' , ), 15, (15, (), [ ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'TwoColorGradient' , u'Style' , u'Variant' , ), 16, (16, (), [ (3, 1, None, None) ,
			(3, 1, None, None) , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'UserPicture' , u'PictureFile' , ), 17, (17, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'UserTextured' , u'TextureFile' , ), 18, (18, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'BackColor' , u'BackColor' , ), 100, (100, (), [ (16393, 10, None, "IID('{000C0312-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'BackColor' , u'BackColor' , ), 100, (100, (), [ (9, 1, None, "IID('{000C0312-0000-0000-C000-000000000046}')") , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'ForeColor' , u'ForeColor' , ), 101, (101, (), [ (16393, 10, None, "IID('{000C0312-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'ForeColor' , u'ForeColor' , ), 101, (101, (), [ (9, 1, None, "IID('{000C0312-0000-0000-C000-000000000046}')") , ], 1 , 4 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'GradientColorType' , u'GradientColorType' , ), 102, (102, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'GradientDegree' , u'GradientDegree' , ), 103, (103, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'GradientStyle' , u'GradientStyle' , ), 104, (104, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'GradientVariant' , u'GradientVariant' , ), 105, (105, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'Pattern' , u'Pattern' , ), 106, (106, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'PresetGradientType' , u'PresetGradientType' , ), 107, (107, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'PresetTexture' , u'PresetTexture' , ), 108, (108, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'TextureName' , u'TextureName' , ), 109, (109, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'TextureType' , u'TextureType' , ), 110, (110, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'Transparency' , u'Transparency' , ), 111, (111, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'Transparency' , u'Transparency' , ), 111, (111, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'Type' , ), 112, (112, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 113, (113, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 113, (113, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 144 , (3, 0, None, None) , 0 , )),
]

FoundFiles_vtables_dispatch_ = 1
FoundFiles_vtables_ = [
	(( u'Item' , u'Index' , u'lcid' , u'pbstrFile' , ), 0, (0, (), [
			(3, 1, None, None) , (3, 5, None, None) , (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pc' , ), 4, (4, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppunkEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 1024 , )),
]

FreeformBuilder_vtables_dispatch_ = 1
FreeformBuilder_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'AddNodes' , u'SegmentType' , u'EditingType' , u'X1' , u'Y1' ,
			u'X2' , u'Y2' , u'X3' , u'Y3' , ), 10, (10, (), [
			(3, 1, None, None) , (3, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (4, 49, '0.0', None) ,
			(4, 49, '0.0', None) , (4, 49, '0.0', None) , (4, 49, '0.0', None) , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'ConvertToShape' , u'Freeform' , ), 11, (11, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
]

GroupShapes_vtables_dispatch_ = 1
GroupShapes_vtables_ = [
	(( u'Parent' , u'ppidisp' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pnShapes' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'Item' , ), 0, (0, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 1024 , )),
	(( u'Range' , u'Index' , u'Range' , ), 10, (10, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C031D-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

HTMLProject_vtables_dispatch_ = 1
HTMLProject_vtables_ = [
	(( u'State' , u'State' , ), 0, (0, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'RefreshProject' , u'Refresh' , ), 1, (1, (), [ (11, 49, 'True', None) , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'RefreshDocument' , u'Refresh' , ), 2, (2, (), [ (11, 49, 'True', None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'HTMLProjectItems' , u'HTMLProjectItems' , ), 3, (3, (), [ (16393, 10, None, "IID('{000C0357-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 4, (4, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Open' , u'OpenKind' , ), 5, (5, (), [ (3, 49, '0', None) , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
]

HTMLProjectItem_vtables_dispatch_ = 1
HTMLProjectItem_vtables_ = [
	(( u'Name' , u'RetValue' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'IsOpen' , u'RetValue' , ), 4, (4, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'LoadFromFile' , u'FileName' , ), 5, (5, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Open' , u'OpenKind' , ), 6, (6, (), [ (3, 49, '0', None) , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'SaveCopyAs' , u'FileName' , ), 7, (7, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'Text' , ), 8, (8, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'Text' , ), 8, (8, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 10, (10, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
]

HTMLProjectItems_vtables_dispatch_ = 1
HTMLProjectItems_vtables_ = [
	(( u'Item' , u'Index' , u'RetValue' , ), 0, (0, (), [ (16396, 1, None, None) ,
			(16393, 10, None, "IID('{000C0358-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'RetValue' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'RetValue' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 1025 , )),
	(( u'Parent' , u'ppidisp' , ), 2, (2, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
]

IAccessible_vtables_dispatch_ = 1
IAccessible_vtables_ = [
	(( u'accParent' , u'ppdispParent' , ), -5000, (-5000, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 1088 , )),
	(( u'accChildCount' , u'pcountChildren' , ), -5001, (-5001, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 1088 , )),
	(( u'accChild' , u'varChild' , u'ppdispChild' , ), -5002, (-5002, (), [ (12, 1, None, None) ,
			(16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 1088 , )),
	(( u'accName' , u'varChild' , u'pszName' , ), -5003, (-5003, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 40 , (3, 0, None, None) , 1088 , )),
	(( u'accName' , u'varChild' , u'pszName' , ), -5003, (-5003, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 40 , (3, 0, None, None) , 1088 , )),
	(( u'accValue' , u'varChild' , u'pszValue' , ), -5004, (-5004, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 44 , (3, 0, None, None) , 1088 , )),
	(( u'accValue' , u'varChild' , u'pszValue' , ), -5004, (-5004, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 44 , (3, 0, None, None) , 1088 , )),
	(( u'accDescription' , u'varChild' , u'pszDescription' , ), -5005, (-5005, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 48 , (3, 0, None, None) , 1088 , )),
	(( u'accDescription' , u'varChild' , u'pszDescription' , ), -5005, (-5005, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 48 , (3, 0, None, None) , 1088 , )),
	(( u'accRole' , u'varChild' , u'pvarRole' , ), -5006, (-5006, (), [ (12, 17, None, None) ,
			(16396, 10, None, None) , ], 1 , 2 , 4 , 1 , 52 , (3, 0, None, None) , 1088 , )),
	(( u'accRole' , u'varChild' , u'pvarRole' , ), -5006, (-5006, (), [ (12, 17, None, None) ,
			(16396, 10, None, None) , ], 1 , 2 , 4 , 1 , 52 , (3, 0, None, None) , 1088 , )),
	(( u'accState' , u'varChild' , u'pvarState' , ), -5007, (-5007, (), [ (12, 17, None, None) ,
			(16396, 10, None, None) , ], 1 , 2 , 4 , 1 , 56 , (3, 0, None, None) , 1088 , )),
	(( u'accState' , u'varChild' , u'pvarState' , ), -5007, (-5007, (), [ (12, 17, None, None) ,
			(16396, 10, None, None) , ], 1 , 2 , 4 , 1 , 56 , (3, 0, None, None) , 1088 , )),
	(( u'accHelp' , u'varChild' , u'pszHelp' , ), -5008, (-5008, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 60 , (3, 0, None, None) , 1088 , )),
	(( u'accHelp' , u'varChild' , u'pszHelp' , ), -5008, (-5008, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 60 , (3, 0, None, None) , 1088 , )),
	(( u'accHelpTopic' , u'pszHelpFile' , u'varChild' , u'pidTopic' , ), -5009, (-5009, (), [
			(16392, 2, None, None) , (12, 17, None, None) , (16387, 10, None, None) , ], 1 , 2 , 4 , 1 , 64 , (3, 0, None, None) , 1088 , )),
	(( u'accHelpTopic' , u'pszHelpFile' , u'varChild' , u'pidTopic' , ), -5009, (-5009, (), [
			(16392, 2, None, None) , (12, 17, None, None) , (16387, 10, None, None) , ], 1 , 2 , 4 , 1 , 64 , (3, 0, None, None) , 1088 , )),
	(( u'accKeyboardShortcut' , u'varChild' , u'pszKeyboardShortcut' , ), -5010, (-5010, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 68 , (3, 0, None, None) , 1088 , )),
	(( u'accKeyboardShortcut' , u'varChild' , u'pszKeyboardShortcut' , ), -5010, (-5010, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 68 , (3, 0, None, None) , 1088 , )),
	(( u'accFocus' , u'pvarChild' , ), -5011, (-5011, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 1088 , )),
	(( u'accSelection' , u'pvarChildren' , ), -5012, (-5012, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 1088 , )),
	(( u'accDefaultAction' , u'varChild' , u'pszDefaultAction' , ), -5013, (-5013, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 80 , (3, 0, None, None) , 1088 , )),
	(( u'accDefaultAction' , u'varChild' , u'pszDefaultAction' , ), -5013, (-5013, (), [ (12, 17, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 1 , 80 , (3, 0, None, None) , 1088 , )),
	(( u'accSelect' , u'flagsSelect' , u'varChild' , ), -5014, (-5014, (), [ (3, 1, None, None) ,
			(12, 17, None, None) , ], 1 , 1 , 4 , 1 , 84 , (3, 0, None, None) , 1088 , )),
	(( u'accLocation' , u'pxLeft' , u'pyTop' , u'pcxWidth' , u'pcyHeight' ,
			u'varChild' , ), -5015, (-5015, (), [ (16387, 2, None, None) , (16387, 2, None, None) , (16387, 2, None, None) ,
			(16387, 2, None, None) , (12, 17, None, None) , ], 1 , 1 , 4 , 1 , 88 , (3, 0, None, None) , 1088 , )),
	(( u'accNavigate' , u'navDir' , u'varStart' , u'pvarEndUpAt' , ), -5016, (-5016, (), [
			(3, 1, None, None) , (12, 17, None, None) , (16396, 10, None, None) , ], 1 , 1 , 4 , 1 , 92 , (3, 0, None, None) , 1088 , )),
	(( u'accHitTest' , u'xLeft' , u'yTop' , u'pvarChild' , ), -5017, (-5017, (), [
			(3, 1, None, None) , (3, 1, None, None) , (16396, 10, None, None) , ], 1 , 1 , 4 , 0 , 96 , (3, 0, None, None) , 1088 , )),
	(( u'accDoDefaultAction' , u'varChild' , ), -5018, (-5018, (), [ (12, 17, None, None) , ], 1 , 1 , 4 , 1 , 100 , (3, 0, None, None) , 1088 , )),
	(( u'accName' , u'varChild' , u'pszName' , ), -5003, (-5003, (), [ (12, 17, None, None) ,
			(8, 1, None, None) , ], 1 , 4 , 4 , 1 , 104 , (3, 0, None, None) , 1088 , )),
	(( u'accName' , u'varChild' , u'pszName' , ), -5003, (-5003, (), [ (12, 17, None, None) ,
			(8, 1, None, None) , ], 1 , 4 , 4 , 1 , 104 , (3, 0, None, None) , 1088 , )),
	(( u'accValue' , u'varChild' , u'pszValue' , ), -5004, (-5004, (), [ (12, 17, None, None) ,
			(8, 1, None, None) , ], 1 , 4 , 4 , 1 , 108 , (3, 0, None, None) , 1088 , )),
	(( u'accValue' , u'varChild' , u'pszValue' , ), -5004, (-5004, (), [ (12, 17, None, None) ,
			(8, 1, None, None) , ], 1 , 4 , 4 , 1 , 108 , (3, 0, None, None) , 1088 , )),
]

ICommandBarButtonEvents_vtables_dispatch_ = 1
ICommandBarButtonEvents_vtables_ = [
	(( u'Click' , u'Ctrl' , u'CancelDefault' , ), 1, (1, (), [ (13, 1, None, "IID('{55F88891-7708-11D1-ACEB-006008961DA5}')") ,
			(16395, 3, None, None) , ], 1 , 1 , 4 , 0 , 28 , (24, 0, None, None) , 0 , )),
]

ICommandBarComboBoxEvents_vtables_dispatch_ = 1
ICommandBarComboBoxEvents_vtables_ = [
	(( u'Change' , u'Ctrl' , ), 1, (1, (), [ (13, 1, None, "IID('{55F88897-7708-11D1-ACEB-006008961DA5}')") , ], 1 , 1 , 4 , 0 , 28 , (24, 0, None, None) , 0 , )),
]

ICommandBarsEvents_vtables_dispatch_ = 1
ICommandBarsEvents_vtables_ = [
	(( u'OnUpdate' , ), 1, (1, (), [ ], 1 , 1 , 4 , 0 , 28 , (24, 0, None, None) , 0 , )),
]

IFind_vtables_dispatch_ = 1
IFind_vtables_ = [
	(( u'SearchPath' , u'pbstr' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'pbstr' , ), 1610743809, (1610743809, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'SubDir' , u'retval' , ), 1610743810, (1610743810, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Title' , u'pbstr' , ), 1610743811, (1610743811, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Author' , u'pbstr' , ), 1610743812, (1610743812, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Keywords' , u'pbstr' , ), 1610743813, (1610743813, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Subject' , u'pbstr' , ), 1610743814, (1610743814, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Options' , u'penmOptions' , ), 1610743815, (1610743815, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'MatchCase' , u'retval' , ), 1610743816, (1610743816, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'pbstr' , ), 1610743817, (1610743817, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'PatternMatch' , u'retval' , ), 1610743818, (1610743818, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'DateSavedFrom' , u'pdatSavedFrom' , ), 1610743819, (1610743819, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'DateSavedTo' , u'pdatSavedTo' , ), 1610743820, (1610743820, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'SavedBy' , u'pbstr' , ), 1610743821, (1610743821, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'DateCreatedFrom' , u'pdatCreatedFrom' , ), 1610743822, (1610743822, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'DateCreatedTo' , u'pdatCreatedTo' , ), 1610743823, (1610743823, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'View' , u'penmView' , ), 1610743824, (1610743824, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'SortBy' , u'penmSortBy' , ), 1610743825, (1610743825, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'ListBy' , u'penmListBy' , ), 1610743826, (1610743826, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'SelectedFile' , u'pintSelectedFile' , ), 1610743827, (1610743827, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'Results' , u'pdisp' , ), 1610743828, (1610743828, (), [ (16393, 10, None, "IID('{000C0338-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'Show' , u'pRows' , ), 1610743829, (1610743829, (), [ (16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'SearchPath' , u'pbstr' , ), 0, (0, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'pbstr' , ), 1610743809, (1610743809, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'SubDir' , u'retval' , ), 1610743810, (1610743810, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'Title' , u'pbstr' , ), 1610743811, (1610743811, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'Author' , u'pbstr' , ), 1610743812, (1610743812, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'Keywords' , u'pbstr' , ), 1610743813, (1610743813, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'Subject' , u'pbstr' , ), 1610743814, (1610743814, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
	(( u'Options' , u'penmOptions' , ), 1610743815, (1610743815, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 144 , (3, 0, None, None) , 0 , )),
	(( u'MatchCase' , u'retval' , ), 1610743816, (1610743816, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 148 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'pbstr' , ), 1610743817, (1610743817, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 152 , (3, 0, None, None) , 0 , )),
	(( u'PatternMatch' , u'retval' , ), 1610743818, (1610743818, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 156 , (3, 0, None, None) , 0 , )),
	(( u'DateSavedFrom' , u'pdatSavedFrom' , ), 1610743819, (1610743819, (), [ (12, 1, None, None) , ], 1 , 4 , 4 , 0 , 160 , (3, 0, None, None) , 0 , )),
	(( u'DateSavedTo' , u'pdatSavedTo' , ), 1610743820, (1610743820, (), [ (12, 1, None, None) , ], 1 , 4 , 4 , 0 , 164 , (3, 0, None, None) , 0 , )),
	(( u'SavedBy' , u'pbstr' , ), 1610743821, (1610743821, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 168 , (3, 0, None, None) , 0 , )),
	(( u'DateCreatedFrom' , u'pdatCreatedFrom' , ), 1610743822, (1610743822, (), [ (12, 1, None, None) , ], 1 , 4 , 4 , 0 , 172 , (3, 0, None, None) , 0 , )),
	(( u'DateCreatedTo' , u'pdatCreatedTo' , ), 1610743823, (1610743823, (), [ (12, 1, None, None) , ], 1 , 4 , 4 , 0 , 176 , (3, 0, None, None) , 0 , )),
	(( u'View' , u'penmView' , ), 1610743824, (1610743824, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 180 , (3, 0, None, None) , 0 , )),
	(( u'SortBy' , u'penmSortBy' , ), 1610743825, (1610743825, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 184 , (3, 0, None, None) , 0 , )),
	(( u'ListBy' , u'penmListBy' , ), 1610743826, (1610743826, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 188 , (3, 0, None, None) , 0 , )),
	(( u'SelectedFile' , u'pintSelectedFile' , ), 1610743827, (1610743827, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 192 , (3, 0, None, None) , 0 , )),
	(( u'Execute' , ), 1610743850, (1610743850, (), [ ], 1 , 1 , 4 , 0 , 196 , (3, 0, None, None) , 0 , )),
	(( u'Load' , u'bstrQueryName' , ), 1610743851, (1610743851, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 200 , (3, 0, None, None) , 0 , )),
	(( u'Save' , u'bstrQueryName' , ), 1610743852, (1610743852, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 204 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , u'bstrQueryName' , ), 1610743853, (1610743853, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 208 , (3, 0, None, None) , 0 , )),
	(( u'FileType' , u'plFileType' , ), 1610743854, (1610743854, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 212 , (3, 0, None, None) , 0 , )),
	(( u'FileType' , u'plFileType' , ), 1610743854, (1610743854, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 216 , (3, 0, None, None) , 0 , )),
]

IFoundFiles_vtables_dispatch_ = 1
IFoundFiles_vtables_ = [
	(( u'Item' , u'Index' , u'pbstr' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pCount' , ), 1610743809, (1610743809, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppunkEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 1025 , )),
]

ILicAgent_vtables_dispatch_ = 1
ILicAgent_vtables_ = [
	(( u'Initialize' , u'dwBPC' , u'dwMode' , u'bstrLicSource' , u'pdwRetCode' ,
			), 1, (1, (), [ (19, 1, None, None) , (19, 1, None, None) , (8, 1, None, None) , (16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'GetFirstName' , u'pbstrVal' , ), 3, (3, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'SetFirstName' , u'bstrNewVal' , ), 4, (4, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'GetLastName' , u'pbstrVal' , ), 5, (5, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'SetLastName' , u'bstrNewVal' , ), 6, (6, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'GetOrgName' , u'pbstrVal' , ), 7, (7, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'SetOrgName' , u'bstrNewVal' , ), 8, (8, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'GetEmail' , u'pbstrVal' , ), 9, (9, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'SetEmail' , u'bstrNewVal' , ), 10, (10, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'GetPhone' , u'pbstrVal' , ), 11, (11, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'SetPhone' , u'bstrNewVal' , ), 12, (12, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'GetAddress1' , u'pbstrVal' , ), 13, (13, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'SetAddress1' , u'bstrNewVal' , ), 14, (14, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'GetCity' , u'pbstrVal' , ), 15, (15, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'SetCity' , u'bstrNewVal' , ), 16, (16, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'GetState' , u'pbstrVal' , ), 17, (17, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'SetState' , u'bstrNewVal' , ), 18, (18, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'GetCountryCode' , u'pbstrVal' , ), 19, (19, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'SetCountryCode' , u'bstrNewVal' , ), 20, (20, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'GetCountryDesc' , u'pbstrVal' , ), 21, (21, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'SetCountryDesc' , u'bstrNewVal' , ), 22, (22, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'GetZip' , u'pbstrVal' , ), 23, (23, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'SetZip' , u'bstrNewVal' , ), 24, (24, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'GetIsoLanguage' , u'pdwVal' , ), 25, (25, (), [ (16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'SetIsoLanguage' , u'dwNewVal' , ), 26, (26, (), [ (19, 1, None, None) , ], 1 , 1 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'GetMSUpdate' , u'pbstrVal' , ), 32, (32, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'SetMSUpdate' , u'bstrNewVal' , ), 33, (33, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'GetMSOffer' , u'pbstrVal' , ), 34, (34, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'SetMSOffer' , u'bstrNewVal' , ), 35, (35, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
	(( u'GetOtherOffer' , u'pbstrVal' , ), 36, (36, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 144 , (3, 0, None, None) , 0 , )),
	(( u'SetOtherOffer' , u'bstrNewVal' , ), 37, (37, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 148 , (3, 0, None, None) , 0 , )),
	(( u'GetAddress2' , u'pbstrVal' , ), 38, (38, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 152 , (3, 0, None, None) , 0 , )),
	(( u'SetAddress2' , u'bstrNewVal' , ), 39, (39, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 156 , (3, 0, None, None) , 0 , )),
	(( u'CheckSystemClock' , u'pdwRetCode' , ), 40, (40, (), [ (16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 160 , (3, 0, None, None) , 0 , )),
	(( u'GetExistingExpiryDate' , u'pDateVal' , ), 41, (41, (), [ (16391, 10, None, None) , ], 1 , 1 , 4 , 0 , 164 , (3, 0, None, None) , 0 , )),
	(( u'GetNewExpiryDate' , u'pDateVal' , ), 42, (42, (), [ (16391, 10, None, None) , ], 1 , 1 , 4 , 0 , 168 , (3, 0, None, None) , 0 , )),
	(( u'GetBillingFirstName' , u'pbstrVal' , ), 43, (43, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 172 , (3, 0, None, None) , 0 , )),
	(( u'SetBillingFirstName' , u'bstrNewVal' , ), 44, (44, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 176 , (3, 0, None, None) , 0 , )),
	(( u'GetBillingLastName' , u'pbstrVal' , ), 45, (45, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 180 , (3, 0, None, None) , 0 , )),
	(( u'SetBillingLastName' , u'bstrNewVal' , ), 46, (46, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 184 , (3, 0, None, None) , 0 , )),
	(( u'GetBillingPhone' , u'pbstrVal' , ), 47, (47, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 188 , (3, 0, None, None) , 0 , )),
	(( u'SetBillingPhone' , u'bstrNewVal' , ), 48, (48, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 192 , (3, 0, None, None) , 0 , )),
	(( u'GetBillingAddress1' , u'pbstrVal' , ), 49, (49, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 196 , (3, 0, None, None) , 0 , )),
	(( u'SetBillingAddress1' , u'bstrNewVal' , ), 50, (50, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 200 , (3, 0, None, None) , 0 , )),
	(( u'GetBillingAddress2' , u'pbstrVal' , ), 51, (51, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 204 , (3, 0, None, None) , 0 , )),
	(( u'SetBillingAddress2' , u'bstrNewVal' , ), 52, (52, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 208 , (3, 0, None, None) , 0 , )),
	(( u'GetBillingCity' , u'pbstrVal' , ), 53, (53, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 212 , (3, 0, None, None) , 0 , )),
	(( u'SetBillingCity' , u'bstrNewVal' , ), 54, (54, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 216 , (3, 0, None, None) , 0 , )),
	(( u'GetBillingState' , u'pbstrVal' , ), 55, (55, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 220 , (3, 0, None, None) , 0 , )),
	(( u'SetBillingState' , u'bstrNewVal' , ), 56, (56, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 224 , (3, 0, None, None) , 0 , )),
	(( u'GetBillingCountryCode' , u'pbstrVal' , ), 57, (57, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 228 , (3, 0, None, None) , 0 , )),
	(( u'SetBillingCountryCode' , u'bstrNewVal' , ), 58, (58, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 232 , (3, 0, None, None) , 0 , )),
	(( u'GetBillingZip' , u'pbstrVal' , ), 59, (59, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 236 , (3, 0, None, None) , 0 , )),
	(( u'SetBillingZip' , u'bstrNewVal' , ), 60, (60, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 240 , (3, 0, None, None) , 0 , )),
	(( u'SaveBillingInfo' , u'bSave' , u'pdwRetVal' , ), 61, (61, (), [ (3, 1, None, None) ,
			(16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 244 , (3, 0, None, None) , 0 , )),
	(( u'IsCCRenewalCountry' , u'bstrCountryCode' , u'pbRetVal' , ), 64, (64, (), [ (8, 1, None, None) ,
			(16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 248 , (3, 0, None, None) , 0 , )),
	(( u'GetVATLabel' , u'bstrCountryCode' , u'pbstrVATLabel' , ), 65, (65, (), [ (8, 1, None, None) ,
			(16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 252 , (3, 0, None, None) , 0 , )),
	(( u'GetCCRenewalExpiryDate' , u'pDateVal' , ), 66, (66, (), [ (16391, 10, None, None) , ], 1 , 1 , 4 , 0 , 256 , (3, 0, None, None) , 0 , )),
	(( u'SetVATNumber' , u'bstrVATNumber' , ), 67, (67, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 260 , (3, 0, None, None) , 0 , )),
	(( u'SetCreditCardType' , u'bstrCCCode' , ), 68, (68, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 264 , (3, 0, None, None) , 0 , )),
	(( u'SetCreditCardNumber' , u'bstrCCNumber' , ), 69, (69, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 268 , (3, 0, None, None) , 0 , )),
	(( u'SetCreditCardExpiryYear' , u'dwCCYear' , ), 70, (70, (), [ (19, 1, None, None) , ], 1 , 1 , 4 , 0 , 272 , (3, 0, None, None) , 0 , )),
	(( u'SetCreditCardExpiryMonth' , u'dwCCMonth' , ), 71, (71, (), [ (19, 1, None, None) , ], 1 , 1 , 4 , 0 , 276 , (3, 0, None, None) , 0 , )),
	(( u'GetCreditCardCount' , u'pdwCount' , ), 72, (72, (), [ (16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 280 , (3, 0, None, None) , 0 , )),
	(( u'GetCreditCardCode' , u'dwIndex' , u'pbstrCode' , ), 73, (73, (), [ (19, 1, None, None) ,
			(16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 284 , (3, 0, None, None) , 0 , )),
	(( u'GetCreditCardName' , u'dwIndex' , u'pbstrName' , ), 74, (74, (), [ (19, 1, None, None) ,
			(16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 288 , (3, 0, None, None) , 0 , )),
	(( u'GetVATNumber' , u'pbstrVATNumber' , ), 75, (75, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 292 , (3, 0, None, None) , 0 , )),
	(( u'GetCreditCardType' , u'pbstrCCCode' , ), 76, (76, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 296 , (3, 0, None, None) , 0 , )),
	(( u'GetCreditCardNumber' , u'pbstrCCNumber' , ), 77, (77, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 300 , (3, 0, None, None) , 0 , )),
	(( u'GetCreditCardExpiryYear' , u'pdwCCYear' , ), 78, (78, (), [ (16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 304 , (3, 0, None, None) , 0 , )),
	(( u'GetCreditCardExpiryMonth' , u'pdwCCMonth' , ), 79, (79, (), [ (16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 308 , (3, 0, None, None) , 0 , )),
	(( u'GetDisconnectOption' , u'pbRetVal' , ), 80, (80, (), [ (16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 312 , (3, 0, None, None) , 0 , )),
	(( u'SetDisconnectOption' , u'bNewVal' , ), 81, (81, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 316 , (3, 0, None, None) , 0 , )),
	(( u'AsyncProcessHandshakeRequest' , u'bReviseCustInfo' , ), 82, (82, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 320 , (3, 0, None, None) , 0 , )),
	(( u'AsyncProcessNewLicenseRequest' , ), 83, (83, (), [ ], 1 , 1 , 4 , 0 , 324 , (3, 0, None, None) , 0 , )),
	(( u'AsyncProcessReissueLicenseRequest' , ), 84, (84, (), [ ], 1 , 1 , 4 , 0 , 328 , (3, 0, None, None) , 0 , )),
	(( u'AsyncProcessRetailRenewalLicenseRequest' , ), 85, (85, (), [ ], 1 , 1 , 4 , 0 , 332 , (3, 0, None, None) , 0 , )),
	(( u'AsyncProcessReviseCustInfoRequest' , ), 86, (86, (), [ ], 1 , 1 , 4 , 0 , 336 , (3, 0, None, None) , 0 , )),
	(( u'AsyncProcessCCRenewalPriceRequest' , ), 87, (87, (), [ ], 1 , 1 , 4 , 0 , 340 , (3, 0, None, None) , 0 , )),
	(( u'AsyncProcessCCRenewalLicenseRequest' , ), 88, (88, (), [ ], 1 , 1 , 4 , 0 , 344 , (3, 0, None, None) , 0 , )),
	(( u'GetAsyncProcessReturnCode' , u'pdwRetCode' , ), 90, (90, (), [ (16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 348 , (3, 0, None, None) , 0 , )),
	(( u'IsUpgradeAvailable' , u'pbUpgradeAvailable' , ), 91, (91, (), [ (16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 352 , (3, 0, None, None) , 0 , )),
	(( u'WantUpgrade' , u'bWantUpgrade' , ), 92, (92, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 356 , (3, 0, None, None) , 0 , )),
	(( u'AsyncProcessDroppedLicenseRequest' , ), 93, (93, (), [ ], 1 , 1 , 4 , 0 , 360 , (3, 0, None, None) , 0 , )),
	(( u'GenerateInstallationId' , u'pbstrVal' , ), 94, (94, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 364 , (3, 0, None, None) , 0 , )),
	(( u'DepositConfirmationId' , u'bstrVal' , u'pdwRetCode' , ), 95, (95, (), [ (8, 1, None, None) ,
			(16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 368 , (3, 0, None, None) , 0 , )),
	(( u'VerifyCheckDigits' , u'bstrCIDIID' , u'pbValue' , ), 96, (96, (), [ (8, 1, None, None) ,
			(16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 372 , (3, 0, None, None) , 0 , )),
	(( u'GetCurrentExpiryDate' , u'pDateVal' , ), 97, (97, (), [ (16391, 10, None, None) , ], 1 , 1 , 4 , 0 , 376 , (3, 0, None, None) , 0 , )),
	(( u'CancelAsyncProcessRequest' , u'bIsLicenseRequest' , ), 98, (98, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 380 , (3, 0, None, None) , 0 , )),
	(( u'GetCurrencyDescription' , u'dwCurrencyIndex' , u'pbstrVal' , ), 100, (100, (), [ (19, 1, None, None) ,
			(16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 384 , (3, 0, None, None) , 0 , )),
	(( u'GetPriceItemCount' , u'pdwCount' , ), 101, (101, (), [ (16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 388 , (3, 0, None, None) , 0 , )),
	(( u'GetPriceItemLabel' , u'dwIndex' , u'pbstrVal' , ), 102, (102, (), [ (19, 1, None, None) ,
			(16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 392 , (3, 0, None, None) , 0 , )),
	(( u'GetPriceItemValue' , u'dwCurrencyIndex' , u'dwIndex' , u'pbstrVal' , ), 103, (103, (), [
			(19, 1, None, None) , (19, 1, None, None) , (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 396 , (3, 0, None, None) , 0 , )),
	(( u'GetInvoiceText' , u'pNewVal' , ), 104, (104, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 400 , (3, 0, None, None) , 0 , )),
	(( u'GetBackendErrorMsg' , u'pbstrErrMsg' , ), 105, (105, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 404 , (3, 0, None, None) , 0 , )),
	(( u'GetCurrencyOption' , u'dwCurrencyOption' , ), 106, (106, (), [ (16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 408 , (3, 0, None, None) , 0 , )),
	(( u'SetCurrencyOption' , u'dwCurrencyOption' , ), 107, (107, (), [ (19, 1, None, None) , ], 1 , 1 , 4 , 0 , 412 , (3, 0, None, None) , 0 , )),
	(( u'GetEndOfLifeHtmlText' , u'pbstrHtmlText' , ), 108, (108, (), [ (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 416 , (3, 0, None, None) , 0 , )),
	(( u'DisplaySSLCert' , u'dwRetCode' , ), 109, (109, (), [ (16403, 10, None, None) , ], 1 , 1 , 4 , 0 , 420 , (3, 0, None, None) , 0 , )),
]

ILicValidator_vtables_dispatch_ = 1
ILicValidator_vtables_ = [
	(( u'Products' , u'pVariant' , ), 1, (1, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Selection' , u'piSel' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Selection' , u'piSel' , ), 2, (2, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
]

ILicWizExternal_vtables_dispatch_ = 1
ILicWizExternal_vtables_ = [
	(( u'PrintHtmlDocument' , u'punkHtmlDoc' , ), 1, (1, (), [ (13, 1, None, None) , ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'InvokeDateTimeApplet' , ), 2, (2, (), [ ], 1 , 1 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'FormatDate' , u'date' , u'pFormat' , u'pDateString' , ), 3, (3, (), [
			(7, 1, None, None) , (8, 49, "u''", None) , (16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 36 , (3, 32, None, None) , 0 , )),
	(( u'ShowHelp' , u'pvarId' , ), 4, (4, (), [ (16396, 17, None, None) , ], 1 , 1 , 4 , 1 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Terminate' , ), 5, (5, (), [ ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'DisableVORWReminder' , u'BPC' , ), 6, (6, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'SaveReceipt' , u'bstrReceipt' , u'pbstrPath' , ), 7, (7, (), [ (8, 1, None, None) ,
			(16392, 10, None, None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'OpenInDefaultBrowser' , u'bstrUrl' , ), 8, (8, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'MsoAlert' , u'bstrText' , u'bstrButtons' , u'bstrIcon' , u'plRet' ,
			), 9, (9, (), [ (8, 1, None, None) , (8, 1, None, None) , (8, 1, None, None) , (16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'DepositPidKey' , u'bstrKey' , u'fMORW' , u'plRet' , ), 10, (10, (), [
			(8, 1, None, None) , (3, 1, None, None) , (16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'WriteLog' , u'bstrMessage' , ), 11, (11, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'ResignDpc' , u'bstrProductCode' , ), 12, (12, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'ResetPID' , ), 13, (13, (), [ ], 1 , 1 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'SetDialogSize' , u'dx' , u'dy' , ), 14, (14, (), [ (3, 1, None, None) ,
			(3, 1, None, None) , ], 1 , 1 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'VerifyClock' , u'lMode' , u'plRet' , ), 15, (15, (), [ (3, 1, None, None) ,
			(16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'SortSelectOptions' , u'pdispSelect' , ), 16, (16, (), [ (9, 1, None, None) , ], 1 , 1 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'InternetDisconnect' , ), 17, (17, (), [ ], 1 , 1 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'GetConnectedState' , u'pfConnected' , ), 18, (18, (), [ (16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'Context' , u'plwctx' , ), 20, (20, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'Validator' , u'ppdispValidator' , ), 21, (21, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'LicAgent' , u'ppdispLicAgent' , ), 22, (22, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'CountryInfo' , u'pbstrUrl' , ), 23, (23, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'WizardVisible' , ), 24, (24, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'WizardTitle' , ), 25, (25, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'AnimationEnabled' , u'fEnabled' , ), 26, (26, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'CurrentHelpId' , ), 27, (27, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'OfficeOnTheWebUrl' , u'bstrUrl' , ), 28, (28, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
]

IMsoDiagram_vtables_dispatch_ = 1
IMsoDiagram_vtables_ = [
	(( u'Parent' , u'Parent' , ), 100, (100, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Nodes' , u'Nodes' , ), 101, (101, (), [ (16393, 10, None, "IID('{000C036E-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'Type' , ), 102, (102, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'AutoLayout' , u'AutoLayout' , ), 103, (103, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'AutoLayout' , u'AutoLayout' , ), 103, (103, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Reverse' , u'Reverse' , ), 104, (104, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Reverse' , u'Reverse' , ), 104, (104, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'AutoFormat' , u'AutoFormat' , ), 105, (105, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'AutoFormat' , u'AutoFormat' , ), 105, (105, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Convert' , u'Type' , ), 10, (10, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'FitText' , ), 11, (11, (), [ ], 1 , 1 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
]

IMsoDispCagNotifySink_vtables_dispatch_ = 1
IMsoDispCagNotifySink_vtables_ = [
	(( u'InsertClip' , u'pClipMoniker' , u'pItemMoniker' , ), 1, (1, (), [ (13, 1, None, None) ,
			(13, 1, None, None) , ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'WindowIsClosing' , ), 2, (2, (), [ ], 1 , 1 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
]

IMsoEServicesDialog_vtables_dispatch_ = 1
IMsoEServicesDialog_vtables_ = [
	(( u'Close' , u'ApplyWebComponentChanges' , ), 1610743808, (1610743808, (), [ (11, 49, 'False', None) , ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'AddTrustedDomain' , u'Domain' , ), 1610743809, (1610743809, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'ApplicationName' , u'retval' , ), 1610743810, (1610743810, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Application' , u'ppdisp' , ), 1610743811, (1610743811, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'WebComponent' , u'ppdisp' , ), 1610743812, (1610743812, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ClipArt' , u'ppdisp' , ), 1610743813, (1610743813, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
]

IMsoEnvelopeVB_vtables_dispatch_ = 1
IMsoEnvelopeVB_vtables_ = [
	(( u'Introduction' , u'pbstrIntro' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Introduction' , u'pbstrIntro' , ), 1, (1, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'ppdisp' , ), 2, (2, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppdisp' , ), 3, (3, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'CommandBars' , u'ppdisp' , ), 4, (4, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
]

LanguageSettings_vtables_dispatch_ = 1
LanguageSettings_vtables_ = [
	(( u'LanguageID' , u'Id' , u'plid' , ), 1, (1, (), [ (3, 1, None, None) ,
			(16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'LanguagePreferredForEditing' , u'lid' , u'pf' , ), 2, (2, (), [ (3, 1, None, None) ,
			(16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 3, (3, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
]

LineFormat_vtables_dispatch_ = 1
LineFormat_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'BackColor' , u'BackColor' , ), 100, (100, (), [ (16393, 10, None, "IID('{000C0312-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'BackColor' , u'BackColor' , ), 100, (100, (), [ (9, 1, None, "IID('{000C0312-0000-0000-C000-000000000046}')") , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'BeginArrowheadLength' , u'BeginArrowheadLength' , ), 101, (101, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'BeginArrowheadLength' , u'BeginArrowheadLength' , ), 101, (101, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'BeginArrowheadStyle' , u'BeginArrowheadStyle' , ), 102, (102, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'BeginArrowheadStyle' , u'BeginArrowheadStyle' , ), 102, (102, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'BeginArrowheadWidth' , u'BeginArrowheadWidth' , ), 103, (103, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'BeginArrowheadWidth' , u'BeginArrowheadWidth' , ), 103, (103, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'DashStyle' , u'DashStyle' , ), 104, (104, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'DashStyle' , u'DashStyle' , ), 104, (104, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'EndArrowheadLength' , u'EndArrowheadLength' , ), 105, (105, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'EndArrowheadLength' , u'EndArrowheadLength' , ), 105, (105, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'EndArrowheadStyle' , u'EndArrowheadStyle' , ), 106, (106, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'EndArrowheadStyle' , u'EndArrowheadStyle' , ), 106, (106, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'EndArrowheadWidth' , u'EndArrowheadWidth' , ), 107, (107, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'EndArrowheadWidth' , u'EndArrowheadWidth' , ), 107, (107, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'ForeColor' , u'ForeColor' , ), 108, (108, (), [ (16393, 10, None, "IID('{000C0312-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'ForeColor' , u'ForeColor' , ), 108, (108, (), [ (9, 1, None, "IID('{000C0312-0000-0000-C000-000000000046}')") , ], 1 , 4 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'Pattern' , u'Pattern' , ), 109, (109, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'Pattern' , u'Pattern' , ), 109, (109, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'Style' , u'Style' , ), 110, (110, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'Style' , u'Style' , ), 110, (110, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'Transparency' , u'Transparency' , ), 111, (111, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'Transparency' , u'Transparency' , ), 111, (111, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 112, (112, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 112, (112, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
	(( u'Weight' , u'Weight' , ), 113, (113, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 144 , (3, 0, None, None) , 0 , )),
	(( u'Weight' , u'Weight' , ), 113, (113, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 148 , (3, 0, None, None) , 0 , )),
	(( u'InsetPen' , u'InsetPen' , ), 114, (114, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 152 , (3, 0, None, None) , 0 , )),
	(( u'InsetPen' , u'InsetPen' , ), 114, (114, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 156 , (3, 0, None, None) , 0 , )),
]

MsoDebugOptions_vtables_dispatch_ = 1
MsoDebugOptions_vtables_ = [
	(( u'FeatureReports' , u'puintFeatureReports' , ), 4, (4, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 64 , )),
	(( u'FeatureReports' , u'puintFeatureReports' , ), 4, (4, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 64 , )),
	(( u'OutputToDebugger' , u'pvarfOutputToDebugger' , ), 5, (5, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'OutputToDebugger' , u'pvarfOutputToDebugger' , ), 5, (5, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'OutputToFile' , u'pvarfOutputToFile' , ), 6, (6, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'OutputToFile' , u'pvarfOutputToFile' , ), 6, (6, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'OutputToMessageBox' , u'pvarfOutputToMessageBox' , ), 7, (7, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'OutputToMessageBox' , u'pvarfOutputToMessageBox' , ), 7, (7, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
]

NewFile_vtables_dispatch_ = 1
NewFile_vtables_ = [
	(( u'Add' , u'FileName' , u'Section' , u'DisplayName' , u'Action' ,
			u'pvarf' , ), 1, (1, (), [ (8, 1, None, None) , (12, 17, None, None) , (12, 17, None, None) ,
			(12, 17, None, None) , (16395, 10, None, None) , ], 1 , 1 , 4 , 3 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Remove' , u'FileName' , u'Section' , u'DisplayName' , u'Action' ,
			u'pvarf' , ), 2, (2, (), [ (8, 1, None, None) , (12, 17, None, None) , (12, 17, None, None) ,
			(12, 17, None, None) , (16395, 10, None, None) , ], 1 , 1 , 4 , 3 , 40 , (3, 0, None, None) , 0 , )),
]

ODSOColumn_vtables_dispatch_ = 1
ODSOColumn_vtables_ = [
	(( u'Index' , u'plIndex' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'pbstrName' , ), 2, (2, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppParent' , ), 3, (3, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Value' , u'pbstrValue' , ), 4, (4, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
]

ODSOColumns_vtables_dispatch_ = 1
ODSOColumns_vtables_ = [
	(( u'Count' , u'plCount' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppParentOdso' , ), 2, (2, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'varIndex' , u'ppColumn' , ), 1610809346, (1610809346, (), [ (12, 1, None, None) ,
			(16393, 10, None, None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
]

ODSOFilter_vtables_dispatch_ = 1
ODSOFilter_vtables_ = [
	(( u'Index' , u'plIndex' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppParent' , ), 2, (2, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Column' , u'pbstrCol' , ), 3, (3, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Column' , u'pbstrCol' , ), 3, (3, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Comparison' , u'pComparison' , ), 4, (4, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Comparison' , u'pComparison' , ), 4, (4, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'CompareTo' , u'pbstrCompareTo' , ), 5, (5, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'CompareTo' , u'pbstrCompareTo' , ), 5, (5, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Conjunction' , u'pConjunction' , ), 6, (6, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Conjunction' , u'pConjunction' , ), 6, (6, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
]

ODSOFilters_vtables_dispatch_ = 1
ODSOFilters_vtables_ = [
	(( u'Count' , u'plCount' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppParentOdso' , ), 2, (2, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'ppColumn' , ), 1610809346, (1610809346, (), [ (3, 1, None, None) ,
			(16393, 10, None, None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'Column' , u'Comparison' , u'Conjunction' , u'bstrCompareTo' ,
			u'DeferUpdate' , ), 1610809347, (1610809347, (), [ (8, 1, None, None) , (3, 1, None, None) , (3, 1, None, None) ,
			(8, 49, "u''", None) , (11, 49, 'False', None) , ], 1 , 1 , 4 , 0 , 48 , (3, 32, None, None) , 0 , )),
	(( u'Delete' , u'Index' , u'DeferUpdate' , ), 1610809348, (1610809348, (), [ (3, 1, None, None) ,
			(11, 49, 'False', None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

OfficeDataSourceObject_vtables_dispatch_ = 1
OfficeDataSourceObject_vtables_ = [
	(( u'ConnectString' , u'pbstrConnect' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'ConnectString' , u'pbstrConnect' , ), 1, (1, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Table' , u'pbstrTable' , ), 2, (2, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Table' , u'pbstrTable' , ), 2, (2, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'DataSource' , u'pbstrSrc' , ), 3, (3, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'DataSource' , u'pbstrSrc' , ), 3, (3, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Columns' , u'ppColumns' , ), 4, (4, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'RowCount' , u'pcRows' , ), 5, (5, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Filters' , u'ppFilters' , ), 6, (6, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Move' , u'MsoMoveRow' , u'RowNbr' , u'rval' , ), 1610743817, (1610743817, (), [
			(3, 1, None, None) , (3, 49, '1', None) , (16387, 10, None, None) , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Open' , u'bstrSrc' , u'bstrConnect' , u'bstrTable' , u'fOpenExclusive' ,
			u'fNeverPrompt' , ), 1610743818, (1610743818, (), [ (8, 49, "u''", None) , (8, 49, "u''", None) , (8, 49, "u''", None) ,
			(3, 49, '0', None) , (3, 49, '1', None) , ], 1 , 1 , 4 , 0 , 68 , (3, 32, None, None) , 0 , )),
	(( u'SetSortOrder' , u'SortField1' , u'SortAscending1' , u'SortField2' , u'SortAscending2' ,
			u'SortField3' , u'SortAscending3' , ), 1610743819, (1610743819, (), [ (8, 1, None, None) , (11, 49, 'True', None) ,
			(8, 49, "u''", None) , (11, 49, 'True', None) , (8, 49, "u''", None) , (11, 49, 'True', None) , ], 1 , 1 , 4 , 0 , 72 , (3, 32, None, None) , 0 , )),
	(( u'ApplyFilter' , ), 1610743820, (1610743820, (), [ ], 1 , 1 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
]

Permission_vtables_dispatch_ = 1
Permission_vtables_ = [
	(( u'Item' , u'Index' , u'UserPerm' , ), 0, (0, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C0375-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'Count' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'EnableTrustedBrowser' , u'Enable' , ), 2, (2, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'EnableTrustedBrowser' , u'Enable' , ), 2, (2, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'UserId' , u'Permission' , u'ExpirationDate' , u'UserPerm' ,
			), 3, (3, (), [ (8, 1, None, None) , (12, 17, None, None) , (12, 17, None, None) , (16393, 10, None, "IID('{000C0375-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 2 , 52 , (3, 0, None, None) , 0 , )),
	(( u'ApplyPolicy' , u'FileName' , ), 4, (4, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 5, (5, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'RemoveAll' , ), 6, (6, (), [ ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Enabled' , u'Enabled' , ), 7, (7, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Enabled' , u'Enabled' , ), 7, (7, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'RequestPermissionURL' , u'Contact' , ), 8, (8, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'RequestPermissionURL' , u'Contact' , ), 8, (8, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'PolicyName' , u'PolicyName' , ), 9, (9, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'PolicyDescription' , u'PolicyDescription' , ), 10, (10, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'StoreLicenses' , u'Enabled' , ), 11, (11, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'StoreLicenses' , u'Enabled' , ), 11, (11, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'DocumentAuthor' , u'Author' , ), 12, (12, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'DocumentAuthor' , u'Author' , ), 12, (12, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'PermissionFromPolicy' , u'FromPolicy' , ), 13, (13, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppunkEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 1024 , )),
]

PictureFormat_vtables_dispatch_ = 1
PictureFormat_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'IncrementBrightness' , u'Increment' , ), 10, (10, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'IncrementContrast' , u'Increment' , ), 11, (11, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Brightness' , u'Brightness' , ), 100, (100, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Brightness' , u'Brightness' , ), 100, (100, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'ColorType' , u'ColorType' , ), 101, (101, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'ColorType' , u'ColorType' , ), 101, (101, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Contrast' , u'Contrast' , ), 102, (102, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Contrast' , u'Contrast' , ), 102, (102, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'CropBottom' , u'CropBottom' , ), 103, (103, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'CropBottom' , u'CropBottom' , ), 103, (103, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'CropLeft' , u'CropLeft' , ), 104, (104, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'CropLeft' , u'CropLeft' , ), 104, (104, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'CropRight' , u'CropRight' , ), 105, (105, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'CropRight' , u'CropRight' , ), 105, (105, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'CropTop' , u'CropTop' , ), 106, (106, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'CropTop' , u'CropTop' , ), 106, (106, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'TransparencyColor' , u'TransparencyColor' , ), 107, (107, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'TransparencyColor' , u'TransparencyColor' , ), 107, (107, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'TransparentBackground' , u'TransparentBackground' , ), 108, (108, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'TransparentBackground' , u'TransparentBackground' , ), 108, (108, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
]

PropertyTest_vtables_dispatch_ = 1
PropertyTest_vtables_ = [
	(( u'Name' , u'pbstrRetVal' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Condition' , u'pConditionRetVal' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Value' , u'pvargRetVal' , ), 3, (3, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'SecondValue' , u'pvargRetVal2' , ), 4, (4, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Connector' , u'pConnector' , ), 5, (5, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

PropertyTests_vtables_dispatch_ = 1
PropertyTests_vtables_ = [
	(( u'Item' , u'Index' , u'lcid' , u'ppIDocProp' , ), 0, (0, (), [
			(3, 1, None, None) , (3, 5, None, None) , (16393, 10, None, "IID('{000C0333-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pc' , ), 4, (4, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'Name' , u'Condition' , u'Value' , u'SecondValue' ,
			u'Connector' , ), 5, (5, (), [ (8, 1, None, None) , (3, 1, None, None) , (12, 17, None, None) ,
			(12, 17, None, None) , (3, 49, '1', None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Remove' , u'Index' , ), 6, (6, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppunkEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 1024 , )),
]

ScopeFolder_vtables_dispatch_ = 1
ScopeFolder_vtables_ = [
	(( u'Name' , u'pbstrName' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Path' , u'pbstrPath' , ), 2, (2, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'ScopeFolders' , u'ScopeFoldersRet' , ), 3, (3, (), [ (16393, 10, None, "IID('{000C0369-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'AddToSearchFolders' , ), 4, (4, (), [ ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
]

ScopeFolders_vtables_dispatch_ = 1
ScopeFolders_vtables_ = [
	(( u'Item' , u'Index' , u'ScopeFolderRet' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C0368-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'iCountRetVal' , ), 4, (4, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppunkEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 1024 , )),
]

Script_vtables_dispatch_ = 1
Script_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1610809344, (1610809344, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Extended' , u'Extended' , ), 1610809345, (1610809345, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Extended' , u'Extended' , ), 1610809345, (1610809345, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Id' , u'Id' , ), 1610809347, (1610809347, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Id' , u'Id' , ), 1610809347, (1610809347, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Language' , u'Language' , ), 1610809349, (1610809349, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Language' , u'Language' , ), 1610809349, (1610809349, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Location' , u'Location' , ), 1610809351, (1610809351, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 1610809352, (1610809352, (), [ ], 1 , 1 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Shape' , u'Object' , ), 1610809353, (1610809353, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'ScriptText' , u'Script' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'ScriptText' , u'Script' , ), 0, (0, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
]

Scripts_vtables_dispatch_ = 1
Scripts_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1610809344, (1610809344, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'Count' , ), 1610809345, (1610809345, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'_NewEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 1024 , )),
	(( u'Item' , u'Index' , u'Item' , ), 0, (0, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C0341-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'Anchor' , u'Location' , u'Language' , u'Id' ,
			u'Extended' , u'ScriptText' , u'Add' , ), 1610809348, (1610809348, (), [ (9, 49, 'None', None) ,
			(3, 49, '2', None) , (3, 49, '2', None) , (8, 49, "u''", None) , (8, 49, "u''", None) , (8, 49, "u''", None) ,
			(16393, 10, None, "IID('{000C0341-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 52 , (3, 32, None, None) , 0 , )),
	(( u'Delete' , ), 1610809349, (1610809349, (), [ ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
]

SearchFolders_vtables_dispatch_ = 1
SearchFolders_vtables_ = [
	(( u'Item' , u'Index' , u'ScopeFolderRet' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C0368-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'iCountRetVal' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'ScopeFolder' , ), 3, (3, (), [ (9, 1, None, "IID('{000C0368-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Remove' , u'Index' , ), 4, (4, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppunkEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 1024 , )),
]

SearchScope_vtables_dispatch_ = 1
SearchScope_vtables_ = [
	(( u'Type' , u'MsoSearchInRetVal' , ), 0, (0, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'ScopeFolder' , u'ScopeFolderRet' , ), 1, (1, (), [ (16393, 10, None, "IID('{000C0368-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
]

SearchScopes_vtables_dispatch_ = 1
SearchScopes_vtables_ = [
	(( u'Item' , u'Index' , u'SearchScopeRet' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C0367-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'iCountRetVal' , ), 4, (4, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppunkEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 1024 , )),
]

ShadowFormat_vtables_dispatch_ = 1
ShadowFormat_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'IncrementOffsetX' , u'Increment' , ), 10, (10, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'IncrementOffsetY' , u'Increment' , ), 11, (11, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ForeColor' , u'ForeColor' , ), 100, (100, (), [ (16393, 10, None, "IID('{000C0312-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'ForeColor' , u'ForeColor' , ), 100, (100, (), [ (9, 1, None, "IID('{000C0312-0000-0000-C000-000000000046}')") , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Obscured' , u'Obscured' , ), 101, (101, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Obscured' , u'Obscured' , ), 101, (101, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'OffsetX' , u'OffsetX' , ), 102, (102, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'OffsetX' , u'OffsetX' , ), 102, (102, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'OffsetY' , u'OffsetY' , ), 103, (103, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'OffsetY' , u'OffsetY' , ), 103, (103, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'Transparency' , u'Transparency' , ), 104, (104, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Transparency' , u'Transparency' , ), 104, (104, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'Type' , ), 105, (105, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'Type' , ), 105, (105, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 106, (106, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 106, (106, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
]

Shape_vtables_dispatch_ = 1
Shape_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Apply' , ), 10, (10, (), [ ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 11, (11, (), [ ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Duplicate' , u'Duplicate' , ), 12, (12, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Flip' , u'FlipCmd' , ), 13, (13, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'IncrementLeft' , u'Increment' , ), 14, (14, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'IncrementRotation' , u'Increment' , ), 15, (15, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'IncrementTop' , u'Increment' , ), 16, (16, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'PickUp' , ), 17, (17, (), [ ], 1 , 1 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'RerouteConnections' , ), 18, (18, (), [ ], 1 , 1 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'ScaleHeight' , u'Factor' , u'RelativeToOriginalSize' , u'fScale' , ), 19, (19, (), [
			(4, 1, None, None) , (3, 1, None, None) , (3, 49, '0', None) , ], 1 , 1 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'ScaleWidth' , u'Factor' , u'RelativeToOriginalSize' , u'fScale' , ), 20, (20, (), [
			(4, 1, None, None) , (3, 1, None, None) , (3, 49, '0', None) , ], 1 , 1 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Select' , u'Replace' , ), 21, (21, (), [ (12, 17, None, None) , ], 1 , 1 , 4 , 1 , 84 , (3, 0, None, None) , 0 , )),
	(( u'SetShapesDefaultProperties' , ), 22, (22, (), [ ], 1 , 1 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'Ungroup' , u'Ungroup' , ), 23, (23, (), [ (16393, 10, None, "IID('{000C031D-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'ZOrder' , u'ZOrderCmd' , ), 24, (24, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'Adjustments' , u'Adjustments' , ), 100, (100, (), [ (16393, 10, None, "IID('{000C0310-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'AutoShapeType' , u'AutoShapeType' , ), 101, (101, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'AutoShapeType' , u'AutoShapeType' , ), 101, (101, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'BlackWhiteMode' , u'BlackWhiteMode' , ), 102, (102, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'BlackWhiteMode' , u'BlackWhiteMode' , ), 102, (102, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'Callout' , u'Callout' , ), 103, (103, (), [ (16393, 10, None, "IID('{000C0311-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'ConnectionSiteCount' , u'ConnectionSiteCount' , ), 104, (104, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'Connector' , u'Connector' , ), 105, (105, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'ConnectorFormat' , u'ConnectorFormat' , ), 106, (106, (), [ (16393, 10, None, "IID('{000C0313-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'Fill' , u'Fill' , ), 107, (107, (), [ (16393, 10, None, "IID('{000C0314-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'GroupItems' , u'GroupItems' , ), 108, (108, (), [ (16393, 10, None, "IID('{000C0316-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'Height' , ), 109, (109, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 144 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'Height' , ), 109, (109, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 148 , (3, 0, None, None) , 0 , )),
	(( u'HorizontalFlip' , u'HorizontalFlip' , ), 110, (110, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 152 , (3, 0, None, None) , 0 , )),
	(( u'Left' , u'Left' , ), 111, (111, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 156 , (3, 0, None, None) , 0 , )),
	(( u'Left' , u'Left' , ), 111, (111, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 160 , (3, 0, None, None) , 0 , )),
	(( u'Line' , u'Line' , ), 112, (112, (), [ (16393, 10, None, "IID('{000C0317-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 164 , (3, 0, None, None) , 0 , )),
	(( u'LockAspectRatio' , u'LockAspectRatio' , ), 113, (113, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 168 , (3, 0, None, None) , 0 , )),
	(( u'LockAspectRatio' , u'LockAspectRatio' , ), 113, (113, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 172 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'Name' , ), 115, (115, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 176 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'Name' , ), 115, (115, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 180 , (3, 0, None, None) , 0 , )),
	(( u'Nodes' , u'Nodes' , ), 116, (116, (), [ (16393, 10, None, "IID('{000C0319-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 184 , (3, 0, None, None) , 0 , )),
	(( u'Rotation' , u'Rotation' , ), 117, (117, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 188 , (3, 0, None, None) , 0 , )),
	(( u'Rotation' , u'Rotation' , ), 117, (117, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 192 , (3, 0, None, None) , 0 , )),
	(( u'PictureFormat' , u'Picture' , ), 118, (118, (), [ (16393, 10, None, "IID('{000C031A-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 196 , (3, 0, None, None) , 0 , )),
	(( u'Shadow' , u'Shadow' , ), 119, (119, (), [ (16393, 10, None, "IID('{000C031B-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 200 , (3, 0, None, None) , 0 , )),
	(( u'TextEffect' , u'TextEffect' , ), 120, (120, (), [ (16393, 10, None, "IID('{000C031F-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 204 , (3, 0, None, None) , 0 , )),
	(( u'TextFrame' , u'TextFrame' , ), 121, (121, (), [ (16393, 10, None, "IID('{000C0320-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 208 , (3, 0, None, None) , 0 , )),
	(( u'ThreeD' , u'ThreeD' , ), 122, (122, (), [ (16393, 10, None, "IID('{000C0321-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 212 , (3, 0, None, None) , 0 , )),
	(( u'Top' , u'Top' , ), 123, (123, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 216 , (3, 0, None, None) , 0 , )),
	(( u'Top' , u'Top' , ), 123, (123, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 220 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'Type' , ), 124, (124, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 224 , (3, 0, None, None) , 0 , )),
	(( u'VerticalFlip' , u'VerticalFlip' , ), 125, (125, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 228 , (3, 0, None, None) , 0 , )),
	(( u'Vertices' , u'Vertices' , ), 126, (126, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 232 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 127, (127, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 236 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 127, (127, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 240 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'Width' , ), 128, (128, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 244 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'Width' , ), 128, (128, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 248 , (3, 0, None, None) , 0 , )),
	(( u'ZOrderPosition' , u'ZOrderPosition' , ), 129, (129, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 252 , (3, 0, None, None) , 0 , )),
	(( u'Script' , u'Script' , ), 130, (130, (), [ (16393, 10, None, "IID('{000C0341-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 256 , (3, 0, None, None) , 0 , )),
	(( u'AlternativeText' , u'AlternativeText' , ), 131, (131, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 260 , (3, 0, None, None) , 0 , )),
	(( u'AlternativeText' , u'AlternativeText' , ), 131, (131, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 264 , (3, 0, None, None) , 0 , )),
	(( u'HasDiagram' , u'pHasDiagram' , ), 132, (132, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 268 , (3, 0, None, None) , 0 , )),
	(( u'Diagram' , u'Diagram' , ), 133, (133, (), [ (16393, 10, None, "IID('{000C036D-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 272 , (3, 0, None, None) , 0 , )),
	(( u'HasDiagramNode' , u'pHasDiagram' , ), 134, (134, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 276 , (3, 0, None, None) , 0 , )),
	(( u'DiagramNode' , u'DiagramNode' , ), 135, (135, (), [ (16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 280 , (3, 0, None, None) , 0 , )),
	(( u'Child' , u'Child' , ), 136, (136, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 284 , (3, 0, None, None) , 0 , )),
	(( u'ParentGroup' , u'Parent' , ), 137, (137, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 288 , (3, 0, None, None) , 0 , )),
	(( u'CanvasItems' , u'CanvasShapes' , ), 138, (138, (), [ (16393, 10, None, "IID('{000C0371-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 292 , (3, 0, None, None) , 64 , )),
	(( u'Id' , u'pid' , ), 139, (139, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 296 , (3, 0, None, None) , 0 , )),
	(( u'CanvasCropLeft' , u'Increment' , ), 140, (140, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 300 , (3, 0, None, None) , 64 , )),
	(( u'CanvasCropTop' , u'Increment' , ), 141, (141, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 304 , (3, 0, None, None) , 64 , )),
	(( u'CanvasCropRight' , u'Increment' , ), 142, (142, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 308 , (3, 0, None, None) , 64 , )),
	(( u'CanvasCropBottom' , u'Increment' , ), 143, (143, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 312 , (3, 0, None, None) , 64 , )),
	(( u'RTF' , ), 144, (144, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 316 , (3, 0, None, None) , 64 , )),
]

ShapeNode_vtables_dispatch_ = 1
ShapeNode_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'EditingType' , u'EditingType' , ), 100, (100, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Points' , u'Points' , ), 101, (101, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'SegmentType' , u'SegmentType' , ), 102, (102, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
]

ShapeNodes_vtables_dispatch_ = 1
ShapeNodes_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'Count' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'Item' , ), 0, (0, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C0318-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'_NewEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 1024 , )),
	(( u'Delete' , u'Index' , ), 11, (11, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Insert' , u'Index' , u'SegmentType' , u'EditingType' , u'X1' ,
			u'Y1' , u'X2' , u'Y2' , u'X3' , u'Y3' ,
			), 12, (12, (), [ (3, 1, None, None) , (3, 1, None, None) , (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 49, '0.0', None) , (4, 49, '0.0', None) , (4, 49, '0.0', None) , (4, 49, '0.0', None) , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'SetEditingType' , u'Index' , u'EditingType' , ), 13, (13, (), [ (3, 1, None, None) ,
			(3, 1, None, None) , ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'SetPosition' , u'Index' , u'X1' , u'Y1' , ), 14, (14, (), [
			(3, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'SetSegmentType' , u'Index' , u'SegmentType' , ), 15, (15, (), [ (3, 1, None, None) ,
			(3, 1, None, None) , ], 1 , 1 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
]

ShapeRange_vtables_dispatch_ = 1
ShapeRange_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'Count' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'Item' , ), 0, (0, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'_NewEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 1024 , )),
	(( u'Align' , u'AlignCmd' , u'RelativeTo' , ), 10, (10, (), [ (3, 1, None, None) ,
			(3, 1, None, None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Apply' , ), 11, (11, (), [ ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 12, (12, (), [ ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Distribute' , u'DistributeCmd' , u'RelativeTo' , ), 13, (13, (), [ (3, 1, None, None) ,
			(3, 1, None, None) , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Duplicate' , u'Duplicate' , ), 14, (14, (), [ (16393, 10, None, "IID('{000C031D-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Flip' , u'FlipCmd' , ), 15, (15, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'IncrementLeft' , u'Increment' , ), 16, (16, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'IncrementRotation' , u'Increment' , ), 17, (17, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'IncrementTop' , u'Increment' , ), 18, (18, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'Group' , u'Group' , ), 19, (19, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'PickUp' , ), 20, (20, (), [ ], 1 , 1 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'Regroup' , u'Regroup' , ), 21, (21, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'RerouteConnections' , ), 22, (22, (), [ ], 1 , 1 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'ScaleHeight' , u'Factor' , u'RelativeToOriginalSize' , u'fScale' , ), 23, (23, (), [
			(4, 1, None, None) , (3, 1, None, None) , (3, 49, '0', None) , ], 1 , 1 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'ScaleWidth' , u'Factor' , u'RelativeToOriginalSize' , u'fScale' , ), 24, (24, (), [
			(4, 1, None, None) , (3, 1, None, None) , (3, 49, '0', None) , ], 1 , 1 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'Select' , u'Replace' , ), 25, (25, (), [ (12, 17, None, None) , ], 1 , 1 , 4 , 1 , 112 , (3, 0, None, None) , 0 , )),
	(( u'SetShapesDefaultProperties' , ), 26, (26, (), [ ], 1 , 1 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'Ungroup' , u'Ungroup' , ), 27, (27, (), [ (16393, 10, None, "IID('{000C031D-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'ZOrder' , u'ZOrderCmd' , ), 28, (28, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'Adjustments' , u'Adjustments' , ), 100, (100, (), [ (16393, 10, None, "IID('{000C0310-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'AutoShapeType' , u'AutoShapeType' , ), 101, (101, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'AutoShapeType' , u'AutoShapeType' , ), 101, (101, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'BlackWhiteMode' , u'BlackWhiteMode' , ), 102, (102, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
	(( u'BlackWhiteMode' , u'BlackWhiteMode' , ), 102, (102, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 144 , (3, 0, None, None) , 0 , )),
	(( u'Callout' , u'Callout' , ), 103, (103, (), [ (16393, 10, None, "IID('{000C0311-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 148 , (3, 0, None, None) , 0 , )),
	(( u'ConnectionSiteCount' , u'ConnectionSiteCount' , ), 104, (104, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 152 , (3, 0, None, None) , 0 , )),
	(( u'Connector' , u'Connector' , ), 105, (105, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 156 , (3, 0, None, None) , 0 , )),
	(( u'ConnectorFormat' , u'ConnectorFormat' , ), 106, (106, (), [ (16393, 10, None, "IID('{000C0313-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 160 , (3, 0, None, None) , 0 , )),
	(( u'Fill' , u'Fill' , ), 107, (107, (), [ (16393, 10, None, "IID('{000C0314-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 164 , (3, 0, None, None) , 0 , )),
	(( u'GroupItems' , u'GroupItems' , ), 108, (108, (), [ (16393, 10, None, "IID('{000C0316-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 168 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'Height' , ), 109, (109, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 172 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'Height' , ), 109, (109, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 176 , (3, 0, None, None) , 0 , )),
	(( u'HorizontalFlip' , u'HorizontalFlip' , ), 110, (110, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 180 , (3, 0, None, None) , 0 , )),
	(( u'Left' , u'Left' , ), 111, (111, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 184 , (3, 0, None, None) , 0 , )),
	(( u'Left' , u'Left' , ), 111, (111, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 188 , (3, 0, None, None) , 0 , )),
	(( u'Line' , u'Line' , ), 112, (112, (), [ (16393, 10, None, "IID('{000C0317-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 192 , (3, 0, None, None) , 0 , )),
	(( u'LockAspectRatio' , u'LockAspectRatio' , ), 113, (113, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 196 , (3, 0, None, None) , 0 , )),
	(( u'LockAspectRatio' , u'LockAspectRatio' , ), 113, (113, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 200 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'Name' , ), 115, (115, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 204 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'Name' , ), 115, (115, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 208 , (3, 0, None, None) , 0 , )),
	(( u'Nodes' , u'Nodes' , ), 116, (116, (), [ (16393, 10, None, "IID('{000C0319-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 212 , (3, 0, None, None) , 0 , )),
	(( u'Rotation' , u'Rotation' , ), 117, (117, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 216 , (3, 0, None, None) , 0 , )),
	(( u'Rotation' , u'Rotation' , ), 117, (117, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 220 , (3, 0, None, None) , 0 , )),
	(( u'PictureFormat' , u'Picture' , ), 118, (118, (), [ (16393, 10, None, "IID('{000C031A-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 224 , (3, 0, None, None) , 0 , )),
	(( u'Shadow' , u'Shadow' , ), 119, (119, (), [ (16393, 10, None, "IID('{000C031B-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 228 , (3, 0, None, None) , 0 , )),
	(( u'TextEffect' , u'TextEffect' , ), 120, (120, (), [ (16393, 10, None, "IID('{000C031F-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 232 , (3, 0, None, None) , 0 , )),
	(( u'TextFrame' , u'TextFrame' , ), 121, (121, (), [ (16393, 10, None, "IID('{000C0320-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 236 , (3, 0, None, None) , 0 , )),
	(( u'ThreeD' , u'ThreeD' , ), 122, (122, (), [ (16393, 10, None, "IID('{000C0321-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 240 , (3, 0, None, None) , 0 , )),
	(( u'Top' , u'Top' , ), 123, (123, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 244 , (3, 0, None, None) , 0 , )),
	(( u'Top' , u'Top' , ), 123, (123, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 248 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'Type' , ), 124, (124, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 252 , (3, 0, None, None) , 0 , )),
	(( u'VerticalFlip' , u'VerticalFlip' , ), 125, (125, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 256 , (3, 0, None, None) , 0 , )),
	(( u'Vertices' , u'Vertices' , ), 126, (126, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 260 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 127, (127, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 264 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 127, (127, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 268 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'Width' , ), 128, (128, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 272 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'Width' , ), 128, (128, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 276 , (3, 0, None, None) , 0 , )),
	(( u'ZOrderPosition' , u'ZOrderPosition' , ), 129, (129, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 280 , (3, 0, None, None) , 0 , )),
	(( u'Script' , u'Script' , ), 130, (130, (), [ (16393, 10, None, "IID('{000C0341-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 284 , (3, 0, None, None) , 0 , )),
	(( u'AlternativeText' , u'AlternativeText' , ), 131, (131, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 288 , (3, 0, None, None) , 0 , )),
	(( u'AlternativeText' , u'AlternativeText' , ), 131, (131, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 292 , (3, 0, None, None) , 0 , )),
	(( u'HasDiagram' , u'pHasDiagram' , ), 132, (132, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 296 , (3, 0, None, None) , 0 , )),
	(( u'Diagram' , u'Diagram' , ), 133, (133, (), [ (16393, 10, None, "IID('{000C036D-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 300 , (3, 0, None, None) , 0 , )),
	(( u'HasDiagramNode' , u'pHasDiagram' , ), 134, (134, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 304 , (3, 0, None, None) , 0 , )),
	(( u'DiagramNode' , u'DiagramNode' , ), 135, (135, (), [ (16393, 10, None, "IID('{000C0370-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 308 , (3, 0, None, None) , 0 , )),
	(( u'Child' , u'Child' , ), 136, (136, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 312 , (3, 0, None, None) , 0 , )),
	(( u'ParentGroup' , u'Parent' , ), 137, (137, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 316 , (3, 0, None, None) , 0 , )),
	(( u'CanvasItems' , u'CanvasShapes' , ), 138, (138, (), [ (16393, 10, None, "IID('{000C0371-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 320 , (3, 0, None, None) , 64 , )),
	(( u'Id' , u'pid' , ), 139, (139, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 324 , (3, 0, None, None) , 0 , )),
	(( u'CanvasCropLeft' , u'Increment' , ), 140, (140, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 328 , (3, 0, None, None) , 64 , )),
	(( u'CanvasCropTop' , u'Increment' , ), 141, (141, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 332 , (3, 0, None, None) , 64 , )),
	(( u'CanvasCropRight' , u'Increment' , ), 142, (142, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 336 , (3, 0, None, None) , 64 , )),
	(( u'CanvasCropBottom' , u'Increment' , ), 143, (143, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 340 , (3, 0, None, None) , 64 , )),
	(( u'RTF' , ), 144, (144, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 344 , (3, 0, None, None) , 64 , )),
]

Shapes_vtables_dispatch_ = 1
Shapes_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'Count' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'Item' , ), 0, (0, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'_NewEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 1024 , )),
	(( u'AddCallout' , u'Type' , u'Left' , u'Top' , u'Width' ,
			u'Height' , u'Callout' , ), 10, (10, (), [ (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'AddConnector' , u'Type' , u'BeginX' , u'BeginY' , u'EndX' ,
			u'EndY' , u'Connector' , ), 11, (11, (), [ (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'AddCurve' , u'SafeArrayOfPoints' , u'Curve' , ), 12, (12, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'AddLabel' , u'Orientation' , u'Left' , u'Top' , u'Width' ,
			u'Height' , u'Label' , ), 13, (13, (), [ (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'AddLine' , u'BeginX' , u'BeginY' , u'EndX' , u'EndY' ,
			u'Line' , ), 14, (14, (), [ (4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'AddPicture' , u'FileName' , u'LinkToFile' , u'SaveWithDocument' , u'Left' ,
			u'Top' , u'Width' , u'Height' , u'Picture' , ), 15, (15, (), [
			(8, 1, None, None) , (3, 1, None, None) , (3, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) ,
			(4, 49, '-1.0', None) , (4, 49, '-1.0', None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'AddPolyline' , u'SafeArrayOfPoints' , u'Polyline' , ), 16, (16, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'AddShape' , u'Type' , u'Left' , u'Top' , u'Width' ,
			u'Height' , u'Shape' , ), 17, (17, (), [ (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'AddTextEffect' , u'PresetTextEffect' , u'Text' , u'FontName' , u'FontSize' ,
			u'FontBold' , u'FontItalic' , u'Left' , u'Top' , u'TextEffect' ,
			), 18, (18, (), [ (3, 1, None, None) , (8, 1, None, None) , (8, 1, None, None) , (4, 1, None, None) ,
			(3, 1, None, None) , (3, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'AddTextbox' , u'Orientation' , u'Left' , u'Top' , u'Width' ,
			u'Height' , u'Textbox' , ), 19, (19, (), [ (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'BuildFreeform' , u'EditingType' , u'X1' , u'Y1' , u'FreeformBuilder' ,
			), 20, (20, (), [ (3, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C0315-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'Range' , u'Index' , u'Range' , ), 21, (21, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C031D-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'SelectAll' , ), 22, (22, (), [ ], 1 , 1 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'Background' , u'Background' , ), 100, (100, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'Default' , u'Default' , ), 101, (101, (), [ (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'AddDiagram' , u'Type' , u'Left' , u'Top' , u'Width' ,
			u'Height' , u'Diagram' , ), 23, (23, (), [ (3, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'AddCanvas' , u'Left' , u'Top' , u'Width' , u'Height' ,
			u'Shape' , ), 25, (25, (), [ (4, 1, None, None) , (4, 1, None, None) , (4, 1, None, None) ,
			(4, 1, None, None) , (16393, 10, None, "IID('{000C031C-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 116 , (3, 0, None, None) , 64 , )),
]

SharedWorkspace_vtables_dispatch_ = 1
SharedWorkspace_vtables_ = [
	(( u'Name' , u'Name' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'Name' , ), 0, (0, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Members' , u'ppMembers' , ), 1, (1, (), [ (16393, 10, None, "IID('{000C0382-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Tasks' , u'ppTasks' , ), 2, (2, (), [ (16393, 10, None, "IID('{000C037A-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Files' , u'ppFiles' , ), 3, (3, (), [ (16393, 10, None, "IID('{000C037C-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Folders' , u'ppFolders' , ), 4, (4, (), [ (16393, 10, None, "IID('{000C037E-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Links' , u'ppLinks' , ), 5, (5, (), [ (16393, 10, None, "IID('{000C0380-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Refresh' , ), 6, (6, (), [ ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'CreateNew' , u'URL' , u'Name' , ), 7, (7, (), [ (12, 17, None, None) ,
			(12, 17, None, None) , ], 1 , 1 , 4 , 2 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 8, (8, (), [ ], 1 , 1 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 9, (9, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'URL' , u'pbstrUrl' , ), 10, (10, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Connected' , u'pfConnected' , ), 11, (11, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'LastRefreshed' , u'pvarLastRefreshed' , ), 12, (12, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'SourceURL' , u'pbstrSourceURL' , ), 13, (13, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'SourceURL' , u'pbstrSourceURL' , ), 13, (13, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'RemoveDocument' , ), 14, (14, (), [ ], 1 , 1 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'Disconnect' , ), 15, (15, (), [ ], 1 , 1 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
]

SharedWorkspaceFile_vtables_dispatch_ = 1
SharedWorkspaceFile_vtables_ = [
	(( u'URL' , u'pbstrFilename' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'CreatedBy' , u'pbstrCreatedBy' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'CreatedDate' , u'CreatedDate' , ), 2, (2, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ModifiedBy' , u'pbstrModifiedBy' , ), 3, (3, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'ModifiedDate' , u'ModifiedDate' , ), 4, (4, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 5, (5, (), [ ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 6, (6, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
]

SharedWorkspaceFiles_vtables_dispatch_ = 1
SharedWorkspaceFiles_vtables_ = [
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 1024 , )),
	(( u'Item' , u'Index' , u'ppidisp' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C037B-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pcItems' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'FileName' , u'ParentFolder' , u'OverwriteIfFileAlreadyExists' , u'KeepInSync' ,
			u'ppFile' , ), 2, (2, (), [ (8, 1, None, None) , (12, 17, None, None) , (12, 17, None, None) ,
			(12, 17, None, None) , (16393, 10, None, "IID('{000C037B-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 3 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 3, (3, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'ItemCountExceeded' , u'pf' , ), 4, (4, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
]

SharedWorkspaceFolder_vtables_dispatch_ = 1
SharedWorkspaceFolder_vtables_ = [
	(( u'FolderName' , u'FolderName' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , u'DeleteEventIfFolderContainsFiles' , ), 1, (1, (), [ (12, 17, None, None) , ], 1 , 1 , 4 , 1 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 2, (2, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
]

SharedWorkspaceFolders_vtables_dispatch_ = 1
SharedWorkspaceFolders_vtables_ = [
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 1024 , )),
	(( u'Item' , u'Index' , u'ppidisp' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C037D-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pcItems' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'FolderName' , u'ParentFolder' , u'ppFolder' , ), 2, (2, (), [
			(8, 1, None, None) , (12, 17, None, None) , (16393, 10, None, "IID('{000C037D-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 1 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 3, (3, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'ItemCountExceeded' , u'pf' , ), 4, (4, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
]

SharedWorkspaceLink_vtables_dispatch_ = 1
SharedWorkspaceLink_vtables_ = [
	(( u'URL' , u'URL' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'URL' , u'URL' , ), 0, (0, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Description' , u'Description' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Description' , u'Description' , ), 1, (1, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Notes' , u'Notes' , ), 2, (2, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Notes' , u'Notes' , ), 2, (2, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'CreatedBy' , u'CreatedBy' , ), 3, (3, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'CreatedDate' , u'CreatedDate' , ), 4, (4, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'ModifiedBy' , u'ModifiedBy' , ), 5, (5, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'ModifiedDate' , u'ModifiedDate' , ), 6, (6, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'Save' , ), 7, (7, (), [ ], 1 , 1 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 8, (8, (), [ ], 1 , 1 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 9, (9, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
]

SharedWorkspaceLinks_vtables_dispatch_ = 1
SharedWorkspaceLinks_vtables_ = [
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 1024 , )),
	(( u'Item' , u'Index' , u'ppidisp' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C037F-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pcItems' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'URL' , u'Description' , u'Notes' , u'ppLink' ,
			), 2, (2, (), [ (8, 1, None, None) , (12, 17, None, None) , (12, 17, None, None) , (16393, 10, None, "IID('{000C037F-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 2 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 3, (3, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'ItemCountExceeded' , u'pf' , ), 4, (4, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
]

SharedWorkspaceMember_vtables_dispatch_ = 1
SharedWorkspaceMember_vtables_ = [
	(( u'DomainName' , u'pbstrDomainName' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'pbstrName' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Email' , u'pbstrEmail' , ), 2, (2, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 3, (3, (), [ ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Id' , u'Id' , ), 4, (4, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 64 , )),
	(( u'Parent' , u'ppidisp' , ), 5, (5, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
]

SharedWorkspaceMembers_vtables_dispatch_ = 1
SharedWorkspaceMembers_vtables_ = [
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 1024 , )),
	(( u'Item' , u'Index' , u'ppidisp' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C0381-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pcItems' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'Email' , u'DomainName' , u'DisplayName' , u'Role' ,
			u'ppMember' , ), 2, (2, (), [ (8, 1, None, None) , (8, 1, None, None) , (8, 1, None, None) ,
			(12, 17, None, None) , (16393, 10, None, "IID('{000C0381-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 1 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 3, (3, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'ItemCountExceeded' , u'pf' , ), 4, (4, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
]

SharedWorkspaceTask_vtables_dispatch_ = 1
SharedWorkspaceTask_vtables_ = [
	(( u'Title' , u'Title' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Title' , u'Title' , ), 0, (0, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'AssignedTo' , u'AssignedTo' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'AssignedTo' , u'AssignedTo' , ), 1, (1, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Status' , u'Status' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Status' , u'Status' , ), 2, (2, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Priority' , u'Priority' , ), 3, (3, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Priority' , u'Priority' , ), 3, (3, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Description' , u'Description' , ), 4, (4, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Description' , u'Description' , ), 4, (4, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'DueDate' , u'DueDate' , ), 5, (5, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'DueDate' , u'DueDate' , ), 5, (5, (), [ (12, 1, None, None) , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'CreatedBy' , u'CreatedBy' , ), 6, (6, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'CreatedDate' , u'CreatedDate' , ), 7, (7, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'ModifiedBy' , u'ModifiedBy' , ), 8, (8, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'ModifiedDate' , u'ModifiedDate' , ), 9, (9, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'Save' , ), 10, (10, (), [ ], 1 , 1 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 11, (11, (), [ ], 1 , 1 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 12, (12, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
]

SharedWorkspaceTasks_vtables_dispatch_ = 1
SharedWorkspaceTasks_vtables_ = [
	(( u'Item' , u'Index' , u'ppidisp' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C0379-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pcItems' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'Title' , u'Status' , u'Priority' , u'Assignee' ,
			u'Description' , u'DueDate' , u'ppTask' , ), 2, (2, (), [ (8, 1, None, None) ,
			(12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) ,
			(16393, 10, None, "IID('{000C0379-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 5 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 3, (3, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'ItemCountExceeded' , u'pf' , ), 4, (4, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 1024 , )),
]

Signature_vtables_dispatch_ = 1
Signature_vtables_ = [
	(( u'Signer' , u'pbstr' , ), 1610809344, (1610809344, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Issuer' , u'pbstr' , ), 1610809345, (1610809345, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'ExpireDate' , u'pvarDate' , ), 1610809346, (1610809346, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'IsValid' , u'pfValid' , ), 1610809347, (1610809347, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'AttachCertificate' , u'pfAttach' , ), 1610809348, (1610809348, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'AttachCertificate' , u'pfAttach' , ), 1610809348, (1610809348, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Delete' , ), 1610809350, (1610809350, (), [ ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 1610809351, (1610809351, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'IsCertificateExpired' , u'pfExpired' , ), 1610809352, (1610809352, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'IsCertificateRevoked' , u'pfExpired' , ), 1610809353, (1610809353, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'SignDate' , u'pvarDate' , ), 1610809354, (1610809354, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
]

SignatureSet_vtables_dispatch_ = 1
SignatureSet_vtables_ = [
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 1024 , )),
	(( u'Count' , u'pcSig' , ), 1610809345, (1610809345, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'iSig' , u'ppidisp' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C0411-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'ppidisp' , ), 1610809347, (1610809347, (), [ (16393, 10, None, "IID('{000C0411-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Commit' , ), 1610809348, (1610809348, (), [ ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 1610809349, (1610809349, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
]

SmartDocument_vtables_dispatch_ = 1
SmartDocument_vtables_ = [
	(( u'SolutionID' , u'pbstrID' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'SolutionID' , u'pbstrID' , ), 1, (1, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'SolutionURL' , u'pbstrUrl' , ), 2, (2, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'SolutionURL' , u'pbstrUrl' , ), 2, (2, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'PickSolution' , u'ConsiderAllSchemas' , ), 3, (3, (), [ (11, 49, 'False', None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'RefreshPane' , ), 4, (4, (), [ ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
]

Sync_vtables_dispatch_ = 1
Sync_vtables_ = [
	(( u'Status' , u'pStatusType' , ), 0, (0, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'WorkspaceLastChangedBy' , u'pbstrWorkspaceLastChangedBy' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'LastSyncTime' , u'pdatSavedTo' , ), 2, (2, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ErrorType' , u'pErrorType' , ), 4, (4, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'GetUpdate' , ), 6, (6, (), [ ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'PutUpdate' , ), 7, (7, (), [ ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'OpenVersion' , u'SyncVersionType' , ), 8, (8, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'ResolveConflict' , u'SyncConflictResolution' , ), 9, (9, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Unsuspend' , ), 10, (10, (), [ ], 1 , 1 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 14, (14, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
]

TextEffectFormat_vtables_dispatch_ = 1
TextEffectFormat_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'ToggleVerticalText' , ), 10, (10, (), [ ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Alignment' , u'Alignment' , ), 100, (100, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Alignment' , u'Alignment' , ), 100, (100, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'FontBold' , u'FontBold' , ), 101, (101, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'FontBold' , u'FontBold' , ), 101, (101, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'FontItalic' , u'FontItalic' , ), 102, (102, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'FontItalic' , u'FontItalic' , ), 102, (102, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'FontName' , u'FontName' , ), 103, (103, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'FontName' , u'FontName' , ), 103, (103, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'FontSize' , u'FontSize' , ), 104, (104, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'FontSize' , u'FontSize' , ), 104, (104, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'KernedPairs' , u'KernedPairs' , ), 105, (105, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'KernedPairs' , u'KernedPairs' , ), 105, (105, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'NormalizedHeight' , u'NormalizedHeight' , ), 106, (106, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'NormalizedHeight' , u'NormalizedHeight' , ), 106, (106, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'PresetShape' , u'PresetShape' , ), 107, (107, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'PresetShape' , u'PresetShape' , ), 107, (107, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'PresetTextEffect' , u'Preset' , ), 108, (108, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'PresetTextEffect' , u'Preset' , ), 108, (108, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'RotatedChars' , u'RotatedChars' , ), 109, (109, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'RotatedChars' , u'RotatedChars' , ), 109, (109, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'Text' , ), 110, (110, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'Text' , ), 110, (110, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'Tracking' , u'Tracking' , ), 111, (111, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'Tracking' , u'Tracking' , ), 111, (111, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
]

TextFrame_vtables_dispatch_ = 1
TextFrame_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'MarginBottom' , u'MarginBottom' , ), 100, (100, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'MarginBottom' , u'MarginBottom' , ), 100, (100, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'MarginLeft' , u'MarginLeft' , ), 101, (101, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'MarginLeft' , u'MarginLeft' , ), 101, (101, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'MarginRight' , u'MarginRight' , ), 102, (102, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'MarginRight' , u'MarginRight' , ), 102, (102, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'MarginTop' , u'MarginTop' , ), 103, (103, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'MarginTop' , u'MarginTop' , ), 103, (103, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Orientation' , u'Orientation' , ), 104, (104, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'Orientation' , u'Orientation' , ), 104, (104, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
]

ThreeDFormat_vtables_dispatch_ = 1
ThreeDFormat_vtables_ = [
	(( u'Parent' , u'Parent' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'IncrementRotationX' , u'Increment' , ), 10, (10, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'IncrementRotationY' , u'Increment' , ), 11, (11, (), [ (4, 1, None, None) , ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ResetRotation' , ), 12, (12, (), [ ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'SetThreeDFormat' , u'PresetThreeDFormat' , ), 13, (13, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'SetExtrusionDirection' , u'PresetExtrusionDirection' , ), 14, (14, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Depth' , u'Depth' , ), 100, (100, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Depth' , u'Depth' , ), 100, (100, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'ExtrusionColor' , u'ExtrusionColor' , ), 101, (101, (), [ (16393, 10, None, "IID('{000C0312-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'ExtrusionColorType' , u'ExtrusionColorType' , ), 102, (102, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'ExtrusionColorType' , u'ExtrusionColorType' , ), 102, (102, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'Perspective' , u'Perspective' , ), 103, (103, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Perspective' , u'Perspective' , ), 103, (103, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'PresetExtrusionDirection' , u'PresetExtrusionDirection' , ), 104, (104, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'PresetLightingDirection' , u'PresetLightingDirection' , ), 105, (105, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'PresetLightingDirection' , u'PresetLightingDirection' , ), 105, (105, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'PresetLightingSoftness' , u'PresetLightingSoftness' , ), 106, (106, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'PresetLightingSoftness' , u'PresetLightingSoftness' , ), 106, (106, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'PresetMaterial' , u'PresetMaterial' , ), 107, (107, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'PresetMaterial' , u'PresetMaterial' , ), 107, (107, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'PresetThreeDFormat' , u'PresetThreeDFormat' , ), 108, (108, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'RotationX' , u'RotationX' , ), 109, (109, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'RotationX' , u'RotationX' , ), 109, (109, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'RotationY' , u'RotationY' , ), 110, (110, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'RotationY' , u'RotationY' , ), 110, (110, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 111, (111, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'Visible' , u'Visible' , ), 111, (111, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
]

UserPermission_vtables_dispatch_ = 1
UserPermission_vtables_ = [
	(( u'UserId' , u'UserId' , ), 0, (0, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Permission' , u'Permission' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Permission' , u'Permission' , ), 1, (1, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ExpirationDate' , u'ExpirationDate' , ), 2, (2, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'ExpirationDate' , u'ExpirationDate' , ), 2, (2, (), [ (12, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'ppidisp' , ), 3, (3, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Remove' , ), 4, (4, (), [ ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
]

WebComponent_vtables_dispatch_ = 1
WebComponent_vtables_ = [
	(( u'Shape' , u'RetValue' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'URL' , u'RetValue' , ), 2, (2, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'URL' , u'RetValue' , ), 2, (2, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'HTML' , u'RetValue' , ), 3, (3, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'HTML' , u'RetValue' , ), 3, (3, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'RetValue' , ), 4, (4, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'RetValue' , ), 4, (4, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'RetValue' , ), 5, (5, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'RetValue' , ), 5, (5, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'RetValue' , ), 6, (6, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'RetValue' , ), 6, (6, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'SetPlaceHolderGraphic' , u'PlaceHolderGraphic' , ), 7, (7, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'Commit' , ), 8, (8, (), [ ], 1 , 1 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'Revert' , ), 9, (9, (), [ ], 1 , 1 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
]

WebComponentFormat_vtables_dispatch_ = 1
WebComponentFormat_vtables_ = [
	(( u'Application' , u'RetValue' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Parent' , u'Parent' , ), 2, (2, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'URL' , u'RetValue' , ), 3, (3, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'URL' , u'RetValue' , ), 3, (3, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'HTML' , u'RetValue' , ), 4, (4, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'HTML' , u'RetValue' , ), 4, (4, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'RetValue' , ), 5, (5, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'RetValue' , ), 5, (5, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'RetValue' , ), 6, (6, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'RetValue' , ), 6, (6, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'RetValue' , ), 7, (7, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'RetValue' , ), 7, (7, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'PreviewGraphic' , u'retval' , ), 8, (8, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'PreviewGraphic' , u'retval' , ), 8, (8, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'LaunchPropertiesWindow' , ), 9, (9, (), [ ], 1 , 1 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
]

WebComponentProperties_vtables_dispatch_ = 1
WebComponentProperties_vtables_ = [
	(( u'Shape' , u'RetValue' , ), 1, (1, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'RetValue' , ), 2, (2, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'RetValue' , ), 2, (2, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'URL' , u'RetValue' , ), 3, (3, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'URL' , u'RetValue' , ), 3, (3, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'HTML' , u'RetValue' , ), 4, (4, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'HTML' , u'RetValue' , ), 4, (4, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'PreviewGraphic' , u'RetValue' , ), 5, (5, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'PreviewGraphic' , u'RetValue' , ), 5, (5, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'PreviewHTML' , u'RetValue' , ), 6, (6, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'PreviewHTML' , u'RetValue' , ), 6, (6, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'RetValue' , ), 7, (7, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'RetValue' , ), 7, (7, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'RetValue' , ), 8, (8, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'RetValue' , ), 8, (8, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'Tag' , u'RetValue' , ), 9, (9, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'Tag' , u'RetValue' , ), 9, (9, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
]

WebComponentWindowExternal_vtables_dispatch_ = 1
WebComponentWindowExternal_vtables_ = [
	(( u'InterfaceVersion' , u'RetValue' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'ApplicationName' , u'RetValue' , ), 2, (2, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'ApplicationVersion' , u'RetValue' , ), 3, (3, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Application' , u'RetValue' , ), 4, (4, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'CloseWindow' , ), 5, (5, (), [ ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'WebComponent' , u'RetValue' , ), 6, (6, (), [ (16393, 10, None, "IID('{000CD100-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
]

WebPageFont_vtables_dispatch_ = 1
WebPageFont_vtables_ = [
	(( u'ProportionalFont' , u'pstr' , ), 10, (10, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'ProportionalFont' , u'pstr' , ), 10, (10, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'ProportionalFontSize' , u'pf' , ), 11, (11, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ProportionalFontSize' , u'pf' , ), 11, (11, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'FixedWidthFont' , u'pstr' , ), 12, (12, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'FixedWidthFont' , u'pstr' , ), 12, (12, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'FixedWidthFontSize' , u'pf' , ), 13, (13, (), [ (16388, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'FixedWidthFontSize' , u'pf' , ), 13, (13, (), [ (4, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
]

WebPageFonts_vtables_dispatch_ = 1
WebPageFonts_vtables_ = [
	(( u'Count' , u'Count' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'Item' , ), 0, (0, (), [ (3, 1, None, None) ,
			(16393, 10, None, "IID('{000C0913-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'_NewEnum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 1024 , )),
]

_CommandBarActiveX_vtables_dispatch_ = 1
_CommandBarActiveX_vtables_ = [
	(( u'ControlCLSID' , u'pbstrClsid' , ), 1610940416, (1610940416, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 332 , (3, 0, None, None) , 0 , )),
	(( u'ControlCLSID' , u'pbstrClsid' , ), 1610940416, (1610940416, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 336 , (3, 0, None, None) , 0 , )),
	(( u'QueryControlInterface' , u'bstrIid' , u'ppUnk' , ), 1610940418, (1610940418, (), [ (8, 1, None, None) ,
			(16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 340 , (3, 0, None, None) , 0 , )),
	(( u'SetInnerObjectFactory' , u'pUnk' , ), 1610940419, (1610940419, (), [ (13, 1, None, None) , ], 1 , 1 , 4 , 0 , 344 , (3, 0, None, None) , 0 , )),
	(( u'EnsureControl' , ), 1610940420, (1610940420, (), [ ], 1 , 1 , 4 , 0 , 348 , (3, 0, None, None) , 0 , )),
	(( u'InitWith' , ), 1610940421, (1610940421, (), [ (13, 1, None, None) , ], 1 , 4 , 4 , 0 , 352 , (3, 0, None, None) , 0 , )),
]

_CommandBarButton_vtables_dispatch_ = 1
_CommandBarButton_vtables_ = [
	(( u'BuiltInFace' , u'pvarfBuiltIn' , ), 1610940416, (1610940416, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 332 , (3, 0, None, None) , 0 , )),
	(( u'BuiltInFace' , u'pvarfBuiltIn' , ), 1610940416, (1610940416, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 336 , (3, 0, None, None) , 0 , )),
	(( u'CopyFace' , ), 1610940418, (1610940418, (), [ ], 1 , 1 , 4 , 0 , 340 , (3, 0, None, None) , 0 , )),
	(( u'FaceId' , u'pid' , ), 1610940419, (1610940419, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 344 , (3, 0, None, None) , 0 , )),
	(( u'FaceId' , u'pid' , ), 1610940419, (1610940419, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 348 , (3, 0, None, None) , 0 , )),
	(( u'PasteFace' , ), 1610940421, (1610940421, (), [ ], 1 , 1 , 4 , 0 , 352 , (3, 0, None, None) , 0 , )),
	(( u'ShortcutText' , u'pbstrText' , ), 1610940422, (1610940422, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 356 , (3, 0, None, None) , 0 , )),
	(( u'ShortcutText' , u'pbstrText' , ), 1610940422, (1610940422, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 360 , (3, 0, None, None) , 0 , )),
	(( u'State' , u'pstate' , ), 1610940424, (1610940424, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 364 , (3, 0, None, None) , 0 , )),
	(( u'State' , u'pstate' , ), 1610940424, (1610940424, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 368 , (3, 0, None, None) , 0 , )),
	(( u'Style' , u'pstyle' , ), 1610940426, (1610940426, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 372 , (3, 0, None, None) , 0 , )),
	(( u'Style' , u'pstyle' , ), 1610940426, (1610940426, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 376 , (3, 0, None, None) , 0 , )),
	(( u'HyperlinkType' , u'phlType' , ), 1610940428, (1610940428, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 380 , (3, 0, None, None) , 0 , )),
	(( u'HyperlinkType' , u'phlType' , ), 1610940428, (1610940428, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 384 , (3, 0, None, None) , 0 , )),
	(( u'Picture' , u'ppdispPicture' , ), 1610940430, (1610940430, (), [ (16393, 10, None, "IID('{7BF80981-BF32-101A-8BBB-00AA00300CAB}')") , ], 1 , 2 , 4 , 0 , 388 , (3, 0, None, None) , 0 , )),
	(( u'Picture' , u'ppdispPicture' , ), 1610940430, (1610940430, (), [ (9, 1, None, "IID('{7BF80981-BF32-101A-8BBB-00AA00300CAB}')") , ], 1 , 4 , 4 , 0 , 392 , (3, 0, None, None) , 0 , )),
	(( u'Mask' , u'ppipictdispMask' , ), 1610940432, (1610940432, (), [ (16393, 10, None, "IID('{7BF80981-BF32-101A-8BBB-00AA00300CAB}')") , ], 1 , 2 , 4 , 0 , 396 , (3, 0, None, None) , 0 , )),
	(( u'Mask' , u'ppipictdispMask' , ), 1610940432, (1610940432, (), [ (9, 1, None, "IID('{7BF80981-BF32-101A-8BBB-00AA00300CAB}')") , ], 1 , 4 , 4 , 0 , 400 , (3, 0, None, None) , 0 , )),
]

_CommandBarComboBox_vtables_dispatch_ = 1
_CommandBarComboBox_vtables_ = [
	(( u'AddItem' , u'Text' , u'Index' , ), 1610940416, (1610940416, (), [ (8, 1, None, None) ,
			(12, 17, None, None) , ], 1 , 1 , 4 , 1 , 332 , (3, 0, None, None) , 0 , )),
	(( u'Clear' , ), 1610940417, (1610940417, (), [ ], 1 , 1 , 4 , 0 , 336 , (3, 0, None, None) , 0 , )),
	(( u'DropDownLines' , u'pcLines' , ), 1610940418, (1610940418, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 340 , (3, 0, None, None) , 0 , )),
	(( u'DropDownLines' , u'pcLines' , ), 1610940418, (1610940418, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 344 , (3, 0, None, None) , 0 , )),
	(( u'DropDownWidth' , u'pdx' , ), 1610940420, (1610940420, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 348 , (3, 0, None, None) , 0 , )),
	(( u'DropDownWidth' , u'pdx' , ), 1610940420, (1610940420, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 352 , (3, 0, None, None) , 0 , )),
	(( u'List' , u'Index' , u'pbstrItem' , ), 1610940422, (1610940422, (), [ (3, 1, None, None) ,
			(16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 356 , (3, 0, None, None) , 0 , )),
	(( u'List' , u'Index' , u'pbstrItem' , ), 1610940422, (1610940422, (), [ (3, 1, None, None) ,
			(8, 1, None, None) , ], 1 , 4 , 4 , 0 , 360 , (3, 0, None, None) , 0 , )),
	(( u'ListCount' , u'pcItems' , ), 1610940424, (1610940424, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 364 , (3, 0, None, None) , 0 , )),
	(( u'ListHeaderCount' , u'pcItems' , ), 1610940425, (1610940425, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 368 , (3, 0, None, None) , 0 , )),
	(( u'ListHeaderCount' , u'pcItems' , ), 1610940425, (1610940425, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 372 , (3, 0, None, None) , 0 , )),
	(( u'ListIndex' , u'pi' , ), 1610940427, (1610940427, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 376 , (3, 0, None, None) , 0 , )),
	(( u'ListIndex' , u'pi' , ), 1610940427, (1610940427, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 380 , (3, 0, None, None) , 0 , )),
	(( u'RemoveItem' , u'Index' , ), 1610940429, (1610940429, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 384 , (3, 0, None, None) , 0 , )),
	(( u'Style' , u'pstyle' , ), 1610940430, (1610940430, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 388 , (3, 0, None, None) , 0 , )),
	(( u'Style' , u'pstyle' , ), 1610940430, (1610940430, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 392 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'pbstrText' , ), 1610940432, (1610940432, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 396 , (3, 0, None, None) , 0 , )),
	(( u'Text' , u'pbstrText' , ), 1610940432, (1610940432, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 400 , (3, 0, None, None) , 0 , )),
]

_CommandBars_vtables_dispatch_ = 1
_CommandBars_vtables_ = [
	(( u'ActionControl' , u'ppcbc' , ), 1610809344, (1610809344, (), [ (16393, 10, None, "IID('{000C0308-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'ActiveMenuBar' , u'ppcb' , ), 1610809345, (1610809345, (), [ (16393, 10, None, "IID('{000C0304-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Add' , u'Name' , u'Position' , u'MenuBar' , u'Temporary' ,
			u'ppcb' , ), 1610809346, (1610809346, (), [ (12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) ,
			(12, 17, None, None) , (16393, 10, None, "IID('{000C0304-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 4 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Count' , u'pcToolbars' , ), 1610809347, (1610809347, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'DisplayTooltips' , u'pvarfDisplayTooltips' , ), 1610809348, (1610809348, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'DisplayTooltips' , u'pvarfDisplayTooltips' , ), 1610809348, (1610809348, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'DisplayKeysInTooltips' , u'pvarfDisplayKeys' , ), 1610809350, (1610809350, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'DisplayKeysInTooltips' , u'pvarfDisplayKeys' , ), 1610809350, (1610809350, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'FindControl' , u'Type' , u'Id' , u'Tag' , u'Visible' ,
			u'ppcbc' , ), 1610809352, (1610809352, (), [ (12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) ,
			(12, 17, None, None) , (16393, 10, None, "IID('{000C0308-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 4 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'Index' , u'ppcb' , ), 0, (0, (), [ (12, 1, None, None) ,
			(16393, 10, None, "IID('{000C0304-0000-0000-C000-000000000046}')") , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'LargeButtons' , u'pvarfLargeButtons' , ), 1610809354, (1610809354, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'LargeButtons' , u'pvarfLargeButtons' , ), 1610809354, (1610809354, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'MenuAnimationStyle' , u'pma' , ), 1610809356, (1610809356, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'MenuAnimationStyle' , u'pma' , ), 1610809356, (1610809356, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'ppienum' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 1024 , )),
	(( u'Parent' , u'ppidisp' , ), 1610809359, (1610809359, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'ReleaseFocus' , ), 1610809360, (1610809360, (), [ ], 1 , 1 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'IdsString' , u'ids' , u'pbstrName' , u'pcch' , ), 1610809361, (1610809361, (), [
			(3, 1, None, None) , (16392, 2, None, None) , (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 64 , )),
	(( u'TmcGetName' , u'tmc' , u'pbstrName' , u'pcch' , ), 1610809362, (1610809362, (), [
			(3, 1, None, None) , (16392, 2, None, None) , (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 64 , )),
	(( u'AdaptiveMenus' , u'pvarfAdaptiveMenus' , ), 1610809363, (1610809363, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'AdaptiveMenus' , u'pvarfAdaptiveMenus' , ), 1610809363, (1610809363, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'FindControls' , u'Type' , u'Id' , u'Tag' , u'Visible' ,
			u'ppcbcs' , ), 1610809365, (1610809365, (), [ (12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) ,
			(12, 17, None, None) , (16393, 10, None, "IID('{000C0306-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 4 , 120 , (3, 0, None, None) , 0 , )),
	(( u'AddEx' , u'TbidOrName' , u'Position' , u'MenuBar' , u'Temporary' ,
			u'TbtrProtection' , u'ppcb' , ), 1610809366, (1610809366, (), [ (12, 17, None, None) , (12, 17, None, None) ,
			(12, 17, None, None) , (12, 17, None, None) , (12, 17, None, None) , (16393, 10, None, "IID('{000C0304-0000-0000-C000-000000000046}')") , ], 1 , 1 , 4 , 5 , 124 , (3, 0, None, None) , 64 , )),
	(( u'DisplayFonts' , u'pvarfDisplayFonts' , ), 1610809367, (1610809367, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'DisplayFonts' , u'pvarfDisplayFonts' , ), 1610809367, (1610809367, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'DisableCustomize' , u'pvarfDisableCustomize' , ), 1610809369, (1610809369, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'DisableCustomize' , u'pvarfDisableCustomize' , ), 1610809369, (1610809369, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
	(( u'DisableAskAQuestionDropdown' , u'pvarfDisableAskAQuestionDropdown' , ), 1610809371, (1610809371, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 144 , (3, 0, None, None) , 0 , )),
	(( u'DisableAskAQuestionDropdown' , u'pvarfDisableAskAQuestionDropdown' , ), 1610809371, (1610809371, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 148 , (3, 0, None, None) , 0 , )),
]

_IMsoDispObj_vtables_dispatch_ = 1
_IMsoDispObj_vtables_ = [
	(( u'Application' , u'ppidisp' , ), 1610743808, (1610743808, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Creator' , u'plCreator' , ), 1610743809, (1610743809, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
]

_IMsoOleAccDispObj_vtables_dispatch_ = 1
_IMsoOleAccDispObj_vtables_ = [
	(( u'Application' , u'ppidisp' , ), 1610809344, (1610809344, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'Creator' , u'plCreator' , ), 1610809345, (1610809345, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
]

RecordMap = {
}

CLSIDToClassMap = {
	'{000C0340-0000-0000-C000-000000000046}' : Scripts,
	'{000C0341-0000-0000-C000-000000000046}' : Script,
	'{2DF8D04D-5BFA-101B-BDE5-00AA0044DE52}' : DocumentProperties,
	'{2DF8D04E-5BFA-101B-BDE5-00AA0044DE52}' : DocumentProperty,
	'{000C0351-0000-0000-C000-000000000046}' : _CommandBarButtonEvents,
	'{000C0352-0000-0000-C000-000000000046}' : _CommandBarsEvents,
	'{000C0353-0000-0000-C000-000000000046}' : LanguageSettings,
	'{000C0354-0000-0000-C000-000000000046}' : _CommandBarComboBoxEvents,
	'{000C0356-0000-0000-C000-000000000046}' : HTMLProject,
	'{000C0357-0000-0000-C000-000000000046}' : HTMLProjectItems,
	'{000C0358-0000-0000-C000-000000000046}' : HTMLProjectItem,
	'{000C0359-0000-0000-C000-000000000046}' : IMsoDispCagNotifySink,
	'{000C035A-0000-0000-C000-000000000046}' : MsoDebugOptions,
	'{000C0360-0000-0000-C000-000000000046}' : AnswerWizard,
	'{000C0361-0000-0000-C000-000000000046}' : AnswerWizardFiles,
	'{000C0362-0000-0000-C000-000000000046}' : FileDialog,
	'{000C0363-0000-0000-C000-000000000046}' : FileDialogSelectedItems,
	'{000C0364-0000-0000-C000-000000000046}' : FileDialogFilter,
	'{000C0365-0000-0000-C000-000000000046}' : FileDialogFilters,
	'{000C0366-0000-0000-C000-000000000046}' : SearchScopes,
	'{000C0367-0000-0000-C000-000000000046}' : SearchScope,
	'{000C0368-0000-0000-C000-000000000046}' : ScopeFolder,
	'{000C0369-0000-0000-C000-000000000046}' : ScopeFolders,
	'{000C036A-0000-0000-C000-000000000046}' : SearchFolders,
	'{000C036C-0000-0000-C000-000000000046}' : FileTypes,
	'{000C036D-0000-0000-C000-000000000046}' : IMsoDiagram,
	'{000C036E-0000-0000-C000-000000000046}' : DiagramNodes,
	'{000C036F-0000-0000-C000-000000000046}' : DiagramNodeChildren,
	'{000C0370-0000-0000-C000-000000000046}' : DiagramNode,
	'{000C0371-0000-0000-C000-000000000046}' : CanvasShapes,
	'{000C0372-0000-0000-C000-000000000046}' : IMsoEServicesDialog,
	'{000C0373-0000-0000-C000-000000000046}' : WebComponentProperties,
	'{000C0375-0000-0000-C000-000000000046}' : UserPermission,
	'{000C0376-0000-0000-C000-000000000046}' : Permission,
	'{000C0377-0000-0000-C000-000000000046}' : SmartDocument,
	'{000C0379-0000-0000-C000-000000000046}' : SharedWorkspaceTask,
	'{000C037A-0000-0000-C000-000000000046}' : SharedWorkspaceTasks,
	'{000C037B-0000-0000-C000-000000000046}' : SharedWorkspaceFile,
	'{000C037C-0000-0000-C000-000000000046}' : SharedWorkspaceFiles,
	'{000C037D-0000-0000-C000-000000000046}' : SharedWorkspaceFolder,
	'{000C037E-0000-0000-C000-000000000046}' : SharedWorkspaceFolders,
	'{000C037F-0000-0000-C000-000000000046}' : SharedWorkspaceLink,
	'{000C0380-0000-0000-C000-000000000046}' : SharedWorkspaceLinks,
	'{000C0381-0000-0000-C000-000000000046}' : SharedWorkspaceMember,
	'{000C0382-0000-0000-C000-000000000046}' : SharedWorkspaceMembers,
	'{000C0385-0000-0000-C000-000000000046}' : SharedWorkspace,
	'{000C0386-0000-0000-C000-000000000046}' : Sync,
	'{000C0387-0000-0000-C000-000000000046}' : DocumentLibraryVersion,
	'{000C0388-0000-0000-C000-000000000046}' : DocumentLibraryVersions,
	'{00194002-D9C3-11D3-8D59-0050048384E3}' : ILicAgent,
	'{919AA22C-B9AD-11D3-8D59-0050048384E3}' : ILicValidator,
	'{000CD100-0000-0000-C000-000000000046}' : WebComponent,
	'{000CD101-0000-0000-C000-000000000046}' : WebComponentWindowExternal,
	'{000CD102-0000-0000-C000-000000000046}' : WebComponentFormat,
	'{000C0410-0000-0000-C000-000000000046}' : SignatureSet,
	'{000C0411-0000-0000-C000-000000000046}' : Signature,
	'{0006F01A-0000-0000-C000-000000000046}' : MsoEnvelope,
	'{000C0313-0000-0000-C000-000000000046}' : ConnectorFormat,
	'{000C0314-0000-0000-C000-000000000046}' : FillFormat,
	'{55F88890-7708-11D1-ACEB-006008961DA5}' : ICommandBarButtonEvents,
	'{55F88891-7708-11D1-ACEB-006008961DA5}' : CommandBarButton,
	'{55F88892-7708-11D1-ACEB-006008961DA5}' : ICommandBarsEvents,
	'{55F88893-7708-11D1-ACEB-006008961DA5}' : CommandBars,
	'{55F88896-7708-11D1-ACEB-006008961DA5}' : ICommandBarComboBoxEvents,
	'{55F88897-7708-11D1-ACEB-006008961DA5}' : CommandBarComboBox,
	'{4CAC6328-B9B0-11D3-8D59-0050048384E3}' : ILicWizExternal,
	'{000672AC-0000-0000-C000-000000000046}' : IMsoEnvelopeVB,
	'{000672AD-0000-0000-C000-000000000046}' : IMsoEnvelopeVBEvents,
	'{618736E0-3C3D-11CF-810C-00AA00389B71}' : IAccessible,
	'{000C1530-0000-0000-C000-000000000046}' : OfficeDataSourceObject,
	'{000C1531-0000-0000-C000-000000000046}' : ODSOColumn,
	'{000C1532-0000-0000-C000-000000000046}' : ODSOColumns,
	'{000C1533-0000-0000-C000-000000000046}' : ODSOFilter,
	'{000C1534-0000-0000-C000-000000000046}' : ODSOFilters,
	'{000C0300-0000-0000-C000-000000000046}' : _IMsoDispObj,
	'{000C0301-0000-0000-C000-000000000046}' : _IMsoOleAccDispObj,
	'{000C0302-0000-0000-C000-000000000046}' : _CommandBars,
	'{000C0304-0000-0000-C000-000000000046}' : CommandBar,
	'{000C0306-0000-0000-C000-000000000046}' : CommandBarControls,
	'{000C0308-0000-0000-C000-000000000046}' : CommandBarControl,
	'{000C030A-0000-0000-C000-000000000046}' : CommandBarPopup,
	'{000C030C-0000-0000-C000-000000000046}' : _CommandBarComboBox,
	'{000C030D-0000-0000-C000-000000000046}' : _CommandBarActiveX,
	'{000C030E-0000-0000-C000-000000000046}' : _CommandBarButton,
	'{000C0310-0000-0000-C000-000000000046}' : Adjustments,
	'{000C0311-0000-0000-C000-000000000046}' : CalloutFormat,
	'{000C0312-0000-0000-C000-000000000046}' : ColorFormat,
	'{000C0913-0000-0000-C000-000000000046}' : WebPageFont,
	'{000C0914-0000-0000-C000-000000000046}' : WebPageFonts,
	'{000C0315-0000-0000-C000-000000000046}' : FreeformBuilder,
	'{000C0316-0000-0000-C000-000000000046}' : GroupShapes,
	'{000C0317-0000-0000-C000-000000000046}' : LineFormat,
	'{000C0318-0000-0000-C000-000000000046}' : ShapeNode,
	'{000C0319-0000-0000-C000-000000000046}' : ShapeNodes,
	'{000C031A-0000-0000-C000-000000000046}' : PictureFormat,
	'{000C031B-0000-0000-C000-000000000046}' : ShadowFormat,
	'{000C031C-0000-0000-C000-000000000046}' : Shape,
	'{000C031D-0000-0000-C000-000000000046}' : ShapeRange,
	'{000C031E-0000-0000-C000-000000000046}' : Shapes,
	'{000C031F-0000-0000-C000-000000000046}' : TextEffectFormat,
	'{000C0320-0000-0000-C000-000000000046}' : TextFrame,
	'{000C0321-0000-0000-C000-000000000046}' : ThreeDFormat,
	'{000C0322-0000-0000-C000-000000000046}' : Assistant,
	'{000C0324-0000-0000-C000-000000000046}' : Balloon,
	'{000C0326-0000-0000-C000-000000000046}' : BalloonCheckboxes,
	'{000C0328-0000-0000-C000-000000000046}' : BalloonCheckbox,
	'{000C032E-0000-0000-C000-000000000046}' : BalloonLabels,
	'{000C0330-0000-0000-C000-000000000046}' : BalloonLabel,
	'{000C0331-0000-0000-C000-000000000046}' : FoundFiles,
	'{000C0332-0000-0000-C000-000000000046}' : FileSearch,
	'{000C0333-0000-0000-C000-000000000046}' : PropertyTest,
	'{000C0334-0000-0000-C000-000000000046}' : PropertyTests,
	'{000C0936-0000-0000-C000-000000000046}' : NewFile,
	'{000C0337-0000-0000-C000-000000000046}' : IFind,
	'{000C0338-0000-0000-C000-000000000046}' : IFoundFiles,
	'{000C0339-0000-0000-C000-000000000046}' : COMAddIns,
	'{000C033A-0000-0000-C000-000000000046}' : COMAddIn,
}
CLSIDToPackageMap = {}
win32com.client.CLSIDToClass.RegisterCLSIDsFromDict( CLSIDToClassMap )
VTablesToPackageMap = {}
VTablesToClassMap = {
	'{000C0340-0000-0000-C000-000000000046}' : 'Scripts',
	'{000C0341-0000-0000-C000-000000000046}' : 'Script',
	'{000C0353-0000-0000-C000-000000000046}' : 'LanguageSettings',
	'{000C0356-0000-0000-C000-000000000046}' : 'HTMLProject',
	'{000C0357-0000-0000-C000-000000000046}' : 'HTMLProjectItems',
	'{000C0358-0000-0000-C000-000000000046}' : 'HTMLProjectItem',
	'{000C0359-0000-0000-C000-000000000046}' : 'IMsoDispCagNotifySink',
	'{000C035A-0000-0000-C000-000000000046}' : 'MsoDebugOptions',
	'{000C0360-0000-0000-C000-000000000046}' : 'AnswerWizard',
	'{000C0361-0000-0000-C000-000000000046}' : 'AnswerWizardFiles',
	'{000C0362-0000-0000-C000-000000000046}' : 'FileDialog',
	'{000C0363-0000-0000-C000-000000000046}' : 'FileDialogSelectedItems',
	'{000C0364-0000-0000-C000-000000000046}' : 'FileDialogFilter',
	'{000C0365-0000-0000-C000-000000000046}' : 'FileDialogFilters',
	'{000C0366-0000-0000-C000-000000000046}' : 'SearchScopes',
	'{000C0367-0000-0000-C000-000000000046}' : 'SearchScope',
	'{000C0368-0000-0000-C000-000000000046}' : 'ScopeFolder',
	'{000C0369-0000-0000-C000-000000000046}' : 'ScopeFolders',
	'{000C036A-0000-0000-C000-000000000046}' : 'SearchFolders',
	'{000C036C-0000-0000-C000-000000000046}' : 'FileTypes',
	'{000C036D-0000-0000-C000-000000000046}' : 'IMsoDiagram',
	'{000C036E-0000-0000-C000-000000000046}' : 'DiagramNodes',
	'{000C036F-0000-0000-C000-000000000046}' : 'DiagramNodeChildren',
	'{000C0370-0000-0000-C000-000000000046}' : 'DiagramNode',
	'{000C0371-0000-0000-C000-000000000046}' : 'CanvasShapes',
	'{000C0372-0000-0000-C000-000000000046}' : 'IMsoEServicesDialog',
	'{000C0373-0000-0000-C000-000000000046}' : 'WebComponentProperties',
	'{000C0375-0000-0000-C000-000000000046}' : 'UserPermission',
	'{000C0376-0000-0000-C000-000000000046}' : 'Permission',
	'{000C0377-0000-0000-C000-000000000046}' : 'SmartDocument',
	'{000C0379-0000-0000-C000-000000000046}' : 'SharedWorkspaceTask',
	'{000C037A-0000-0000-C000-000000000046}' : 'SharedWorkspaceTasks',
	'{000C037B-0000-0000-C000-000000000046}' : 'SharedWorkspaceFile',
	'{000C037C-0000-0000-C000-000000000046}' : 'SharedWorkspaceFiles',
	'{000C037D-0000-0000-C000-000000000046}' : 'SharedWorkspaceFolder',
	'{000C037E-0000-0000-C000-000000000046}' : 'SharedWorkspaceFolders',
	'{000C037F-0000-0000-C000-000000000046}' : 'SharedWorkspaceLink',
	'{000C0380-0000-0000-C000-000000000046}' : 'SharedWorkspaceLinks',
	'{000C0381-0000-0000-C000-000000000046}' : 'SharedWorkspaceMember',
	'{000C0382-0000-0000-C000-000000000046}' : 'SharedWorkspaceMembers',
	'{000C0385-0000-0000-C000-000000000046}' : 'SharedWorkspace',
	'{000C0386-0000-0000-C000-000000000046}' : 'Sync',
	'{000C0387-0000-0000-C000-000000000046}' : 'DocumentLibraryVersion',
	'{000C0388-0000-0000-C000-000000000046}' : 'DocumentLibraryVersions',
	'{00194002-D9C3-11D3-8D59-0050048384E3}' : 'ILicAgent',
	'{919AA22C-B9AD-11D3-8D59-0050048384E3}' : 'ILicValidator',
	'{000C0300-0000-0000-C000-000000000046}' : '_IMsoDispObj',
	'{000C0301-0000-0000-C000-000000000046}' : '_IMsoOleAccDispObj',
	'{000C0302-0000-0000-C000-000000000046}' : '_CommandBars',
	'{000C0410-0000-0000-C000-000000000046}' : 'SignatureSet',
	'{000C0411-0000-0000-C000-000000000046}' : 'Signature',
	'{000C0313-0000-0000-C000-000000000046}' : 'ConnectorFormat',
	'{000C0314-0000-0000-C000-000000000046}' : 'FillFormat',
	'{55F88890-7708-11D1-ACEB-006008961DA5}' : 'ICommandBarButtonEvents',
	'{55F88892-7708-11D1-ACEB-006008961DA5}' : 'ICommandBarsEvents',
	'{55F88896-7708-11D1-ACEB-006008961DA5}' : 'ICommandBarComboBoxEvents',
	'{4CAC6328-B9B0-11D3-8D59-0050048384E3}' : 'ILicWizExternal',
	'{000672AC-0000-0000-C000-000000000046}' : 'IMsoEnvelopeVB',
	'{618736E0-3C3D-11CF-810C-00AA00389B71}' : 'IAccessible',
	'{000C0330-0000-0000-C000-000000000046}' : 'BalloonLabel',
	'{000C0331-0000-0000-C000-000000000046}' : 'FoundFiles',
	'{000C0332-0000-0000-C000-000000000046}' : 'FileSearch',
	'{000C0333-0000-0000-C000-000000000046}' : 'PropertyTest',
	'{000C0334-0000-0000-C000-000000000046}' : 'PropertyTests',
	'{000CD100-0000-0000-C000-000000000046}' : 'WebComponent',
	'{000CD101-0000-0000-C000-000000000046}' : 'WebComponentWindowExternal',
	'{000CD102-0000-0000-C000-000000000046}' : 'WebComponentFormat',
	'{000C0304-0000-0000-C000-000000000046}' : 'CommandBar',
	'{000C0306-0000-0000-C000-000000000046}' : 'CommandBarControls',
	'{000C0308-0000-0000-C000-000000000046}' : 'CommandBarControl',
	'{000C030A-0000-0000-C000-000000000046}' : 'CommandBarPopup',
	'{000C030C-0000-0000-C000-000000000046}' : '_CommandBarComboBox',
	'{000C030D-0000-0000-C000-000000000046}' : '_CommandBarActiveX',
	'{000C030E-0000-0000-C000-000000000046}' : '_CommandBarButton',
	'{000C0310-0000-0000-C000-000000000046}' : 'Adjustments',
	'{000C0311-0000-0000-C000-000000000046}' : 'CalloutFormat',
	'{000C0312-0000-0000-C000-000000000046}' : 'ColorFormat',
	'{000C0913-0000-0000-C000-000000000046}' : 'WebPageFont',
	'{000C0914-0000-0000-C000-000000000046}' : 'WebPageFonts',
	'{000C0315-0000-0000-C000-000000000046}' : 'FreeformBuilder',
	'{000C0316-0000-0000-C000-000000000046}' : 'GroupShapes',
	'{000C0317-0000-0000-C000-000000000046}' : 'LineFormat',
	'{000C0318-0000-0000-C000-000000000046}' : 'ShapeNode',
	'{000C0319-0000-0000-C000-000000000046}' : 'ShapeNodes',
	'{000C031A-0000-0000-C000-000000000046}' : 'PictureFormat',
	'{000C031B-0000-0000-C000-000000000046}' : 'ShadowFormat',
	'{000C031C-0000-0000-C000-000000000046}' : 'Shape',
	'{000C031D-0000-0000-C000-000000000046}' : 'ShapeRange',
	'{000C031E-0000-0000-C000-000000000046}' : 'Shapes',
	'{000C031F-0000-0000-C000-000000000046}' : 'TextEffectFormat',
	'{000C0320-0000-0000-C000-000000000046}' : 'TextFrame',
	'{000C0321-0000-0000-C000-000000000046}' : 'ThreeDFormat',
	'{000C0322-0000-0000-C000-000000000046}' : 'Assistant',
	'{000C0324-0000-0000-C000-000000000046}' : 'Balloon',
	'{000C0326-0000-0000-C000-000000000046}' : 'BalloonCheckboxes',
	'{000C0328-0000-0000-C000-000000000046}' : 'BalloonCheckbox',
	'{000C032E-0000-0000-C000-000000000046}' : 'BalloonLabels',
	'{000C1530-0000-0000-C000-000000000046}' : 'OfficeDataSourceObject',
	'{000C1531-0000-0000-C000-000000000046}' : 'ODSOColumn',
	'{000C1532-0000-0000-C000-000000000046}' : 'ODSOColumns',
	'{000C1533-0000-0000-C000-000000000046}' : 'ODSOFilter',
	'{000C1534-0000-0000-C000-000000000046}' : 'ODSOFilters',
	'{000C0936-0000-0000-C000-000000000046}' : 'NewFile',
	'{000C0337-0000-0000-C000-000000000046}' : 'IFind',
	'{000C0338-0000-0000-C000-000000000046}' : 'IFoundFiles',
	'{000C0339-0000-0000-C000-000000000046}' : 'COMAddIns',
	'{000C033A-0000-0000-C000-000000000046}' : 'COMAddIn',
}


NamesToIIDMap = {
	'FileDialogFilters' : '{000C0365-0000-0000-C000-000000000046}',
	'CanvasShapes' : '{000C0371-0000-0000-C000-000000000046}',
	'SharedWorkspaceFolder' : '{000C037D-0000-0000-C000-000000000046}',
	'WebPageFont' : '{000C0913-0000-0000-C000-000000000046}',
	'PropertyTest' : '{000C0333-0000-0000-C000-000000000046}',
	'BalloonCheckbox' : '{000C0328-0000-0000-C000-000000000046}',
	'DocumentLibraryVersion' : '{000C0387-0000-0000-C000-000000000046}',
	'ThreeDFormat' : '{000C0321-0000-0000-C000-000000000046}',
	'WebComponentFormat' : '{000CD102-0000-0000-C000-000000000046}',
	'ICommandBarButtonEvents' : '{55F88890-7708-11D1-ACEB-006008961DA5}',
	'FileSearch' : '{000C0332-0000-0000-C000-000000000046}',
	'FileDialogFilter' : '{000C0364-0000-0000-C000-000000000046}',
	'IFoundFiles' : '{000C0338-0000-0000-C000-000000000046}',
	'LanguageSettings' : '{000C0353-0000-0000-C000-000000000046}',
	'WebPageFonts' : '{000C0914-0000-0000-C000-000000000046}',
	'DocumentProperty' : '{2DF8D04E-5BFA-101B-BDE5-00AA0044DE52}',
	'ODSOFilters' : '{000C1534-0000-0000-C000-000000000046}',
	'_CommandBarButton' : '{000C030E-0000-0000-C000-000000000046}',
	'IMsoEnvelopeVB' : '{000672AC-0000-0000-C000-000000000046}',
	'ODSOColumn' : '{000C1531-0000-0000-C000-000000000046}',
	'HTMLProjectItem' : '{000C0358-0000-0000-C000-000000000046}',
	'PictureFormat' : '{000C031A-0000-0000-C000-000000000046}',
	'ODSOColumns' : '{000C1532-0000-0000-C000-000000000046}',
	'FileDialogSelectedItems' : '{000C0363-0000-0000-C000-000000000046}',
	'SharedWorkspaceFile' : '{000C037B-0000-0000-C000-000000000046}',
	'HTMLProject' : '{000C0356-0000-0000-C000-000000000046}',
	'ShapeNodes' : '{000C0319-0000-0000-C000-000000000046}',
	'ScopeFolder' : '{000C0368-0000-0000-C000-000000000046}',
	'DocumentLibraryVersions' : '{000C0388-0000-0000-C000-000000000046}',
	'Script' : '{000C0341-0000-0000-C000-000000000046}',
	'BalloonLabel' : '{000C0330-0000-0000-C000-000000000046}',
	'Balloon' : '{000C0324-0000-0000-C000-000000000046}',
	'CommandBarPopup' : '{000C030A-0000-0000-C000-000000000046}',
	'SearchFolders' : '{000C036A-0000-0000-C000-000000000046}',
	'IMsoEServicesDialog' : '{000C0372-0000-0000-C000-000000000046}',
	'FoundFiles' : '{000C0331-0000-0000-C000-000000000046}',
	'Permission' : '{000C0376-0000-0000-C000-000000000046}',
	'IAccessible' : '{618736E0-3C3D-11CF-810C-00AA00389B71}',
	'SharedWorkspaceFolders' : '{000C037E-0000-0000-C000-000000000046}',
	'ShapeRange' : '{000C031D-0000-0000-C000-000000000046}',
	'_CommandBarActiveX' : '{000C030D-0000-0000-C000-000000000046}',
	'FillFormat' : '{000C0314-0000-0000-C000-000000000046}',
	'BalloonLabels' : '{000C032E-0000-0000-C000-000000000046}',
	'WebComponentWindowExternal' : '{000CD101-0000-0000-C000-000000000046}',
	'HTMLProjectItems' : '{000C0357-0000-0000-C000-000000000046}',
	'ShadowFormat' : '{000C031B-0000-0000-C000-000000000046}',
	'TextEffectFormat' : '{000C031F-0000-0000-C000-000000000046}',
	'SmartDocument' : '{000C0377-0000-0000-C000-000000000046}',
	'COMAddIns' : '{000C0339-0000-0000-C000-000000000046}',
	'WebComponent' : '{000CD100-0000-0000-C000-000000000046}',
	'SharedWorkspaceMember' : '{000C0381-0000-0000-C000-000000000046}',
	'Shapes' : '{000C031E-0000-0000-C000-000000000046}',
	'WebComponentProperties' : '{000C0373-0000-0000-C000-000000000046}',
	'Signature' : '{000C0411-0000-0000-C000-000000000046}',
	'COMAddIn' : '{000C033A-0000-0000-C000-000000000046}',
	'FreeformBuilder' : '{000C0315-0000-0000-C000-000000000046}',
	'OfficeDataSourceObject' : '{000C1530-0000-0000-C000-000000000046}',
	'SearchScopes' : '{000C0366-0000-0000-C000-000000000046}',
	'IFind' : '{000C0337-0000-0000-C000-000000000046}',
	'SharedWorkspaceMembers' : '{000C0382-0000-0000-C000-000000000046}',
	'Shape' : '{000C031C-0000-0000-C000-000000000046}',
	'SharedWorkspaceTasks' : '{000C037A-0000-0000-C000-000000000046}',
	'LineFormat' : '{000C0317-0000-0000-C000-000000000046}',
	'_IMsoDispObj' : '{000C0300-0000-0000-C000-000000000046}',
	'SharedWorkspaceLinks' : '{000C0380-0000-0000-C000-000000000046}',
	'SharedWorkspaceFiles' : '{000C037C-0000-0000-C000-000000000046}',
	'MsoDebugOptions' : '{000C035A-0000-0000-C000-000000000046}',
	'FileDialog' : '{000C0362-0000-0000-C000-000000000046}',
	'ColorFormat' : '{000C0312-0000-0000-C000-000000000046}',
	'DiagramNode' : '{000C0370-0000-0000-C000-000000000046}',
	'Assistant' : '{000C0322-0000-0000-C000-000000000046}',
	'ScopeFolders' : '{000C0369-0000-0000-C000-000000000046}',
	'DiagramNodes' : '{000C036E-0000-0000-C000-000000000046}',
	'ConnectorFormat' : '{000C0313-0000-0000-C000-000000000046}',
	'NewFile' : '{000C0936-0000-0000-C000-000000000046}',
	'_CommandBarComboBox' : '{000C030C-0000-0000-C000-000000000046}',
	'ICommandBarsEvents' : '{55F88892-7708-11D1-ACEB-006008961DA5}',
	'DiagramNodeChildren' : '{000C036F-0000-0000-C000-000000000046}',
	'AnswerWizardFiles' : '{000C0361-0000-0000-C000-000000000046}',
	'IMsoDiagram' : '{000C036D-0000-0000-C000-000000000046}',
	'BalloonCheckboxes' : '{000C0326-0000-0000-C000-000000000046}',
	'_CommandBarsEvents' : '{000C0352-0000-0000-C000-000000000046}',
	'SharedWorkspaceTask' : '{000C0379-0000-0000-C000-000000000046}',
	'ShapeNode' : '{000C0318-0000-0000-C000-000000000046}',
	'ILicWizExternal' : '{4CAC6328-B9B0-11D3-8D59-0050048384E3}',
	'SignatureSet' : '{000C0410-0000-0000-C000-000000000046}',
	'SearchScope' : '{000C0367-0000-0000-C000-000000000046}',
	'CommandBar' : '{000C0304-0000-0000-C000-000000000046}',
	'Sync' : '{000C0386-0000-0000-C000-000000000046}',
	'UserPermission' : '{000C0375-0000-0000-C000-000000000046}',
	'ILicValidator' : '{919AA22C-B9AD-11D3-8D59-0050048384E3}',
	'CommandBarControls' : '{000C0306-0000-0000-C000-000000000046}',
	'TextFrame' : '{000C0320-0000-0000-C000-000000000046}',
	'ODSOFilter' : '{000C1533-0000-0000-C000-000000000046}',
	'AnswerWizard' : '{000C0360-0000-0000-C000-000000000046}',
	'SharedWorkspaceLink' : '{000C037F-0000-0000-C000-000000000046}',
	'_CommandBarButtonEvents' : '{000C0351-0000-0000-C000-000000000046}',
	'_CommandBarComboBoxEvents' : '{000C0354-0000-0000-C000-000000000046}',
	'Scripts' : '{000C0340-0000-0000-C000-000000000046}',
	'FileTypes' : '{000C036C-0000-0000-C000-000000000046}',
	'_CommandBars' : '{000C0302-0000-0000-C000-000000000046}',
	'ILicAgent' : '{00194002-D9C3-11D3-8D59-0050048384E3}',
	'ICommandBarComboBoxEvents' : '{55F88896-7708-11D1-ACEB-006008961DA5}',
	'DocumentProperties' : '{2DF8D04D-5BFA-101B-BDE5-00AA0044DE52}',
	'IMsoDispCagNotifySink' : '{000C0359-0000-0000-C000-000000000046}',
	'PropertyTests' : '{000C0334-0000-0000-C000-000000000046}',
	'CalloutFormat' : '{000C0311-0000-0000-C000-000000000046}',
	'_IMsoOleAccDispObj' : '{000C0301-0000-0000-C000-000000000046}',
	'IMsoEnvelopeVBEvents' : '{000672AD-0000-0000-C000-000000000046}',
	'Adjustments' : '{000C0310-0000-0000-C000-000000000046}',
	'GroupShapes' : '{000C0316-0000-0000-C000-000000000046}',
	'SharedWorkspace' : '{000C0385-0000-0000-C000-000000000046}',
	'CommandBarControl' : '{000C0308-0000-0000-C000-000000000046}',
}

win32com.client.constants.__dicts__.append(constants.__dict__)

