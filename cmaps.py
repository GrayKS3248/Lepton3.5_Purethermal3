# External modules
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import colormaps

# Std modules
from dataclasses import dataclass
from typing import ClassVar

def get_cmap(fmap, name):
    vals = np.zeros((len(fmap), 4))
    vals[:,2] = [(c&255)/255 for c in fmap]
    vals[:,1] = [((c>>8)&255)/255 for c in fmap]
    vals[:,0] = [((c>>16)&255)/255 for c in fmap]
    vals[:,3] = 1.0
    cmap = ListedColormap(vals, name=name)
    return cmap

class Subscriptable:
    def __class_getitem__(cls, item):
        return cls._get_child_dict()[item]

    @classmethod
    def _get_child_dict(cls):
        return {key.lower(): value for key, value in cls.__dict__.items()
                if not key.startswith('_')}

@dataclass
class Cmaps(Subscriptable):
    _ARCTIC: ClassVar[list[int]] = [15510, 16023, 81816, 147864, 213913, 279706, 345755, 411548, 477597, 543645, 609438, 675487, 741536, 807329, 873377, 939170, 1005219, 1071268, 1137061, 1203109, 1269158, 1334951, 1401000, 1466793, 1532842, 1598890, 1664683, 1665196, 1731245, 1797038, 1863086, 1928879, 1994928, 2060977, 2126770, 2192818, 2258867, 2324660, 2390709, 2456502, 2522551, 2588599, 2654392, 2720441, 2786490, 2852283, 2918331, 2984380, 3050173, 3116222, 3182015, 3248064, 3314112, 3379905, 3380418, 3446467, 3512260, 3578308, 3644101, 3710150, 3776199, 3841992, 3908040, 3974089, 4039882, 4105931, 4171724, 4237773, 4303821, 4369614, 4435663, 4501712, 4567505, 4633553, 4699346, 4765395, 4831444, 4897237, 4963285, 5029334, 5029591, 5095640, 5161433, 5227482, 5293530, 5359323, 5425372, 5490908, 5556186, 5621721, 5687000, 5752278, 5817557, 5883092, 5948371, 6013649, 6078928, 6144463, 6209741, 6275020, 6340299, 6405577, 6471112, 6536391, 6601669, 6666948, 6732483, 6732226, 6797504, 6862783, 6928318, 6993596, 7058875, 7124154, 7189432, 7254967, 7320246, 7385524, 7450803, 7516338, 7581617, 7646895, 7712174, 7777709, 7842987, 7908266, 7973545, 8038823, 8104358, 8169637, 8234915, 8300194, 8365729, 8365472, 8430750, 8496029, 8561564, 8626842, 8692121, 8757400, 8822678, 8888213, 8953492, 9018770, 9084049, 9149584, 9214863, 9280141, 9345420, 9410955, 9476233, 9541512, 9606791, 9672069, 9737604, 9802883, 9868161, 9933440, 9998975, 10064254, 10063996, 10129275, 10194810, 10260088, 10325367, 10390646, 10455924, 10521459, 10586738, 10652016, 10717295, 10782830, 10848109, 10913387, 10978666, 11044201, 11109479, 11174758, 11240037, 11305315, 11370850, 11436129, 11501407, 11566686, 11632221, 11697500, 11697242, 11762521, 11828056, 11893334, 11958613, 12023892, 12089170, 12154705, 12219984, 12285262, 12350541, 12416076, 12481355, 12546633, 12611912, 12677447, 12742725, 12808004, 12873283, 12938561, 13004096, 13069375, 13134653, 13199932, 13265467, 13330746, 13396024, 13395767, 13461302, 13526580, 13591859, 13657138, 13722674, 13788210, 13853746, 13919282, 13984818, 14050354, 14115890, 14181426, 14246962, 14312498, 14378034, 14443570, 14509106, 14574642, 14640178, 14705714, 14771250, 14836786, 14902322, 14967858, 15033394, 15098930, 15164466, 15230002, 15295538, 15361074, 15426610, 15492146, 15557682, 15623218, 15688754, 15754290, 15819826, 15885362, 15950898, 16016434, 16081970, 16147506, 16213042, 16278578, 16344114, 16409650, 16409651, 16409652, 16409653, 16409654, 16409655, 16409656, 16409657, 16409658, 16409659, 16409660, 16409661, 16410174, 16410431, 16410943, 16411456, 16411713, 16412226, 16412483, 16412996, 16413509, 16413766, 16414279, 16414536, 16415049, 16415562, 16415819, 16416332, 16416845, 16417102, 16417615, 16417872, 16418385, 16418898, 16419155, 16419668, 16419925, 16420438, 16420951, 16421208, 16421721, 16422234, 16422491, 16423004, 16423261, 16423774, 16424287, 16424544, 16425057, 16425314, 16425827, 16426340, 16426597, 16427110, 16427623, 16427880, 16428393, 16428650, 16429163, 16429676, 16429933, 16430446, 16430959, 16431216, 16431729, 16431986, 16432499, 16433012, 16433268, 16433781, 16434038, 16434551, 16435064, 16435321, 16435834, 16436347, 16436604, 16437117, 16437374, 16437887, 16438400, 16438657, 16439170, 16439427, 16439940, 16440453, 16440710, 16441223, 16441736, 16441993, 16442506, 16442763, 16443276, 16443789, 16444046, 16444559, 16444816, 16445329, 16445586, 16445587, 16445588, 16445589, 16445590, 16445591, 16445592, 16445593, 16445594, 16445595, 16445596, 16445597, 16445598, 16445599, 16445600, 16445601, 16445602, 16445603, 16445604, 16445605, 16445606, 16445607, 16445608, 16445609]
    _BLACK_HOT: ClassVar[list[int]] = [15461355, 15395562, 15329769, 15263976, 15198183, 15132390, 15066597, 15000804, 14935011, 14869218, 14803425, 14737632, 14671839, 14606046, 14540253, 14474460, 14408667, 14342874, 14277081, 14211288, 14145495, 14079702, 14013909, 13948116, 13882323, 13816530, 13750737, 13684944, 13619151, 13553358, 13487565, 13421772, 13355979, 13290186, 13224393, 13158600, 13092807, 13027014, 12961221, 12895428, 12829635, 12763842, 12698049, 12632256, 12566463, 12500670, 12434877, 12369084, 12303291, 12237498, 12171705, 12105912, 12040119, 11974326, 11908533, 11842740, 11776947, 11711154, 11645361, 11579568, 11513775, 11447982, 11382189, 11316396, 11250603, 11184810, 11119017, 11053224, 10987431, 10921638, 10855845, 10790052, 10724259, 10658466, 10592673, 10526880, 10461087, 10395294, 10329501, 10263708, 10197915, 10132122, 10066329, 10000536, 9934743, 9868950, 9803157, 9737364, 9671571, 9605778, 9539985, 9474192, 9408399, 9342606, 9276813, 9211020, 9145227, 9079434, 9013641, 8947848, 8882055, 8816262, 8750469, 8684676, 8618883, 8553090, 8487297, 8421504, 8355711, 8289918, 8224125, 8158332, 8092539, 8026746, 7960953, 7895160, 7829367, 7763574, 7697781, 7631988, 7566195, 7500402, 7434609, 7368816, 7303023, 7237230, 7171437, 7105644, 7039851, 6974058, 6908265, 6842472, 6776679, 6710886, 6645093, 6579300, 6513507, 6447714, 6381921, 6316128, 6250335, 6184542, 6118749, 6052956, 5987163, 5921370, 5855577, 5789784, 5723991, 5658198, 5592405, 5526612, 5460819, 5395026, 5329233, 5263440, 5197647, 5131854, 5066061, 5000268, 4934475, 4868682, 4802889, 4737096, 4671303, 4605510, 4539717, 4473924, 4408131, 4342338, 4276545, 4210752, 4144959, 4079166, 4013373, 3947580, 3881787, 3815994, 3750201, 3684408, 3618615, 3552822, 3487029, 3421236, 3355443, 3289650, 3223857, 3158064, 3092271, 3026478, 2960685, 2894892, 2829099, 2763306, 2697513, 2631720, 2565927, 2500134, 2434341, 2368548, 2302755, 2236962, 2171169, 2105376, 2039583, 1973790, 1907997, 1842204, 1776411, 1710618, 1644825, 1579032, 1513239, 1447446, 1381653, 1315860, 1250067, 1184274, 1118481, 1052688, 986895, 921102, 855309, 789516, 723723, 657930]
    _IRONBOW: ClassVar[list[int]] = [10, 20, 30, 37, 42, 46, 50, 54, 58, 62, 66, 70, 74, 79, 82, 65621, 65623, 131161, 131164, 196702, 262241, 262243, 327781, 393319, 458857, 524395, 589934, 655472, 721011, 786548, 852085, 852086, 917623, 1048696, 1179769, 1245307, 1376380, 1507453, 1638526, 1769600, 1835137, 1966211, 2097284, 2228357, 2359430, 2490503, 2621577, 2752649, 2883722, 3014795, 3145868, 3276941, 3408014, 3539086, 3670159, 3735696, 3866769, 3932306, 4063379, 4128915, 4259988, 4325525, 4456597, 4522134, 4653206, 4784278, 4849814, 4980887, 5111959, 5177495, 5308567, 5374104, 5505176, 5636248, 5767321, 5898393, 6029465, 6095002, 6226074, 6357147, 6488219, 6553755, 6684827, 6815899, 6946971, 7078044, 7143580, 7274652, 7340188, 7405725, 7536797, 7667869, 7798941, 7864477, 7995549, 8126621, 8257693, 8323229, 8454301, 8585373, 8650909, 8781981, 8847517, 8978589, 9044125, 9109661, 9240733, 9371804, 9502876, 9633948, 9765020, 9830555, 9961627, 10027163, 10158235, 10223771, 10289307, 10420379, 10485915, 10616987, 10682523, 10748059, 10879130, 10944666, 11010202, 11075737, 11141273, 11206809, 11337881, 11403672, 11469208, 11534744, 11600279, 11665815, 11731350, 11797142, 11862677, 11928213, 11994005, 12059541, 12125333, 12190869, 12190868, 12256659, 12322195, 12387731, 12453522, 12519058, 12584849, 12650640, 12650896, 12716687, 12782222, 12782478, 12848269, 12913804, 12979595, 12979850, 13045641, 13111432, 13177223, 13243014, 13243269, 13308805, 13309060, 13374850, 13440641, 13506432, 13506686, 13572220, 13572475, 13638265, 13704056, 13704310, 13769845, 13770100, 13835890, 13836401, 13902191, 13902446, 13968235, 13968489, 14034279, 14100069, 14165860, 14166114, 14232160, 14297950, 14298204, 14364250, 14364503, 14430036, 14495825, 14496078, 14561866, 14562119, 14627908, 14628161, 14628413, 14694202, 14694455, 14694707, 14760496, 14826285, 14826538, 14892326, 14892579, 14958368, 14958877, 14959132, 15024923, 15025177, 15025432, 15091222, 15157013, 15157268, 15157523, 15223314, 15223824, 15223823, 15289614, 15289613, 15355404, 15355660, 15421451, 15421706, 15421962, 15422217, 15488009, 15488520, 15488776, 15489032, 15554823, 15555079, 15555334, 15621126, 15621125, 15621381, 15621637, 15687428, 15687684, 15687940, 15688196, 15753987, 15754243, 15754499, 15820291, 15820547, 15820803, 15821058, 15821314, 15821570, 15887361, 15887617, 15887873, 15953665, 15953921, 15954177, 15954433, 16020224, 16020480, 16020736, 16020992, 16021248, 16021504, 16022016, 16087808, 16088064, 16088576, 16088832, 16154624, 16154880, 16155136, 16220928, 16221184, 16221440, 16221696, 16287488, 16287744, 16288000, 16288256, 16288512, 16288768, 16354560, 16354816, 16355072, 16355328, 16355584, 16355840, 16356096, 16421888, 16422144, 16422400, 16488448, 16488704, 16488960, 16489472, 16555264, 16555776, 16556032, 16556288, 16622080, 16622336, 16622592, 16623104, 16623360, 16623616, 16624128, 16624384, 16624640, 16624896, 16625152, 16690944, 16691200, 16691456, 16691712, 16691968, 16692224, 16692480, 16692736, 16693248, 16693504, 16693760, 16694016, 16694272, 16694528, 16694784, 16695296, 16695552, 16695808, 16696064, 16696320, 16696576, 16696832, 16697088, 16697344, 16697601, 16697857, 16698113, 16698370, 16698626, 16698883, 16699140, 16699397, 16699654, 16700168, 16700425, 16700682, 16700938, 16701195, 16701452, 16701709, 16767502, 16767760, 16768018, 16768020, 16768278, 16768537, 16768539, 16768798, 16769056, 16769314, 16769572, 16769574, 16769832, 16770091, 16770094, 16770353, 16770613, 16770616, 16770876, 16771135, 16771395, 16771654, 16771913, 16771917, 16772176, 16772436, 16772695, 16772699, 16772703, 16772963, 16772967, 16773226, 16773230, 16773490, 16773495, 16773499, 16773760, 16773765, 16773770, 16774030, 16774290, 16774294, 16774298, 16774558, 16774562, 16774566, 16774826, 16774831, 16775091, 16775094, 16775354, 16775357, 16775361, 16775364, 16775623, 16775626, 16775629, 16775889, 16775892, 16776152, 16776411, 16776415, 16776674, 16776677, 16776680, 16776939, 16776942, 16776945, 16776948, 16777206]
    _OUTDOOR_ALERT: ClassVar[list[int]] = [16119285, 16053492, 15987699, 15921906, 15856113, 15790320, 15724527, 15658734, 15592941, 15527148, 15461355, 15395562, 15329769, 15263976, 15198183, 15132390, 15066597, 15000804, 14935011, 14869218, 14803425, 14737632, 14671839, 14606046, 14540253, 14474460, 14408667, 14342874, 14277081, 14211288, 14145495, 14079702, 14013909, 13948116, 13882323, 13816530, 13750737, 13684944, 13619151, 13553358, 13487565, 13421772, 13355979, 13290186, 13224393, 13158600, 13092807, 13027014, 12961221, 12895428, 12829635, 12763842, 12698049, 12632256, 12566463, 12500670, 12434877, 12369084, 12303291, 12237498, 12171705, 12105912, 12040119, 11974326, 11908533, 11842740, 11776947, 11711154, 11645361, 11579568, 11513775, 11447982, 11382189, 11316396, 11250603, 11184810, 11119017, 11053224, 10987431, 10921638, 10855845, 10790052, 10724259, 10658466, 10592673, 10526880, 10461087, 10395294, 10329501, 10263708, 10197915, 10132122, 10066329, 10000536, 9934743, 9868950, 9803157, 9737364, 9671571, 9605778, 9539985, 9474192, 9408399, 9342606, 9276813, 9211020, 9145227, 9079434, 9013641, 8947848, 8882055, 8816262, 8750469, 8684676, 8618883, 8553090, 8487297, 8421504, 8355711, 8289918, 8224125, 8158332, 8092539, 8026746, 7960953, 7895160, 7829367, 7763574, 7697781, 7631988, 7566195, 7500402, 7434609, 7368816, 7303023, 7237230, 7171437, 7105644, 7039851, 6974058, 6908265, 6842472, 6776679, 6710886, 6645093, 6579300, 6513507, 6447714, 6381921, 6316128, 6250335, 6184542, 6118749, 6052956, 5987163, 5921370, 5855577, 5789784, 5723991, 5658198, 5592405, 5526612, 5460819, 5395026, 5329233, 5263440, 5197647, 5131854, 5066061, 5000268, 4934475, 4868682, 4802889, 4737096, 4671303, 4605510, 4539717, 4473924, 4408131, 4342338, 4276545, 4210752, 4144959, 4079166, 4013373, 3947580, 3881787, 3815994, 3750201, 3684408, 3618615, 3552822, 3487029, 3421236, 3355443, 3289650, 3223857, 3158064, 3092271, 3026478, 2960685, 2894892, 2829099, 2763306, 2697513, 2631720, 2565927, 2500134, 2434341, 2368548, 2302755, 2236962, 2171169, 2105376, 2039583, 1973790, 1907997, 1842204, 1776411, 1710618, 1644825, 1579032, 1513239, 1447446, 1381653, 1315860, 1250067, 1184274, 1118481, 1052688, 986895, 921102, 855309, 789516, 723723, 657930, 3934730, 4262410, 4590090, 4917770, 5245707, 5573387, 5901324, 6229261, 6557198, 6884878, 7212815, 7540752, 7868432, 8196369, 8524306, 8851986, 9179923, 9507860, 9835540, 10163477, 10491414, 10819351, 11147031, 11474968, 11737369, 12065049, 12392986, 12720923, 13048603, 13376540, 13704477, 14032157, 14755845, 14825525, 15022896, 15220266, 15417637, 15615008, 15746586, 15943957, 16141328, 16338698, 16536069, 16733440, 16733952, 16734720, 16735488, 16736256, 16737024, 16737792, 16738304, 16739072, 16739840, 16740608, 16741376, 16742144, 16742656, 16743424, 16744192, 16744960, 16745728, 16746496, 16747008, 16747776, 16748544, 16749312, 16750080, 16750848, 16751360, 16752128, 16752896, 16753664, 16754432, 16755200, 16755719, 16756495, 16757270, 16758046, 16758821, 16759597, 16760116, 16760892, 16761667, 16762443, 16763218, 16763994, 16764513, 16765289, 16766064, 16766840, 16767615, 16768391, 16768910, 16769686, 16770461, 16771237, 16772012, 16772788, 16773307, 16774083, 16774858, 16775634]
    _RAINBOW: ClassVar[list[int]] = [255, 767, 1535, 2303, 3071, 3839, 4607, 5375, 6143, 6911, 7679, 8447, 9215, 9983, 10751, 11519, 12287, 13055, 13823, 14335, 15103, 15871, 16639, 17407, 18175, 18943, 19711, 20479, 21247, 22015, 22783, 23551, 24319, 25087, 25855, 26623, 27391, 27903, 28671, 29439, 30207, 30975, 31743, 32511, 33279, 34047, 34815, 35583, 36351, 37119, 37887, 38655, 39423, 40191, 40959, 41471, 42239, 43007, 43775, 44543, 45311, 46079, 46847, 47615, 48383, 49151, 49919, 50687, 51455, 52223, 52991, 53759, 54527, 55039, 55807, 56575, 57343, 58111, 58879, 59647, 60415, 61183, 61951, 62719, 63487, 64255, 65023, 65533, 65530, 65527, 65524, 65522, 65519, 65516, 65513, 65510, 65507, 65504, 65501, 65498, 65495, 65492, 65489, 65486, 65483, 65480, 65477, 65474, 65471, 65469, 65466, 65463, 65460, 65457, 65454, 65451, 65448, 65445, 65442, 65439, 65436, 65433, 65430, 65427, 65424, 65421, 65418, 65416, 65413, 65410, 65407, 65404, 65401, 65398, 65395, 65392, 65389, 65386, 65383, 65380, 65377, 65374, 65371, 65368, 65365, 65363, 65360, 65357, 65354, 65351, 65348, 65345, 65342, 65339, 65336, 65333, 65330, 65327, 65324, 65321, 65318, 65315, 65312, 65310, 65307, 65304, 65301, 65298, 65295, 65292, 65289, 65286, 65283, 65280, 196352, 392960, 589568, 786176, 982784, 1179392, 1376000, 1507072, 1703680, 1900288, 2096896, 2293504, 2490112, 2686720, 2883328, 3079936, 3276544, 3473152, 3669760, 3866368, 4062976, 4259584, 4456192, 4652800, 4849408, 4980480, 5177088, 5373696, 5570304, 5766912, 5963520, 6160128, 6356736, 6553344, 6749952, 6946560, 7143168, 7339776, 7536384, 7732992, 7929600, 8126208, 8322816, 8453888, 8650496, 8847104, 9043712, 9240320, 9436928, 9633536, 9830144, 10026752, 10223360, 10419968, 10616576, 10813184, 11009792, 11206400, 11403008, 11599616, 11796224, 11927296, 12123904, 12320512, 12517120, 12713728, 12910336, 13106944, 13303552, 13500160, 13696768, 13893376, 14089984, 14286592, 14483200, 14679808, 14876416, 15073024, 15269632, 15400704, 15597312, 15793920, 15990528, 16187136, 16383744, 16580352, 16776704, 16775936, 16775168, 16774400, 16773632, 16772864, 16772096, 16771328, 16770560, 16769792, 16769024, 16768512, 16767744, 16766976, 16766208, 16765440, 16764672, 16763904, 16763136, 16762368, 16761600, 16760832, 16760064, 16759296, 16758528, 16757760, 16756992, 16756224, 16755456, 16754944, 16754176, 16753408, 16752640, 16751872, 16751104, 16750336, 16749568, 16748800, 16748032, 16747264, 16746496, 16745728, 16744960, 16744192, 16743424, 16742656, 16741888, 16741376, 16740608, 16739840, 16739072, 16738304, 16737536, 16736768, 16736000, 16735232, 16734464, 16733696, 16732928, 16732160, 16731392, 16730624, 16729856, 16729088, 16728320, 16727808, 16727040, 16726272, 16725504, 16724736, 16723968, 16723200, 16722432, 16721664, 16720896, 16720128, 16719360, 16718592, 16717824, 16717056, 16716288, 16715520, 16714752, 16714240, 16713472, 16712704, 16711936, 16711681, 16711684, 16711687, 16711689, 16711692, 16711695, 16711698, 16711701, 16711703, 16711706, 16711709, 16711712, 16711714, 16711717, 16711720, 16711723, 16711726, 16711728, 16711731, 16711734, 16711737, 16711739, 16711742, 16711745, 16711748, 16711750, 16711753, 16711756, 16711759, 16711762, 16711764, 16711767, 16711770, 16711773, 16711775, 16711778, 16711781, 16711784, 16711786, 16711789, 16711792, 16711795, 16711798, 16711800, 16711803, 16711806, 16711809, 16711811, 16711814, 16711817, 16711820, 16711823, 16711825, 16711828, 16711831, 16711834, 16711836, 16711839, 16711842, 16711845, 16711847, 16711850, 16711853, 16711856, 16711859, 16711861, 16711864, 16711867, 16711870, 16711872, 16711875, 16711878, 16711881, 16711883, 16711886, 16711889, 16711892, 16711895, 16711897, 16711900, 16711903, 16711906, 16711908, 16711911, 16711914, 16711917]
    _RAINBOW_HC: ClassVar[list[int]] = [150, 3211926, 3212438, 3147414, 3147926, 3083158, 3083670, 3018646, 3019158, 2954134, 2954902, 2889878, 2890390, 2825366, 2826134, 2826646, 2761622, 2762134, 2697110, 2697878, 2632854, 2633366, 2568342, 2568854, 2504086, 2504598, 2439574, 2440086, 2440854, 2375830, 2376342, 2311318, 2311830, 2247062, 2247574, 2182550, 2183062, 2118294, 2118806, 2053782, 2054294, 2054806, 1990038, 1990550, 1925526, 1926038, 1861014, 1861782, 1796758, 1797270, 1732246, 1733014, 1667990, 1668502, 1669014, 1603990, 1604758, 1539734, 1540246, 1475222, 1475990, 1410966, 1411478, 1346454, 1346966, 1282198, 1282710, 1283222, 1218198, 1218710, 1153942, 1154454, 1089430, 1089942, 1025174, 1025686, 960662, 961174, 896150, 896918, 897430, 832406, 832918, 768150, 768662, 703638, 704150, 639126, 639894, 574870, 575382, 510358, 510870, 511638, 446614, 447126, 382102, 382870, 317846, 318358, 253334, 253846, 189078, 189590, 124566, 125078, 125846, 60822, 61334, 61588, 61585, 61583, 61581, 61579, 61576, 61574, 61572, 61569, 61567, 61565, 61562, 61560, 61558, 61555, 61553, 61551, 61549, 61546, 61544, 61542, 61539, 61537, 61535, 61532, 61530, 61528, 61525, 61523, 61521, 61518, 61516, 61514, 61512, 61509, 61507, 61505, 61502, 61500, 61498, 61495, 61493, 61491, 61488, 61486, 61484, 61482, 61479, 61477, 61475, 61472, 61470, 61468, 61465, 61463, 61461, 61458, 61456, 61454, 61452, 61449, 61447, 61445, 61442, 61440, 258048, 520192, 716800, 978944, 1241088, 1503232, 1765376, 2027520, 2289664, 2551808, 2813952, 3076096, 3338240, 3600384, 3862528, 4124672, 4321280, 4583424, 4845568, 5107712, 5369856, 5632000, 5894144, 6156288, 6418432, 6680576, 6942720, 7204864, 7467008, 7663616, 7925760, 8187904, 8450048, 8712192, 8974336, 9236480, 9498624, 9760768, 10022912, 10285056, 10547200, 10809344, 11071488, 11268096, 11530240, 11792384, 12054528, 12316672, 12578816, 12840960, 13103104, 13365248, 13627392, 13889536, 14151680, 14413824, 14675968, 14872576, 15134720, 15396864, 15659008, 15921152, 16183296, 16445440, 16707584, 16772096, 16771328, 16770304, 16769280, 16768512, 16767488, 16766464, 16765440, 16764672, 16763648, 16762624, 16761856, 16760832, 16759808, 16759040, 16758016, 16756992, 16756224, 16755200, 16754176, 16753152, 16752384, 16751360, 16750336, 16749568, 16748544, 16747520, 16746752, 16745728, 16744704, 16743936, 16742912, 16741888, 16740864, 16740096, 16739072, 16738048, 16737280, 16736256, 16735232, 16734464, 16733440, 16732416, 16731392, 16730624, 16729600, 16728576, 16727808, 16726784, 16725760, 16724992, 16723968, 16722944, 16722176, 16721152, 16720128, 16719104, 16718336, 16717312, 16716288, 16715520, 16714496, 16713472, 16712704, 16711680, 16711937, 16712451, 16712964, 16713478, 16713992, 16714505, 16715019, 16715533, 16716047, 16716560, 16716818, 16717332, 16717845, 16718359, 16718873, 16719386, 16719900, 16720414, 16720928, 16721441, 16721955, 16722469, 16722982, 16723240, 16723754, 16724267, 16724781, 16725295, 16725808, 16726322, 16726836, 16727350, 16727863, 16728377, 16728891, 16729404, 16729918, 16730176, 16730689, 16731203, 16731717, 16732230, 16732744, 16733258, 16733772, 16734285, 16734799, 16735313, 16735826, 16736340, 16736598, 16737111, 16737625, 16738139, 16738652, 16739166, 16739680, 16740194, 16740707, 16741221, 16741735, 16742248, 16742762, 16743020, 16743533, 16744047, 16744561, 16745074, 16745588, 16746102, 16746616, 16747129, 16747643, 16748157, 16748670, 16749184, 16749442, 16749955, 16750469, 16750983, 16751497, 16752010, 16752524, 16753038, 16753551, 16754065, 16754579, 16755092, 16755606, 16756120, 16756377, 16756891, 16757405, 16757919, 16758432, 16758946, 16759460, 16759973, 16760487, 16761001, 16761514, 16762028, 16762542, 16762799, 16763313, 16763827, 16764341, 16764854, 16765368, 16765882, 16766395, 16766909, 16767423, 16767936, 16768450, 16768964, 16769221, 16769735, 16770249, 16770763, 16771276, 16771790, 16772304, 16772817, 16773331, 16773845, 16774358, 16774872, 16775386]
    _WHITE_HOT: ClassVar[list[int]] = [657930, 723723, 789516, 855309, 921102, 986895, 1052688, 1118481, 1184274, 1250067, 1315860, 1381653, 1447446, 1513239, 1579032, 1644825, 1710618, 1776411, 1842204, 1907997, 1973790, 2039583, 2105376, 2171169, 2236962, 2302755, 2368548, 2434341, 2500134, 2565927, 2631720, 2697513, 2763306, 2829099, 2894892, 2960685, 3026478, 3092271, 3158064, 3223857, 3289650, 3355443, 3421236, 3487029, 3552822, 3618615, 3684408, 3750201, 3815994, 3881787, 3947580, 4013373, 4079166, 4144959, 4210752, 4276545, 4342338, 4408131, 4473924, 4539717, 4605510, 4671303, 4737096, 4802889, 4868682, 4934475, 5000268, 5066061, 5131854, 5197647, 5263440, 5329233, 5395026, 5460819, 5526612, 5592405, 5658198, 5723991, 5789784, 5855577, 5921370, 5987163, 6052956, 6118749, 6184542, 6250335, 6316128, 6381921, 6447714, 6513507, 6579300, 6645093, 6710886, 6776679, 6842472, 6908265, 6974058, 7039851, 7105644, 7171437, 7237230, 7303023, 7368816, 7434609, 7500402, 7566195, 7631988, 7697781, 7763574, 7829367, 7895160, 7960953, 8026746, 8092539, 8158332, 8224125, 8289918, 8355711, 8421504, 8487297, 8553090, 8618883, 8684676, 8750469, 8816262, 8882055, 8947848, 9013641, 9079434, 9145227, 9211020, 9276813, 9342606, 9408399, 9474192, 9539985, 9605778, 9671571, 9737364, 9803157, 9868950, 9934743, 10000536, 10066329, 10132122, 10197915, 10263708, 10329501, 10395294, 10461087, 10526880, 10592673, 10658466, 10724259, 10790052, 10855845, 10921638, 10987431, 11053224, 11119017, 11184810, 11250603, 11316396, 11382189, 11447982, 11513775, 11579568, 11645361, 11711154, 11776947, 11842740, 11908533, 11974326, 12040119, 12105912, 12171705, 12237498, 12303291, 12369084, 12434877, 12500670, 12566463, 12632256, 12698049, 12763842, 12829635, 12895428, 12961221, 13027014, 13092807, 13158600, 13224393, 13290186, 13355979, 13421772, 13487565, 13553358, 13619151, 13684944, 13750737, 13816530, 13882323, 13948116, 14013909, 14079702, 14145495, 14211288, 14277081, 14342874, 14408667, 14474460, 14540253, 14606046, 14671839, 14737632, 14803425, 14869218, 14935011, 15000804, 15066597, 15132390, 15198183, 15263976, 15329769, 15395562]

    AFMHOT: ClassVar[ListedColormap] = colormaps['afmhot']
    ARCTIC: ClassVar[ListedColormap] = get_cmap(_ARCTIC, 'arctic')
    BLACK_HOT: ClassVar[ListedColormap] = get_cmap(_BLACK_HOT, 'black_hot')
    CIVIDIS: ClassVar[ListedColormap] = colormaps['cividis']
    INFERNO: ClassVar[ListedColormap] = colormaps['inferno']
    IRONBOW: ClassVar[ListedColormap] = get_cmap(_IRONBOW, 'ironbow')
    MAGMA: ClassVar[ListedColormap] = colormaps['magma']
    OUTDOOR_ALERT: ClassVar[ListedColormap] = get_cmap(_OUTDOOR_ALERT, 'outdoor_alert')
    RAINBOW: ClassVar[ListedColormap] = get_cmap(_RAINBOW, 'rainbow')
    RAINBOW_HC: ClassVar[ListedColormap] = get_cmap(_RAINBOW_HC, 'rainbow_hc')
    VIRIDIS: ClassVar[ListedColormap] = colormaps['viridis']
    WHITE_HOT: ClassVar[ListedColormap] = get_cmap(_WHITE_HOT, 'white_hot')
    