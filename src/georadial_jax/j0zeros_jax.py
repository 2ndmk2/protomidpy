import jax.numpy as jnp

J0ZEROS = jnp.array([2.404825557695773,5.520078110286311,8.653727912911013,11.791534439014281,14.930917708487787,18.071063967910924,21.21163662987926,24.352471530749302,27.493479132040257,30.634606468431976,33.77582021357357,\
36.917098353664045,40.05842576462824,43.19979171317673,46.341188371661815,49.482609897397815,52.624051841115,55.76551075501998,58.90698392608094,62.04846919022717,65.18996480020687,\
68.3314693298568,71.47298160359374,74.61450064370185,77.75602563038805,80.89755587113763,84.0390907769382,87.18062984364116,90.32217263721049,93.46371878194478,96.60526795099626,\
99.7468198586806,102.8883742541948,106.02993091645162,109.17148964980538,112.3130502804949,115.45461265366694,118.59617663087253,121.73774208795096,124.87930891323295,128.02087700600833,\
131.1624462752139,134.30401663830546,137.44558802028428,140.58716035285428,143.72873357368974,146.87030762579664,150.01188245695477,153.15345801922788,156.29503426853353,159.43661116426316,\
162.57818866894667,165.71976674795502,168.86134536923583,172.0029245030782,175.14450412190274,178.28608420007376,181.42766471373105,184.5692456406387,187.71082696004936,190.85240865258152,\
193.99399070010912,197.1355730856614,200.2771557933324,203.41873880819864,206.56032211624446,209.70190570429406,212.8434895599495,215.98507367153402,219.12665802804057,222.2682426190843,\
225.40982743485932,228.5514124660988,231.69299770403853,234.83458314038324,237.97616876727565,241.11775457726802,244.2593405632957,247.40092671865284,250.54251303696995,253.6840995121931,\
256.82568613856444,259.9672729106045,263.1088598230955,266.2504468710659,269.39203404977604,272.5336213547049,275.67520878153744,278.8167963261531,281.9583839846149,285.09997175315954,\
288.2415596281877,291.3831476062552,294.524735684065,297.66632385845895,300.80791212641117,303.9495004850206,307.09108893150506,310.232677463195,313.37426607752786,316.5158547720429,\
319.6574435443762,322.7990323922556,325.9406213134967,329.08221030599856,332.2237993677397,335.3653884967742,338.50697769122854,341.64856694929813,344.79015626924405,347.9317456493902,\
351.0733350881206,354.21492458387644,357.35651413515376,360.4981037405011,363.63969339851707,366.7812831078483,369.9228728671875,373.0644626752713,376.2060525308785,379.3476424328285,\
382.48923237997934,385.63082237122643,388.77241240550063,391.9140024817674,395.05559259902486,398.1971827563029,401.3387729526616,404.48036318719045,407.6219534590069,410.7635437672554,\
413.90513411110635,417.0467244897553,420.18831490242167,423.3299053483481,426.47149582679964,429.6130863370627,432.75467687844457,435.8962674502724,439.0378580518925,442.17944868267,\
445.3210393419878,448.46263002924604,451.6042207438617,454.7458114852678,457.8874022529128,461.02899304626044,464.1705838647888,467.31217470799004,470.45376557536986,473.59535646644713,\
476.73694738075335,479.87853831783235,483.02012927723973,486.1617202585427,489.30331126131944,492.444902285159,495.58649332966087,498.7280843944346,501.8696754790994,505.0112665832841,\
508.1528577066267,511.29444884877404,514.4360400093816,517.5776311881133,520.719222384641,523.8608135986445,527.0024048298114,530.1439960778366,533.2855873424221,536.4271786232769,\
539.5687699201169,542.7103612326644,545.8519525606483,548.9935439038036,552.1351352618711,555.276726634598,558.4183180217369,561.5599094230456,564.7015008382879,567.8430922672325,\
570.984683709653,574.1262751653285,577.2678666340423,580.409458115583,583.5510496097431,586.6926411163201,589.8342326351157,592.9758241659354,596.1174157085892,599.2590072628911,\
602.4005988286588,605.5421904057138,608.6837819938813,611.8253735929902,614.9669652028729,618.1085568233649,621.2501484543054,624.3917400955368,627.5333317469042,630.6749234082564,\
633.8165150794449,636.9581067603241,640.0996984507512,643.2412901505867,646.3828818596929,649.5244735779357,652.666065305183,655.8076570413053,658.9492487861759,662.09084053967,\
665.2324323016657,668.3740240720429,671.515615850684,674.6572076374736,677.7987994322984,680.9403912350472,684.0819830456107,687.223574863882,690.3651666897557,693.5067585231285,\
696.6483503638989,699.7899422119673,702.9315340672358,706.0731259296084,709.2147177989907,712.3563096752899,715.4979015584149,718.6394934482762,721.7810853447858,724.9226772478572,\
728.0642691574056,731.2058610733475,734.3474529956007,737.4890449240847,740.6306368587202,743.7722287994292,746.9138207461351,750.0554126987624,753.1970046572372,756.3385966214867,\
759.4801885914388,762.6217805670236,765.7633725481713,768.9049645348141,772.0465565268846,775.188148524317,778.3297405270463,781.4713325350085,784.612924548141,787.7545165663819,\
790.8961085896701,794.0377006179459,797.1792926511503,800.3208846892252,803.4624767321134,806.6040687797588,809.7456608321061,812.8872528891005,816.0288449506886,819.1704370168175,\
822.312029087435,825.45362116249,828.595213241932,831.7368053257112,834.8783974137788,838.0199895060864,841.1615816025867,844.3031737032326,847.4447658079782,850.5863579167782,\
853.7279500295875,856.8695421463623,860.011134267059,863.1527263916347,866.2943185200475,869.4359106522554,872.5775027882177,875.7190949278939,878.8606870712442,882.0022792182293,\
885.1438713688106,888.2854635229497,891.4270556806093,894.5686478417521,897.7102400063416,900.8518321743418,903.9934243457169,907.1350165204321,910.2766086984528,913.4182008797447,\
916.5597930642743,919.7013852520085,922.8429774429144,925.9845696369598,929.1261618341128,932.2677540343421,935.4093462376167,938.5509384439058,941.6925306531795,944.8341228654078,\
947.9757150805616,951.1173072986116,954.2588995195295,957.4004917432868,960.5420839698558,963.6836761992089,966.825268431319,969.9668606661594,973.1084529037036,976.2500451439255,\
979.3916373867992,982.5332296322995,985.674821880401,988.8164141310792,991.9580063843094,995.0995986400676,998.2411908983298,1001.3827831590725,1004.5243754222724,1007.6659676879066,\
1010.8075599559522,1013.949152226387,1017.0907444991888,1020.2323367743356,1023.373929051806,1026.5155213315786,1029.6571136136324,1032.7987058979463,1035.9402981845,1039.081890473273,\
1042.2234827642455,1045.3650750573975,1048.5066673527094,1051.648259650162,1054.7898519497357,1057.9314442514121,1061.0730365551724,1064.2146288609981,1067.3562211688711,1070.497813478773,\
1073.6394057906864,1076.7809981045934,1079.922590420477,1083.0641827383195,1086.205775058104,1089.3473673798142,1092.488959703433,1095.630552028944,1098.7721443563312,1101.9137366855782,\
1105.0553290166697,1108.1969213495895,1111.3385136843222,1114.4801060208526,1117.6216983591655,1120.763290699246,1123.9048830410788,1127.04647538465,1130.1880677299446,1133.3296600769484,\
1136.4712524256472,1139.612844776027,1142.754437128074,1145.8960294817741,1149.0376218371143,1152.1792141940812,1155.3208065526608,1158.4623989128409,1161.6039912746078,1164.7455836379488,\
1167.8871760028514,1171.0287683693032,1174.1703607372913,1177.3119531068037,1180.4535454778281,1183.5951378503526,1186.7367302243651,1189.878322599854,1193.0199149768075,1196.1615073552143,\
1199.3030997350625,1202.4446921163412,1205.5862844990393,1208.7278768831452,1211.8694692686486,1215.011061655538,1218.1526540438033,1221.2942464334335,1224.4358388244184,1227.5774312167473,\
1230.71902361041,1233.8606160053962,1237.002208401696,1240.1438007992995,1243.2853931981965,1246.4269855983775,1249.5685779998325,1252.7101704025524,1255.8517628065274,1258.9933552117482,\
1262.1349476182052,1265.2765400258895,1268.418132434792,1271.5597248449035,1274.7013172562151,1277.842909668718,1280.9845020824034,1284.1260944972628,1287.2676869132874,1290.4092793304685,\
1293.5508717487983,1296.6924641682679,1299.834056588869,1302.9756490105938,1306.117241433434,1309.2588338573814,1312.4004262824285,1315.542018708567,1318.6836111357893,1321.8252035640876,\
1324.9667959934543,1328.1083884238817,1331.2499808553623,1334.3915732878888,1337.5331657214538,1340.67475815605,1343.81635059167,1346.957943028307,1350.0995354659535,1353.2411279046028,\
1356.3827203442474,1359.5243127848812,1362.6659052264968,1365.8074976690873,1368.9490901126464,1372.0906825571674,1375.2322750026435,1378.3738674490683,1381.5154598964352,1384.657052344738,\
1387.7986447939702,1390.9402372441255,1394.0818296951977,1397.2234221471806,1400.365014600068,1403.506607053854,1406.6481995085326,1409.7897919640975,1412.931384420543,1416.0729768778633,\
1419.2145693360526,1422.356161795105,1425.4977542550148,1428.6393467157764,1431.7809391773842,1434.9225316398329,1438.0641241031165,1441.2057165672297,1444.3473090321675,1447.4889014979237,\
1450.6304939644938,1453.7720864318721,1456.9136789000536,1460.0552713690329,1463.1968638388048,1466.3384563093646,1469.4800487807067,1472.6216412528267,1475.7632337257191,1478.9048261993794,\
1482.0464186738025,1485.1880111489834,1488.3296036249176,1491.4711961016,1494.6127885790263,1497.7543810571917,1500.8959735360913,1504.0375660157208,1507.1791584960754,1510.3207509771507,\
1513.4623434589423,1516.6039359414456,1519.7455284246562,1522.8871209085696,1526.028713393182,1529.1703058784883,1532.311898364485,1535.4534908511673,1538.5950833385311,1541.7366758265725,\
1544.8782683152872,1548.0198608046712,1551.16145329472,1554.3030457854304,1557.4446382767976,1560.5862307688178,1563.7278232614874,1566.8694157548023,1570.0110082487586,1573.1526007433524,\
1576.29419323858,1579.4357857344376,1582.5773782309216,1585.718970728028,1588.860563225753,1592.0021557240932,1595.1437482230451,1598.2853407226048,1601.4269332227689,1604.5685257235339,\
1607.710118224896,1610.851710726852,1613.993303229398,1617.134895732531,1620.2764882362476,1623.4180807405442,1626.5596732454173,1629.701265750864,1632.8428582568808,1635.984450763464,\
1639.1260432706113,1642.2676357783184,1645.4092282865827,1648.550820795401,1651.6924133047698,1654.8340058146864,1657.9755983251473,1661.1171908361498,1664.2587833476905,1667.4003758597664,\
1670.5419683723746,1673.6835608855122,1676.8251533991759,1679.966745913363,1683.1083384280705,1686.2499309432956,1689.3915234590352,1692.5331159752864,1695.6747084920466,1698.816301009313,\
1701.9578935270827,1705.0994860453527,1708.2410785641205,1711.3826710833832,1714.5242636031383,1717.6658561233828,1720.8074486441144,1723.94904116533,1727.0906336870273,1730.2322262092036,\
1733.3738187318563,1736.5154112549826,1739.6570037785802,1742.7985963026465,1745.9401888271789,1749.0817813521746,1752.2233738776317,1755.3649664035474,1758.5065589299195,1761.648151456745,\
1764.789743984022,1767.9313365117478,1771.0729290399202,1774.2145215685368,1777.356114097595,1780.4977066270928,1783.6392991570278,1786.7808916873976,1789.9224842181998,1793.0640767494324,\
1796.205669281093,1799.3472618131793,1802.4888543456893,1805.6304468786204,1808.7720394119708,1811.9136319457382,1815.0552244799203,1818.1968170145149,1821.3384095495203,1824.4800020849339,\
1827.6215946207537,1830.763187156978,1833.904779693604,1837.0463722306301,1840.1879647680544,1843.3295573058747,1846.4711498440888,1849.612742382695,1852.754334921691,1855.895927461075,\
1859.0375200008448,1862.1791125409989,1865.320705081535,1868.4622976224514,1871.6038901637457,1874.7454827054166,1877.887075247462,1881.02866778988,1884.1702603326687,1887.3118528758262,\
1890.4534454193508,1893.5950379632407,1896.7366305074938,1899.8782230521088,1903.0198155970832,1906.161408142416,1909.3030006881047,1912.4445932341482,1915.5861857805442,1918.7277783272914,\
1921.8693708743879,1925.0109634218318,1928.1525559696217,1931.294148517756,1934.4357410662326,1937.57733361505,1940.7189261642068,1943.8605187137011,1947.0021112635313,1950.143703813696,\
1953.285296364193,1956.4268889150214,1959.5684814661793,1962.710074017665,1965.851666569477,1968.9932591216138,1972.1348516740738,1975.2764442268556,1978.4180367799574,1981.559629333378,\
1984.7012218871157,1987.842814441169,1990.9844069955363,1994.1259995502164,1997.2675921052075,2000.4091846605083,2003.5507772161175,2006.6923697720333,2009.8339623282545,2012.9755548847797,\
2016.1171474416074,2019.2587399987362,2022.4003325561646,2025.5419251138912,2028.683517671915,2031.825110230234,2034.9667027888474,2038.1082953477535,2041.249887906951,2044.3914804664387,\
2047.533073026215,2050.674665586279,2053.816258146629,2056.957850707264,2060.0994432681823,2063.241035829383,2066.3826283908643,2069.524220952625,2072.6658135146645,2075.8074060769814,\
2078.948998639574,2082.0905912024405,2085.232183765581,2088.3737763289937,2091.515368892677,2094.65696145663,2097.798554020852,2100.9401465853407,2104.0817391500955,2107.2233317151154,\
2110.364924280399,2113.506516845945,2116.648109411752,2119.7897019778197,2122.9312945441466,2126.072887110731,2129.2144796775724,2132.356072244669,2135.4976648120205,2138.639257379625,\
2141.7808499474822,2144.9224425155903,2148.0640350839485,2151.2056276525554,2154.34722022141,2157.488812790512,2160.6304053598597,2163.7719979294516,2166.9135904992872,2170.0551830693653,\
2173.196775639685,2176.338368210245,2179.479960781044,2182.6215533520817,2185.7631459233567,2188.9047384948676,2192.046331066614,2195.187923638595,2198.329516210809,2201.471108783255,\
2204.6127013559326,2207.7542939288405,2210.8958865019777,2214.037479075343,2217.179071648936,2220.320664222755,2223.4622567967995,2226.6038493710685,2229.745441945561,2232.8870345202763,\
2236.028627095213,2239.1702196703704,2242.3118122457477,2245.4534048213436,2248.5949973971574,2251.7365899731885,2254.8781825494357,2258.019775125898,2261.161367702575,2264.302960279465,\
2267.4445528565675,2270.586145433882,2273.727738011407,2276.8693305891416,2280.010923167086,2283.152515745238,2286.2941083235974,2289.4357009021633,2292.577293480935,2295.7188860599113,\
2298.8604786390915,2302.002071218475,2305.1436637980605,2308.285256377848,2311.4268489578353,2314.568441538023,2317.7100341184096,2320.851626698994,2323.9932192797764,2327.134811860755,\
2330.276404441929,2333.4179970232985,2336.559589604862,2339.7011821866195,2342.842774768569,2345.9843673507107,2349.1259599330433,2352.2675525155664,2355.4091450982787,2358.55073768118,\
2361.69233026427,2364.8339228475465,2367.9755154310096,2371.117108014659,2374.2587005984933,2377.4002931825116,2380.541885766714,2383.683478351099,2386.8250709356666,2389.9666635204153,\
2393.108256105345,2396.249848690454,2399.391441275743,2402.5330338612107,2405.674626446856,2408.816219032679,2411.957811618678,2415.099404204853,2418.240996791203,2421.382589377728,\
2424.524181964426,2427.665774551298,2430.8073671383418,2433.948959725558,2437.0905523129445,2440.232144900502,2443.3737374882294,2446.515330076126,2449.6569226641905,2452.7985152524234,\
2455.940107840823,2459.08170042939,2462.223293018122,2465.36488560702,2468.5064781960823,2471.6480707853084,2474.7896633746986,2477.9312559642512,2481.0728485539657,2484.214441143842,\
2487.3560337338795,2490.497626324077,2493.6392189144344,2496.7808115049506,2499.922404095626,2503.0639966864587,2506.2055892774492,2509.347181868596,2512.488774459899,2515.630367051358,\
2518.7719596429715,2521.9135522347397,2525.0551448266615,2528.196737418737,2531.338330010965,2534.4799226033447,2537.6215151958763,2540.763107788559,2543.904700381392,2547.046292974375,\
2550.1878855675072,2553.3294781607883,2556.471070754217,2559.6126633477943,2562.7542559415183,2565.8958485353887,2569.0374411294056,2572.179033723568,2575.320626317875,2578.4622189123265,\
2581.6038115069223,2584.7454041016613,2587.886996696543,2591.028589291567,2594.170181886733,2597.3117744820406,2600.453367077489,2603.5949596730775,2606.736552268806,2609.8781448646737,\
2613.01973746068,2616.1613300568247,2619.3029226531075,2622.4445152495273,2625.586107846084,2628.727700442777,2631.8692930396055,2635.0108856365696,2638.1524782336687,2641.2940708309025,\
2644.4356634282694,2647.57725602577,2650.718848623404,2653.8604412211703,2657.0020338190684,2660.143626417098,2663.285219015259,2666.4268116135504,2669.568404211972,2672.709996810523,\
2675.8515894092034,2678.993182008013,2682.1347746069505,2685.276367206016,2688.4179598052087,2691.559552404529,2694.701145003975,2697.842737603548,2700.984330203246,2704.1259228030694,\
2707.2675154030176,2710.40910800309,2713.5507006032867,2716.6922932036064,2719.8338858040493,2722.975478404615,2726.117071005303,2729.258663606113,2732.4002562070436,2735.5418488080954,\
2738.683441409268,2741.8250340105606,2744.966626611973,2748.1082192135045,2751.249811815155,2754.391404416924,2757.5329970188113,2760.674589620816,2763.816182222938,2766.957774825177,\
2770.0993674275323,2773.240960030004,2776.382552632591,2779.5241452352934,2782.6657378381105,2785.8073304410423,2788.9489230440886,2792.090515647248,2795.232108250521,2798.373700853907,\
2801.5152934574057,2804.656886061017,2807.7984786647394,2810.9400712685733,2814.0816638725187,2817.2232564765745,2820.364849080741,2823.5064416850173,2826.6480342894033,2829.7896268938985,\
2832.9312194985023,2836.0728121032153,2839.214404708036,2842.3559973129645,2845.4975899180004,2848.6391825231435,2851.7807751283935,2854.9223677337495,2858.063960339212,2861.20555294478,\
2864.347145550453,2867.488738156231,2870.630330762114,2873.7719233681014,2876.9135159741927,2880.0551085803877,2883.196701186686,2886.338293793087,2889.4798863995907,2892.6214790061963,\
2895.7630716129042,2898.9046642197136,2902.046256826625,2905.1878494336365,2908.3294420407487,2911.4710346479615,2914.612627255274,2917.7542198626866,2920.895812470198,2924.0374050778087,\
2927.178997685518,2930.320590293326,2933.4621829012317,2936.603775509235,2939.745368117336,2942.886960725534,2946.028553333829,2949.1701459422206,2952.311738550708,2955.4533311592913,\
2958.5949237679706,2961.736516376745,2964.8781089856143,2968.0197015945782,2971.1612942036363,2974.302886812789,2977.4444794220353,2980.586072031375,2983.7276646408077,2986.8692572503332,\
2990.0108498599516,2993.152442469662,2996.294035079465,2999.4356276893595,3002.5772202993453,3005.718812909422,3008.86040551959,3012.0019981298483,3015.143590740197,3018.285183350636,\
3021.4267759611644,3024.5683685717822,3027.709961182489,3030.8515537932853,3033.99314640417,3037.1347390151427,3040.276331626204,3043.417924237353,3046.5595168485893,3049.701109459913,\
3052.8427020713234,3055.9842946828207,3059.1258872944045,3062.2674799060746,3065.4090725178303,3068.550665129672,3071.692257741599,3074.833850353611,3077.975442965708,3081.1170355778895,\
3084.2586281901554,3087.4002208025054,3090.5418134149395,3093.683406027457,3096.824998640058,3099.9665912527416,3103.108183865508,3106.2497764783575,3109.391369091289,3112.532961704303,\
3115.6745543173984,3118.8161469305755,3121.9577395438337,3125.099332157173,3128.2409247705937,3131.3825173840946,3134.524109997676,3137.6657026113376,3140.8072952250786])
