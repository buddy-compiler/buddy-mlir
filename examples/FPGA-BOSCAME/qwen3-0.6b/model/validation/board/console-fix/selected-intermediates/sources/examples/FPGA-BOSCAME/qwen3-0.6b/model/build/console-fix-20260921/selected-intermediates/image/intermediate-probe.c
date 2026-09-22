#define PROBE_COUNT 134
#define PREFILL_LENGTH 16
#define DECODE_STEPS 8
#define MAX_ATOL 1.000000000e-03f
#define MEAN_ATOL 1.000000000e-04f
#define PROBE_PROGRESS 0

#include "support.h"
#include "nr_runtime.h"
extern const float intermediate_reference_raw[];
typedef struct { unsigned kind, rank, dtype, count; int64_t shape[4]; unsigned offset[9]; } Probe;
static const Probe probes[PROBE_COUNT] = {
  {0,3,0,16384,{1,16,1024},{0}},
  {0,3,0,16384,{1,16,1024},{16384}},
  {0,4,0,32768,{1,16,16,128},{32768}},
  {0,4,0,32768,{1,16,16,128},{65536}},
  {0,4,0,16384,{1,16,8,128},{98304}},
  {0,4,0,16384,{1,16,8,128},{114688}},
  {0,3,0,16384,{1,16,1024},{131072}},
  {0,3,0,16384,{1,16,1024},{147456}},
  {0,3,0,16384,{1,16,1024},{163840}},
  {0,3,0,16384,{1,16,1024},{180224}},
  {0,4,0,32768,{1,16,16,128},{196608}},
  {0,4,0,32768,{1,16,16,128},{229376}},
  {0,4,0,16384,{1,16,8,128},{262144}},
  {0,4,0,16384,{1,16,8,128},{278528}},
  {0,3,0,16384,{1,16,1024},{294912}},
  {0,3,0,16384,{1,16,1024},{311296}},
  {0,3,0,16384,{1,16,1024},{327680}},
  {0,2,0,16384,{16,1024},{344064}},
  {0,2,1,16384,{16,1024},{360448}},
  {0,1,0,16,{16},{376832}},
  {0,2,0,32768,{16,2048},{376848}},
  {0,2,1,16384,{16,1024},{409616}},
  {0,1,0,16,{16},{426000}},
  {0,2,0,16384,{16,1024},{426016}},
  {0,2,1,16384,{16,1024},{442400}},
  {0,1,0,16,{16},{458784}},
  {0,2,0,16384,{16,1024},{458800}},
  {0,2,0,32768,{16,2048},{475184}},
  {0,2,1,32768,{16,2048},{507952}},
  {0,1,0,16,{16},{540720}},
  {0,2,0,16384,{16,1024},{540736}},
  {0,2,0,16384,{16,1024},{557120}},
  {0,2,1,16384,{16,1024},{573504}},
  {0,1,0,16,{16},{589888}},
  {0,2,0,49152,{16,3072},{589904}},
  {0,2,1,16384,{16,1024},{639056}},
  {0,1,0,16,{16},{655440}},
  {0,2,0,49152,{16,3072},{655456}},
  {0,2,0,49152,{16,3072},{704608}},
  {0,2,1,49152,{16,3072},{753760}},
  {0,1,0,16,{16},{802912}},
  {0,2,0,16384,{16,1024},{802928}},
  {0,2,0,16384,{16,1024},{819312}},
  {0,2,1,16384,{16,1024},{835696}},
  {0,1,0,16,{16},{852080}},
  {0,2,0,32768,{16,2048},{852096}},
  {0,2,1,16384,{16,1024},{884864}},
  {0,1,0,16,{16},{901248}},
  {0,2,0,16384,{16,1024},{901264}},
  {0,2,1,16384,{16,1024},{917648}},
  {0,1,0,16,{16},{934032}},
  {0,2,0,16384,{16,1024},{934048}},
  {0,2,0,32768,{16,2048},{950432}},
  {0,2,1,32768,{16,2048},{983200}},
  {0,1,0,16,{16},{1015968}},
  {0,2,0,16384,{16,1024},{1015984}},
  {0,2,0,16384,{16,1024},{1032368}},
  {0,2,1,16384,{16,1024},{1048752}},
  {0,1,0,16,{16},{1065136}},
  {0,2,0,49152,{16,3072},{1065152}},
  {0,2,1,16384,{16,1024},{1114304}},
  {0,1,0,16,{16},{1130688}},
  {0,2,0,49152,{16,3072},{1130704}},
  {0,2,0,49152,{16,3072},{1179856}},
  {0,2,1,49152,{16,3072},{1229008}},
  {0,1,0,16,{16},{1278160}},
  {0,2,0,16384,{16,1024},{1278176}},
  {1,3,0,1024,{1,1,1024},{1294560,1295584,1296608,1297632,1298656,1299680,1300704,1301728}},
  {1,3,0,1024,{1,1,1024},{1302752,1303776,1304800,1305824,1306848,1307872,1308896,1309920}},
  {1,4,0,2048,{1,1,16,128},{1310944,1312992,1315040,1317088,1319136,1321184,1323232,1325280}},
  {1,4,0,2048,{1,1,16,128},{1327328,1329376,1331424,1333472,1335520,1337568,1339616,1341664}},
  {1,4,0,1024,{1,1,8,128},{1343712,1344736,1345760,1346784,1347808,1348832,1349856,1350880}},
  {1,4,0,1024,{1,1,8,128},{1351904,1352928,1353952,1354976,1356000,1357024,1358048,1359072}},
  {1,3,0,1024,{1,1,1024},{1360096,1361120,1362144,1363168,1364192,1365216,1366240,1367264}},
  {1,3,0,1024,{1,1,1024},{1368288,1369312,1370336,1371360,1372384,1373408,1374432,1375456}},
  {1,3,0,1024,{1,1,1024},{1376480,1377504,1378528,1379552,1380576,1381600,1382624,1383648}},
  {1,3,0,1024,{1,1,1024},{1384672,1385696,1386720,1387744,1388768,1389792,1390816,1391840}},
  {1,4,0,2048,{1,1,16,128},{1392864,1394912,1396960,1399008,1401056,1403104,1405152,1407200}},
  {1,4,0,2048,{1,1,16,128},{1409248,1411296,1413344,1415392,1417440,1419488,1421536,1423584}},
  {1,4,0,1024,{1,1,8,128},{1425632,1426656,1427680,1428704,1429728,1430752,1431776,1432800}},
  {1,4,0,1024,{1,1,8,128},{1433824,1434848,1435872,1436896,1437920,1438944,1439968,1440992}},
  {1,3,0,1024,{1,1,1024},{1442016,1443040,1444064,1445088,1446112,1447136,1448160,1449184}},
  {1,3,0,1024,{1,1,1024},{1450208,1451232,1452256,1453280,1454304,1455328,1456352,1457376}},
  {1,3,0,1024,{1,1,1024},{1458400,1459424,1460448,1461472,1462496,1463520,1464544,1465568}},
  {1,2,0,1024,{1,1024},{1466592,1467616,1468640,1469664,1470688,1471712,1472736,1473760}},
  {1,2,1,1024,{1,1024},{1474784,1475808,1476832,1477856,1478880,1479904,1480928,1481952}},
  {1,1,0,1,{1},{1482976,1482977,1482978,1482979,1482980,1482981,1482982,1482983}},
  {1,2,0,2048,{1,2048},{1482984,1485032,1487080,1489128,1491176,1493224,1495272,1497320}},
  {1,2,1,1024,{1,1024},{1499368,1500392,1501416,1502440,1503464,1504488,1505512,1506536}},
  {1,1,0,1,{1},{1507560,1507561,1507562,1507563,1507564,1507565,1507566,1507567}},
  {1,2,0,1024,{1,1024},{1507568,1508592,1509616,1510640,1511664,1512688,1513712,1514736}},
  {1,2,1,1024,{1,1024},{1515760,1516784,1517808,1518832,1519856,1520880,1521904,1522928}},
  {1,1,0,1,{1},{1523952,1523953,1523954,1523955,1523956,1523957,1523958,1523959}},
  {1,2,0,1024,{1,1024},{1523960,1524984,1526008,1527032,1528056,1529080,1530104,1531128}},
  {1,2,0,2048,{1,2048},{1532152,1534200,1536248,1538296,1540344,1542392,1544440,1546488}},
  {1,2,1,2048,{1,2048},{1548536,1550584,1552632,1554680,1556728,1558776,1560824,1562872}},
  {1,1,0,1,{1},{1564920,1564921,1564922,1564923,1564924,1564925,1564926,1564927}},
  {1,2,0,1024,{1,1024},{1564928,1565952,1566976,1568000,1569024,1570048,1571072,1572096}},
  {1,2,0,1024,{1,1024},{1573120,1574144,1575168,1576192,1577216,1578240,1579264,1580288}},
  {1,2,1,1024,{1,1024},{1581312,1582336,1583360,1584384,1585408,1586432,1587456,1588480}},
  {1,1,0,1,{1},{1589504,1589505,1589506,1589507,1589508,1589509,1589510,1589511}},
  {1,2,0,3072,{1,3072},{1589512,1592584,1595656,1598728,1601800,1604872,1607944,1611016}},
  {1,2,1,1024,{1,1024},{1614088,1615112,1616136,1617160,1618184,1619208,1620232,1621256}},
  {1,1,0,1,{1},{1622280,1622281,1622282,1622283,1622284,1622285,1622286,1622287}},
  {1,2,0,3072,{1,3072},{1622288,1625360,1628432,1631504,1634576,1637648,1640720,1643792}},
  {1,2,0,3072,{1,3072},{1646864,1649936,1653008,1656080,1659152,1662224,1665296,1668368}},
  {1,2,1,3072,{1,3072},{1671440,1674512,1677584,1680656,1683728,1686800,1689872,1692944}},
  {1,1,0,1,{1},{1696016,1696017,1696018,1696019,1696020,1696021,1696022,1696023}},
  {1,2,0,1024,{1,1024},{1696024,1697048,1698072,1699096,1700120,1701144,1702168,1703192}},
  {1,2,0,1024,{1,1024},{1704216,1705240,1706264,1707288,1708312,1709336,1710360,1711384}},
  {1,2,1,1024,{1,1024},{1712408,1713432,1714456,1715480,1716504,1717528,1718552,1719576}},
  {1,1,0,1,{1},{1720600,1720601,1720602,1720603,1720604,1720605,1720606,1720607}},
  {1,2,0,2048,{1,2048},{1720608,1722656,1724704,1726752,1728800,1730848,1732896,1734944}},
  {1,2,1,1024,{1,1024},{1736992,1738016,1739040,1740064,1741088,1742112,1743136,1744160}},
  {1,1,0,1,{1},{1745184,1745185,1745186,1745187,1745188,1745189,1745190,1745191}},
  {1,2,0,1024,{1,1024},{1745192,1746216,1747240,1748264,1749288,1750312,1751336,1752360}},
  {1,2,1,1024,{1,1024},{1753384,1754408,1755432,1756456,1757480,1758504,1759528,1760552}},
  {1,1,0,1,{1},{1761576,1761577,1761578,1761579,1761580,1761581,1761582,1761583}},
  {1,2,0,1024,{1,1024},{1761584,1762608,1763632,1764656,1765680,1766704,1767728,1768752}},
  {1,2,0,2048,{1,2048},{1769776,1771824,1773872,1775920,1777968,1780016,1782064,1784112}},
  {1,2,1,2048,{1,2048},{1786160,1788208,1790256,1792304,1794352,1796400,1798448,1800496}},
  {1,1,0,1,{1},{1802544,1802545,1802546,1802547,1802548,1802549,1802550,1802551}},
  {1,2,0,1024,{1,1024},{1802552,1803576,1804600,1805624,1806648,1807672,1808696,1809720}},
  {1,2,0,1024,{1,1024},{1810744,1811768,1812792,1813816,1814840,1815864,1816888,1817912}},
  {1,2,1,1024,{1,1024},{1818936,1819960,1820984,1822008,1823032,1824056,1825080,1826104}},
  {1,1,0,1,{1},{1827128,1827129,1827130,1827131,1827132,1827133,1827134,1827135}},
  {1,2,0,3072,{1,3072},{1827136,1830208,1833280,1836352,1839424,1842496,1845568,1848640}},
  {1,2,1,1024,{1,1024},{1851712,1852736,1853760,1854784,1855808,1856832,1857856,1858880}},
  {1,1,0,1,{1},{1859904,1859905,1859906,1859907,1859908,1859909,1859910,1859911}},
  {1,2,0,3072,{1,3072},{1859912,1862984,1866056,1869128,1872200,1875272,1878344,1881416}},
  {1,2,0,3072,{1,3072},{1884488,1887560,1890632,1893704,1896776,1899848,1902920,1905992}},
  {1,2,1,3072,{1,3072},{1909064,1912136,1915208,1918280,1921352,1924424,1927496,1930568}},
  {1,1,0,1,{1},{1933640,1933641,1933642,1933643,1933644,1933645,1933646,1933647}},
  {1,2,0,1024,{1,1024},{1933648,1934672,1935696,1936720,1937744,1938768,1939792,1940816}},
};
static unsigned current_kind, current_step, current_position, invocation, failed, active_valid;
static unsigned seen[PROBE_COUNT], checked[PROBE_COUNT];
static float maxima[PROBE_COUNT], means[PROBE_COUNT];
static void bits(float value) { union { float f; uint32_t u; } v={value}; nr_hex32(v.u); }
void qwen_intermediate_begin(unsigned position, unsigned length) {
  failed = 0;
  current_kind = length == PREFILL_LENGTH ? 0 : 1;
  current_step = current_kind ? position-PREFILL_LENGTH : 0;
  current_position = position;
  if ((invocation == 0 && (position != 0 || length != PREFILL_LENGTH)) ||
      (invocation != 0 && (position != PREFILL_LENGTH+invocation-1 || length != 1)) ||
      invocation > DECODE_STEPS || current_step >= 8) failed = 1;
  active_valid = !failed;
  for(unsigned i=0;i<PROBE_COUNT;i++) {seen[i]=checked[i]=0;maxima[i]=means[i]=0;}
}
static void check(unsigned index, const void *descriptor) {
  if(index>=PROBE_COUNT) {failed=1;return;}
  const Probe *p = &probes[index];
#if PROBE_PROGRESS
  nr_puts("[probe] entry=");nr_hex32(index);
  nr_puts(" rank=");nr_hex32(p->rank);
  nr_puts(" descriptor=");nr_hex64((uintptr_t)descriptor);
  nr_puts("\r\n");
#endif
  if(p->rank<1 || p->rank>4) {failed=1;return;}
  if(p->kind != current_kind) return;
  if (++seen[index] != 1 || !active_valid || !descriptor) { failed=1; return; }
  void *aligned; int64_t offset, size[4], stride[4];
  memcpy(&aligned, (const char*)descriptor+8, 8);
  memcpy(&offset, (const char*)descriptor+16, 8);
  memcpy(size, (const char*)descriptor+24, p->rank*8);
  memcpy(stride, (const char*)descriptor+24+p->rank*8, p->rank*8);
  if(!aligned || offset<0 || offset>0x10000000) {failed=1;return;}
  for(unsigned j=0;j<p->rank;j++)
    if(size[j]!=p->shape[j] || stride[j]<0 || stride[j]>0x10000000) {failed=1;return;}
  const float *gold=intermediate_reference_raw+p->offset[current_step];
  double sum=0; float max=0;
  for(unsigned i=0;i<p->count;i++) {
    unsigned flat=i; int64_t at=offset;
    for(unsigned j=p->rank;j>0;j--) {at += (flat%size[j-1])*stride[j-1];flat/=size[j-1];}
    float value=p->dtype ? (float)((const int8_t*)aligned)[at] : ((const float*)aligned)[at];
    float diff=value-gold[i];if(diff<0)diff=-diff;
    if(!(value<=3.402823466e38f && value>=-3.402823466e38f && diff<=3.402823466e38f)) {failed=1;return;}
    if(diff>max)max=diff;sum+=diff;
  }
  checked[index]=1;maxima[index]=max;means[index]=(float)(sum/p->count);
  if(max>MAX_ATOL || means[index]>MEAN_ATOL) failed=1;
}
int qwen_intermediate_end(void) {
  for(unsigned i=0;i<PROBE_COUNT;i++) if(probes[i].kind==current_kind) {
    if(seen[i]!=1 || checked[i]!=1)failed=1;
    nr_puts("[intermediate] position=");nr_hex32(current_position);
    nr_puts(" entry=");nr_hex32(i);nr_puts(" calls=");nr_hex32(seen[i]);
    nr_puts(" checked=");nr_hex32(checked[i]);
    nr_puts(" count=");nr_hex32(probes[i].count);
    nr_puts(" max_abs_bits=");bits(maxima[i]);
    nr_puts(" mean_abs_bits=");bits(means[i]);nr_puts("\r\n");
  }
  nr_puts("[intermediate] complete position=");nr_hex32(current_position);
  nr_puts(failed ? " FAIL\r\n" : " PASS\r\n");
  invocation++;
  return failed ? 1 : 0;
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_128x128__forward_prefill_mul_203(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_128x128__forward_prefill_mul_203(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(4, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_128x128__forward_prefill_mul_203(a0, a1, a2);
  ame_fence();
  check(5, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_128x128__forward_prefill_mul_217(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_128x128__forward_prefill_mul_217(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(12, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_128x128__forward_prefill_mul_217(a0, a1, a2);
  ame_fence();
  check(13, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_199(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_199(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(0, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_199(a0, a1, a2);
  ame_fence();
  check(1, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_209(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_209(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(6, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_209(a0, a1, a2);
  ame_fence();
  check(7, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_213(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_213(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(8, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_213(a0, a1, a2);
  ame_fence();
  check(9, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_223(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_223(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(14, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_223(a0, a1, a2);
  ame_fence();
  check(15, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_227(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_227(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(16, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_227(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_16x128__forward_decode_mul_201(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_16x128__forward_decode_mul_201(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(69, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_16x128__forward_decode_mul_201(a0, a1, a2);
  ame_fence();
  check(70, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_16x128__forward_decode_mul_215(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_16x128__forward_decode_mul_215(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(77, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_16x128__forward_decode_mul_215(a0, a1, a2);
  ame_fence();
  check(78, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_199(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_199(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(67, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_199(a0, a1, a2);
  ame_fence();
  check(68, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_209(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_209(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(73, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_209(a0, a1, a2);
  ame_fence();
  check(74, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_213(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_213(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(75, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_213(a0, a1, a2);
  ame_fence();
  check(76, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_223(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_223(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(81, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_223(a0, a1, a2);
  ame_fence();
  check(82, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_227(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_227(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(83, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_227(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_256x128__forward_prefill_mul_201(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_256x128__forward_prefill_mul_201(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(2, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_256x128__forward_prefill_mul_201(a0, a1, a2);
  ame_fence();
  check(3, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_256x128__forward_prefill_mul_215(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_256x128__forward_prefill_mul_215(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(10, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_256x128__forward_prefill_mul_215(a0, a1, a2);
  ame_fence();
  check(11, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_8x128__forward_decode_mul_203(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_8x128__forward_decode_mul_203(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(71, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_8x128__forward_decode_mul_203(a0, a1, a2);
  ame_fence();
  check(72, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_8x128__forward_decode_mul_217(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_8x128__forward_decode_mul_217(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(79, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_8x128__forward_decode_mul_217(a0, a1, a2);
  ame_fence();
  check(80, a0);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_100_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_100_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(25, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_100_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(26, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_100_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_100_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(92, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_100_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(93, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_100_matmul_16x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_100_matmul_16x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(24, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_100_matmul_16x1024x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_100_matmul_1x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_100_matmul_1x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(91, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_100_matmul_1x1024x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_101_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_101_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(29, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_101_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(30, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_101_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_101_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(96, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_101_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(97, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_101_matmul_16x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_101_matmul_16x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(28, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_101_matmul_16x1024x2048(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_101_matmul_1x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_101_matmul_1x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(95, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_101_matmul_1x1024x2048(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_101_quantize_16x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_101_quantize_16x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(27, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_101_quantize_16x2048(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_101_quantize_1x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_101_quantize_1x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(94, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_101_quantize_1x2048(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_102_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_102_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(33, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_102_dequantize_16x3072(a0, a1, a2, a3);
  ame_fence();
  check(34, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_102_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_102_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(100, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_102_dequantize_1x3072(a0, a1, a2, a3);
  ame_fence();
  check(101, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_102_matmul_16x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_102_matmul_16x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(32, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_102_matmul_16x3072x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_102_matmul_1x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_102_matmul_1x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(99, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_102_matmul_1x3072x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_102_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_102_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(31, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_102_quantize_16x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_102_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_102_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(98, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_102_quantize_1x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_103_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_103_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(36, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_103_dequantize_16x3072(a0, a1, a2, a3);
  ame_fence();
  check(37, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_103_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_103_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(103, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_103_dequantize_1x3072(a0, a1, a2, a3);
  ame_fence();
  check(104, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_103_matmul_16x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_103_matmul_16x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(35, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_103_matmul_16x3072x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_103_matmul_1x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_103_matmul_1x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(102, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_103_matmul_1x3072x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_104_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_104_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(40, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_104_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(41, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_104_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_104_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(107, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_104_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(108, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_104_matmul_16x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_104_matmul_16x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(39, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_104_matmul_16x1024x3072(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_104_matmul_1x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_104_matmul_1x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(106, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_104_matmul_1x1024x3072(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_104_quantize_16x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_104_quantize_16x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(38, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_104_quantize_16x3072(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_104_quantize_1x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_104_quantize_1x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(105, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_104_quantize_1x3072(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_105_dequantize_16x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_105_dequantize_16x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(44, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_105_dequantize_16x2048(a0, a1, a2, a3);
  ame_fence();
  check(45, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_105_dequantize_1x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_105_dequantize_1x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(111, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_105_dequantize_1x2048(a0, a1, a2, a3);
  ame_fence();
  check(112, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_105_matmul_16x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_105_matmul_16x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(43, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_105_matmul_16x2048x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_105_matmul_1x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_105_matmul_1x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(110, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_105_matmul_1x2048x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_105_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_105_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(42, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_105_quantize_16x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_105_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_105_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(109, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_105_quantize_1x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_106_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_106_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(47, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_106_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(48, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_106_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_106_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(114, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_106_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(115, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_106_matmul_16x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_106_matmul_16x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(46, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_106_matmul_16x1024x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_106_matmul_1x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_106_matmul_1x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(113, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_106_matmul_1x1024x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_107_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_107_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(50, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_107_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(51, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_107_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_107_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(117, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_107_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(118, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_107_matmul_16x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_107_matmul_16x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(49, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_107_matmul_16x1024x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_107_matmul_1x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_107_matmul_1x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(116, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_107_matmul_1x1024x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_108_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_108_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(54, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_108_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(55, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_108_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_108_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(121, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_108_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(122, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_108_matmul_16x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_108_matmul_16x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(53, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_108_matmul_16x1024x2048(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_108_matmul_1x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_108_matmul_1x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(120, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_108_matmul_1x1024x2048(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_108_quantize_16x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_108_quantize_16x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(52, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_108_quantize_16x2048(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_108_quantize_1x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_108_quantize_1x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(119, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_108_quantize_1x2048(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_109_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_109_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(58, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_109_dequantize_16x3072(a0, a1, a2, a3);
  ame_fence();
  check(59, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_109_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_109_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(125, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_109_dequantize_1x3072(a0, a1, a2, a3);
  ame_fence();
  check(126, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_109_matmul_16x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_109_matmul_16x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(57, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_109_matmul_16x3072x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_109_matmul_1x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_109_matmul_1x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(124, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_109_matmul_1x3072x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_109_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_109_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(56, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_109_quantize_16x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_109_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_109_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(123, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_109_quantize_1x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_110_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_110_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(61, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_110_dequantize_16x3072(a0, a1, a2, a3);
  ame_fence();
  check(62, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_110_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_110_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(128, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_110_dequantize_1x3072(a0, a1, a2, a3);
  ame_fence();
  check(129, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_110_matmul_16x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_110_matmul_16x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(60, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_110_matmul_16x3072x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_110_matmul_1x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_110_matmul_1x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(127, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_110_matmul_1x3072x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_111_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_111_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(65, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_111_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(66, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_111_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_111_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(132, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_111_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(133, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_111_matmul_16x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_111_matmul_16x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(64, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_111_matmul_16x1024x3072(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_111_matmul_1x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_111_matmul_1x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(131, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_111_matmul_1x1024x3072(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_111_quantize_16x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_111_quantize_16x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(63, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_111_quantize_16x3072(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_111_quantize_1x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_111_quantize_1x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(130, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_111_quantize_1x3072(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_98_dequantize_16x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_98_dequantize_16x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(19, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_98_dequantize_16x2048(a0, a1, a2, a3);
  ame_fence();
  check(20, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_98_dequantize_1x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_98_dequantize_1x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(86, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_98_dequantize_1x2048(a0, a1, a2, a3);
  ame_fence();
  check(87, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_98_matmul_16x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_98_matmul_16x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(18, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_98_matmul_16x2048x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_98_matmul_1x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_98_matmul_1x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(85, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_98_matmul_1x2048x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_98_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_98_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(17, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_98_quantize_16x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_98_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_98_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(84, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_98_quantize_1x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_99_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_99_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  check(22, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_99_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(23, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_99_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_99_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  check(89, a1);
  __real__mlir_ciface_qwen_graph_w8a8_mm_99_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(90, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_99_matmul_16x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_99_matmul_16x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 0) failed=1;
  check(21, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_99_matmul_16x1024x1024(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_99_matmul_1x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_99_matmul_1x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  if(current_kind != 1) failed=1;
  check(88, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_99_matmul_1x1024x1024(a0, a1, a2);
  ame_fence();
}
