#define PROBE_COUNT 92
#define PREFILL_LENGTH 16
#define DECODE_STEPS 8
#define MAX_ATOL 1.000000000e-03f
#define MEAN_ATOL 1.000000000e-04f
#define PROBE_PROGRESS 1

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
  {0,3,0,49152,{1,16,3072},{180224}},
  {0,3,0,49152,{1,16,3072},{229376}},
  {0,2,0,16384,{16,1024},{278528}},
  {0,2,1,16384,{16,1024},{294912}},
  {0,1,0,16,{16},{311296}},
  {0,2,0,32768,{16,2048},{311312}},
  {0,2,0,16384,{16,1024},{344080}},
  {0,2,1,16384,{16,1024},{360464}},
  {0,1,0,16,{16},{376848}},
  {0,2,0,16384,{16,1024},{376864}},
  {0,2,0,16384,{16,1024},{393248}},
  {0,2,1,16384,{16,1024},{409632}},
  {0,1,0,16,{16},{426016}},
  {0,2,0,16384,{16,1024},{426032}},
  {0,2,0,32768,{16,2048},{442416}},
  {0,2,1,32768,{16,2048},{475184}},
  {0,1,0,16,{16},{507952}},
  {0,2,0,16384,{16,1024},{507968}},
  {0,2,0,16384,{16,1024},{524352}},
  {0,2,1,16384,{16,1024},{540736}},
  {0,1,0,16,{16},{557120}},
  {0,2,0,49152,{16,3072},{557136}},
  {0,2,0,16384,{16,1024},{606288}},
  {0,2,1,16384,{16,1024},{622672}},
  {0,1,0,16,{16},{639056}},
  {0,2,0,49152,{16,3072},{639072}},
  {0,2,0,49152,{16,3072},{688224}},
  {0,2,1,49152,{16,3072},{737376}},
  {0,1,0,16,{16},{786528}},
  {0,2,0,16384,{16,1024},{786544}},
  {0,2,0,1024,{1,1024},{802928}},
  {0,2,1,1024,{1,1024},{803952}},
  {0,1,0,1,{1},{804976}},
  {0,2,0,151936,{1,151936},{804977}},
  {0,4,0,32768,{1,16,16,128},{956913}},
  {0,4,0,131072,{1,16,16,512},{989681}},
  {0,4,0,32768,{1,16,16,128},{1120753}},
  {1,3,0,1024,{1,1,1024},{1153521,1154545,1155569,1156593,1157617,1158641,1159665,1160689}},
  {1,3,0,1024,{1,1,1024},{1161713,1162737,1163761,1164785,1165809,1166833,1167857,1168881}},
  {1,4,0,2048,{1,1,16,128},{1169905,1171953,1174001,1176049,1178097,1180145,1182193,1184241}},
  {1,4,0,2048,{1,1,16,128},{1186289,1188337,1190385,1192433,1194481,1196529,1198577,1200625}},
  {1,4,0,1024,{1,1,8,128},{1202673,1203697,1204721,1205745,1206769,1207793,1208817,1209841}},
  {1,4,0,1024,{1,1,8,128},{1210865,1211889,1212913,1213937,1214961,1215985,1217009,1218033}},
  {1,3,0,1024,{1,1,1024},{1219057,1220081,1221105,1222129,1223153,1224177,1225201,1226225}},
  {1,3,0,1024,{1,1,1024},{1227249,1228273,1229297,1230321,1231345,1232369,1233393,1234417}},
  {1,3,0,1024,{1,1,1024},{1235441,1236465,1237489,1238513,1239537,1240561,1241585,1242609}},
  {1,3,0,3072,{1,1,3072},{1243633,1246705,1249777,1252849,1255921,1258993,1262065,1265137}},
  {1,3,0,3072,{1,1,3072},{1268209,1271281,1274353,1277425,1280497,1283569,1286641,1289713}},
  {1,2,0,1024,{1,1024},{1292785,1293809,1294833,1295857,1296881,1297905,1298929,1299953}},
  {1,2,1,1024,{1,1024},{1300977,1302001,1303025,1304049,1305073,1306097,1307121,1308145}},
  {1,1,0,1,{1},{1309169,1309170,1309171,1309172,1309173,1309174,1309175,1309176}},
  {1,2,0,2048,{1,2048},{1309177,1311225,1313273,1315321,1317369,1319417,1321465,1323513}},
  {1,2,0,1024,{1,1024},{1325561,1326585,1327609,1328633,1329657,1330681,1331705,1332729}},
  {1,2,1,1024,{1,1024},{1333753,1334777,1335801,1336825,1337849,1338873,1339897,1340921}},
  {1,1,0,1,{1},{1341945,1341946,1341947,1341948,1341949,1341950,1341951,1341952}},
  {1,2,0,1024,{1,1024},{1341953,1342977,1344001,1345025,1346049,1347073,1348097,1349121}},
  {1,2,0,1024,{1,1024},{1350145,1351169,1352193,1353217,1354241,1355265,1356289,1357313}},
  {1,2,1,1024,{1,1024},{1358337,1359361,1360385,1361409,1362433,1363457,1364481,1365505}},
  {1,1,0,1,{1},{1366529,1366530,1366531,1366532,1366533,1366534,1366535,1366536}},
  {1,2,0,1024,{1,1024},{1366537,1367561,1368585,1369609,1370633,1371657,1372681,1373705}},
  {1,2,0,2048,{1,2048},{1374729,1376777,1378825,1380873,1382921,1384969,1387017,1389065}},
  {1,2,1,2048,{1,2048},{1391113,1393161,1395209,1397257,1399305,1401353,1403401,1405449}},
  {1,1,0,1,{1},{1407497,1407498,1407499,1407500,1407501,1407502,1407503,1407504}},
  {1,2,0,1024,{1,1024},{1407505,1408529,1409553,1410577,1411601,1412625,1413649,1414673}},
  {1,2,0,1024,{1,1024},{1415697,1416721,1417745,1418769,1419793,1420817,1421841,1422865}},
  {1,2,1,1024,{1,1024},{1423889,1424913,1425937,1426961,1427985,1429009,1430033,1431057}},
  {1,1,0,1,{1},{1432081,1432082,1432083,1432084,1432085,1432086,1432087,1432088}},
  {1,2,0,3072,{1,3072},{1432089,1435161,1438233,1441305,1444377,1447449,1450521,1453593}},
  {1,2,0,1024,{1,1024},{1456665,1457689,1458713,1459737,1460761,1461785,1462809,1463833}},
  {1,2,1,1024,{1,1024},{1464857,1465881,1466905,1467929,1468953,1469977,1471001,1472025}},
  {1,1,0,1,{1},{1473049,1473050,1473051,1473052,1473053,1473054,1473055,1473056}},
  {1,2,0,3072,{1,3072},{1473057,1476129,1479201,1482273,1485345,1488417,1491489,1494561}},
  {1,2,0,3072,{1,3072},{1497633,1500705,1503777,1506849,1509921,1512993,1516065,1519137}},
  {1,2,1,3072,{1,3072},{1522209,1525281,1528353,1531425,1534497,1537569,1540641,1543713}},
  {1,1,0,1,{1},{1546785,1546786,1546787,1546788,1546789,1546790,1546791,1546792}},
  {1,2,0,1024,{1,1024},{1546793,1547817,1548841,1549865,1550889,1551913,1552937,1553961}},
  {1,2,0,1024,{1,1024},{1554985,1556009,1557033,1558057,1559081,1560105,1561129,1562153}},
  {1,2,1,1024,{1,1024},{1563177,1564201,1565225,1566249,1567273,1568297,1569321,1570345}},
  {1,1,0,1,{1},{1571369,1571370,1571371,1571372,1571373,1571374,1571375,1571376}},
  {1,2,0,151936,{1,151936},{1571377,1723313,1875249,2027185,2179121,2331057,2482993,2634929}},
  {1,4,0,2048,{1,16,1,128},{2786865,2788913,2790961,2793009,2795057,2797105,2799153,2801201}},
  {1,4,0,8192,{1,16,1,512},{2803249,2811441,2819633,2827825,2836017,2844209,2852401,2860593}},
  {1,4,0,2048,{1,16,1,128},{2868785,2870833,2872881,2874929,2876977,2879025,2881073,2883121}},
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
extern void __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_pv_attention_pv_position_16x16x128x512(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2, MemRef4 *a3);
void __wrap__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_pv_attention_pv_position_16x16x128x512(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2, MemRef4 *a3) {
  if(current_kind != 0) failed=1;
  __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_pv_attention_pv_position_16x16x128x512(a0, a1, a2, a3);
  ame_fence();
  check(45, a3);
}
extern void __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_pv_attention_pv_position_16x1x128x512(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2, MemRef4 *a3);
void __wrap__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_pv_attention_pv_position_16x1x128x512(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2, MemRef4 *a3) {
  if(current_kind != 1) failed=1;
  __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_pv_attention_pv_position_16x1x128x512(a0, a1, a2, a3);
  ame_fence();
  check(91, a3);
}
extern void __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_qk_attention_qk_position_native_16x16x512x128(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2, MemRef4 *a3);
void __wrap__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_qk_attention_qk_position_native_16x16x512x128(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2, MemRef4 *a3) {
  if(current_kind != 0) failed=1;
  check(43, a0);
  __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_qk_attention_qk_position_native_16x16x512x128(a0, a1, a2, a3);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_qk_attention_qk_position_native_16x1x512x128(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2, MemRef4 *a3);
void __wrap__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_qk_attention_qk_position_native_16x1x512x128(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2, MemRef4 *a3) {
  if(current_kind != 1) failed=1;
  check(89, a0);
  __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_qk_attention_qk_position_native_16x1x512x128(a0, a1, a2, a3);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_softmax_softmax_16x16x512(MemRef4 *a0, MemRef1 *a1, MemRef1 *a2, MemRef4 *a3);
void __wrap__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_softmax_softmax_16x16x512(MemRef4 *a0, MemRef1 *a1, MemRef1 *a2, MemRef4 *a3) {
  if(current_kind != 0) failed=1;
  __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_softmax_softmax_16x16x512(a0, a1, a2, a3);
  ame_fence();
  check(44, a3);
}
extern void __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_softmax_softmax_16x1x512(MemRef4 *a0, MemRef1 *a1, MemRef1 *a2, MemRef4 *a3);
void __wrap__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_softmax_softmax_16x1x512(MemRef4 *a0, MemRef1 *a1, MemRef1 *a2, MemRef4 *a3) {
  if(current_kind != 1) failed=1;
  __real__mlir_ciface_qwen_graph_attn__scaled_dot_product_flash_attention_for_cpu_softmax_softmax_16x1x512(a0, a1, a2, a3);
  ame_fence();
  check(90, a3);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_128x128__forward_prefill_mul_7(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_128x128__forward_prefill_mul_7(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(4, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_128x128__forward_prefill_mul_7(a0, a1, a2);
  ame_fence();
  check(5, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_13(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_13(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(6, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_13(a0, a1, a2);
  ame_fence();
  check(7, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_17(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_17(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(8, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_17(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_3(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_3(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(0, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_16x1024__forward_prefill_mul_3(a0, a1, a2);
  ame_fence();
  check(1, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_16x128__forward_decode_mul_5(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_16x128__forward_decode_mul_5(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(48, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_16x128__forward_decode_mul_5(a0, a1, a2);
  ame_fence();
  check(49, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_13(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_13(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(52, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_13(a0, a1, a2);
  ame_fence();
  check(53, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_17(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_17(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(54, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_17(a0, a1, a2);
  ame_fence();
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_3(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_3(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(46, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_1x1024__forward_decode_mul_3(a0, a1, a2);
  ame_fence();
  check(47, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_256x128__forward_prefill_mul_5(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_256x128__forward_prefill_mul_5(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(2, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_256x128__forward_prefill_mul_5(a0, a1, a2);
  ame_fence();
  check(3, a0);
}
extern void __real__mlir_ciface_qwen_graph_rmsnorm_8x128__forward_decode_mul_7(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_rmsnorm_8x128__forward_decode_mul_7(MemRef4 *a0, MemRef4 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(50, a1);
  __real__mlir_ciface_qwen_graph_rmsnorm_8x128__forward_decode_mul_7(a0, a1, a2);
  ame_fence();
  check(51, a0);
}
extern void __real__mlir_ciface_qwen_graph_silu_16x3072__forward_prefill_mul_14(MemRef3 *a0, MemRef3 *a1);
void __wrap__mlir_ciface_qwen_graph_silu_16x3072__forward_prefill_mul_14(MemRef3 *a0, MemRef3 *a1) {
  if(current_kind != 0) failed=1;
  check(9, a1);
  __real__mlir_ciface_qwen_graph_silu_16x3072__forward_prefill_mul_14(a0, a1);
  ame_fence();
  check(10, a0);
}
extern void __real__mlir_ciface_qwen_graph_silu_1x3072__forward_decode_mul_14(MemRef3 *a0, MemRef3 *a1);
void __wrap__mlir_ciface_qwen_graph_silu_1x3072__forward_decode_mul_14(MemRef3 *a0, MemRef3 *a1) {
  if(current_kind != 1) failed=1;
  check(55, a1);
  __real__mlir_ciface_qwen_graph_silu_1x3072__forward_decode_mul_14(a0, a1);
  ame_fence();
  check(56, a0);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_1_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_1_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_1_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(18, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_1_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_1_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_1_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(64, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_1_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_1_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(15, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_1_quantize_16x1024(a0, a1, a2);
  ame_fence();
  check(16, a1);
  check(17, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_1_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_1_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(61, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_1_quantize_1x1024(a0, a1, a2);
  ame_fence();
  check(62, a1);
  check(63, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_2_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_2_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_2_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(22, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_2_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_2_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_2_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(68, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_2_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_2_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(19, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_2_quantize_16x1024(a0, a1, a2);
  ame_fence();
  check(20, a1);
  check(21, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_2_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_2_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(65, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_2_quantize_1x1024(a0, a1, a2);
  ame_fence();
  check(66, a1);
  check(67, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_3_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_3_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_3_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(26, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_3_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_3_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_3_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(72, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_3_quantize_16x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_3_quantize_16x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(23, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_3_quantize_16x2048(a0, a1, a2);
  ame_fence();
  check(24, a1);
  check(25, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_3_quantize_1x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_3_quantize_1x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(69, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_3_quantize_1x2048(a0, a1, a2);
  ame_fence();
  check(70, a1);
  check(71, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_4_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_4_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_4_dequantize_16x3072(a0, a1, a2, a3);
  ame_fence();
  check(30, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_4_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_4_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_4_dequantize_1x3072(a0, a1, a2, a3);
  ame_fence();
  check(76, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_4_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_4_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(27, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_4_quantize_16x1024(a0, a1, a2);
  ame_fence();
  check(28, a1);
  check(29, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_4_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_4_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(73, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_4_quantize_1x1024(a0, a1, a2);
  ame_fence();
  check(74, a1);
  check(75, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_5_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_5_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_5_dequantize_16x3072(a0, a1, a2, a3);
  ame_fence();
  check(34, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_5_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_5_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_5_dequantize_1x3072(a0, a1, a2, a3);
  ame_fence();
  check(80, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_5_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_5_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(31, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_5_quantize_16x1024(a0, a1, a2);
  ame_fence();
  check(32, a1);
  check(33, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_5_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_5_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(77, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_5_quantize_1x1024(a0, a1, a2);
  ame_fence();
  check(78, a1);
  check(79, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_6_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_6_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_6_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  check(38, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_6_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_6_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_6_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  check(84, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_6_quantize_16x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_6_quantize_16x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(35, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_6_quantize_16x3072(a0, a1, a2);
  ame_fence();
  check(36, a1);
  check(37, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_6_quantize_1x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_6_quantize_1x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(81, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_6_quantize_1x3072(a0, a1, a2);
  ame_fence();
  check(82, a1);
  check(83, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_7_dequantize_1x151936(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_7_dequantize_1x151936(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  __real__mlir_ciface_qwen_graph_w8a8_mm_7_dequantize_1x151936(a0, a1, a2, a3);
  ame_fence();
  check(42, a3);
  check(88, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_7_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_7_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  check(39, a0);
  check(85, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_7_quantize_1x1024(a0, a1, a2);
  ame_fence();
  check(40, a1);
  check(41, a2);
  check(86, a1);
  check(87, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_dequantize_16x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_dequantize_16x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 0) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_dequantize_16x2048(a0, a1, a2, a3);
  ame_fence();
  check(14, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_dequantize_1x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_dequantize_1x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  if(current_kind != 1) failed=1;
  __real__mlir_ciface_qwen_graph_w8a8_mm_dequantize_1x2048(a0, a1, a2, a3);
  ame_fence();
  check(60, a3);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 0) failed=1;
  check(11, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_quantize_16x1024(a0, a1, a2);
  ame_fence();
  check(12, a1);
  check(13, a2);
}
extern void __real__mlir_ciface_qwen_graph_w8a8_mm_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_qwen_graph_w8a8_mm_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  if(current_kind != 1) failed=1;
  check(57, a0);
  __real__mlir_ciface_qwen_graph_w8a8_mm_quantize_1x1024(a0, a1, a2);
  ame_fence();
  check(58, a1);
  check(59, a2);
}
