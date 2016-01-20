/** @file real_measurementsetup_defines.h */
#ifndef MEASUREMENTSETUP_DEFINES
#define MEASUREMENTSETUP_DEFINES

#define N0Z 13      // 1st detector's number of segments in z
#define N0Y 13      // 1st detector's number of segments in y
#define N1Z 13      // 2nd detector's number of segments in z
#define N1Y 13      // 2nd detector's number of segments in y
#define NA  180     // number of angular positions
#define DA     2.     // angular step
#define POS0X -0.457  // position of 1st detector's center in x [m]
#define POS1X  0.457  // position of 2nd detector's center in x [m]
#define SEGX   0.020  // x edge length of one detector segment [m]
#define SEGY   0.004  // y edge length of one detector segment [m]
#define SEGZ   0.004  // z edge length of one detector segment [m]
#define NCHANNELS NA*N0Z*N0Y*N1Z*N1Y

#endif  // #ifndef MEASUREMENTSETUP_DEFINES

