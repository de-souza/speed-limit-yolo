#ifndef UTILS_H
#define UTILS_H

#include <darknet.h>

image **load_alphabet();
detection best_detection(detection *dets, int num, float thresh, size_t class);
image crop_from_detection(const image im, const detection det, const int extend_px);

#endif // UTILS_H
