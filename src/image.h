#ifndef IMAGE_H
#define IMAGE_H

#include <darknet.h>

void show_detections(const image im, detection *dets, const int num, const float thresh, char **names, const int classes);
detection best_detection(detection *dets, int num, float thresh, size_t class);
image crop_from_detection(const image im, const detection det, const int extend_px);
void detect_image(char *cfgfile, char *weightfile, const float thresh, char *input, char **names, const int classes, const int show, const int extend_px, const char *output);

#endif // IMAGE_H
