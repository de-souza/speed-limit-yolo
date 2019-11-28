#ifndef IMAGE_H
#define IMAGE_H

void detect_image(char *cfgfile, char *weightfile, const float thresh, char *input, char **names, const int classes, const int show, const int extend_px, const char *output);

#endif // IMAGE_H
