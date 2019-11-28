#ifndef VIDEO_H
#define VIDEO_H

void detect_video(char *cfgfile, char *weightfile, const float thresh, const char *filename, char **names, const int classes);

#endif // VIDEO_H
