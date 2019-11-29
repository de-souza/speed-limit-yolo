#ifndef VIDEO_H
#define VIDEO_H

#include <darknet.h>
#include <opencv2/videoio/videoio_c.h>

void my_embed_image(image source, image dest, int dx, int dy);
image my_make_empty_image(int w, int h, int c);
void my_letterbox_image_into(image im, int w, int h, image boxed);
void ipl_into_image(IplImage *src, image im);
image my_ipl_to_image(IplImage *src);
image my_get_image_from_stream(CvCapture *cap);
int size_network(network *net);
void remember_network(network *net);
detection *avg_predictions(network *net, int *nboxes);
void *detect_in_thread(void *ptr);
void *fetch_in_thread(void *ptr);
void *display_in_thread(void);
void detect_video(char *cfgfile, char *weightfile, const float thresh, const char *filename, char **names, const int classes);

#endif // VIDEO_H
