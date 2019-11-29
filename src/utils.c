#include "utils.h"

image **load_alphabet()
{
    int i, j;
    const int nsize = 8;
    image **alphabets = calloc(nsize, sizeof(image));
    for (j = 0; j < nsize; ++j) {
        alphabets[j] = calloc(128, sizeof(image));
        for (i = 32; i < 127; ++i) {
            char buff[256];
            sprintf(buff, DARKNET_PATH "/data/labels/%d_%d.png", i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}

detection best_detection(detection *dets, int num, float thresh, size_t class)
{
    for (int i = 0; i < num; ++i)
        if (dets[i].prob[class] > thresh)
            return dets[i];
    return dets[0];
}

image crop_from_detection(const image im, const detection det, const int extend_px)
{
    int dx = (int) ((det.bbox.x - (det.bbox.w/2)) * im.w) - extend_px;
    int dy = (int) ((det.bbox.y - (det.bbox.h/2)) * im.h) - extend_px;
    int w = (int) (det.bbox.w * im.w) + 2*extend_px;
    int h = (int) (det.bbox.h * im.h) + 2*extend_px;
    return crop_image(im, dx, dy, w, h);
}
