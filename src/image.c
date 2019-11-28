#include "image.h"

#include <darknet.h>

void show_detections(const image im, detection *dets, const int num, const float thresh, char **names, const int classes)
{
    image **alphabet = load_alphabet();
    image canvas = copy_image(im);
    draw_detections(canvas, dets, num, thresh, names, alphabet, classes);
    fflush(stdout);
    show_image(canvas, "Detections", 0);
    free_image(canvas);
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

void detect_image(char *cfgfile, char *weightfile, const float thresh, char *input, char **names, const int classes, const int show, const int extend_px, const char *output)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    image im = load_image_color(input, 0, 0);
    image sized = resize_image(im, net->w, net->h);
    network_predict(net, sized.data);
    int num = 0;
    detection* dets = get_network_boxes(net, 1, 1, thresh, 0, NULL, 0, &num);
    do_nms_sort(dets, num, classes, thresh);
    if (show)
        show_detections(im, dets, num, thresh, names, classes);
    detection best = best_detection(dets, num, thresh, 0);
    image cropped = crop_from_detection(im, best, extend_px);
    if (show)
        show_image(cropped, output, 0);
    save_image(cropped, output);
    free_detections(dets, num);
    free_image(im);
    free_image(sized);
    free_image(cropped);
}
