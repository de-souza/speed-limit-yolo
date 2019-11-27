#include <stdio.h>
#include <darknet.h>
#include <tesseract/capi.h>

void show_detections(image im, detection *dets, int num, float thresh, char **names, int classes)
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

image crop_from_detection(image im, detection det)
{
    int dx = (int) ((det.bbox.x - (det.bbox.w/2)) * im.w);
    int dy = (int) ((det.bbox.y - (det.bbox.h/2)) * im.h);
    int w = (int) (det.bbox.w * im.w);
    int h = (int) (det.bbox.h * im.h);
    return crop_image(im, dx, dy, w, h);
}

int main()
{
    char *cfgfile = "cfg/saferauto.cfg";
    char *weightfile = "cfg/saferauto.weights";
    char *input = DARKNET_PATH "/data/FullIJCNN2013/00083.ppm";
    float thresh = 0.5;
    char *names[] = {"prohibitory", "danger", "mandatory", "stop", "yield"};
    int classes = (int) (sizeof(names) / sizeof(char *));
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    image im = load_image_color(input, 0, 0);
    image sized = resize_image(im, net->w, net->h);
    network_predict(net, sized.data);
    int num = 0;
    detection* dets = get_network_boxes(net, 1, 1, thresh, 0, NULL, 0, &num);
    do_nms_sort(dets, num, classes, thresh);
    show_detections(im, dets, num, thresh, names, classes);
    detection best = best_detection(dets, num, thresh, 0);
    image cropped = crop_from_detection(im, best);
    show_image(cropped, "Detection", 0);
    free_detections(dets, num);
    free_image(im);
    free_image(sized);
    free_image(cropped);
}
