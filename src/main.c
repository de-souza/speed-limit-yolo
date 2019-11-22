#include <stdio.h>
#include <darknet.h>

int main() {
    char *cfgfile = DARKNET_PATH "/cfg/yolov3-tiny.cfg";
    char *weightfile = DARKNET_PATH "/yolov3-tiny.weights";
    char *input = DARKNET_PATH "/data/dog.jpg";
    float thresh = 0.5;

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    image im = load_image_color(input, 0, 0);
    image sized = resize_image(im, net->w, net->h);
    float *X = sized.data;
    double time = what_time_is_it_now();
    network_predict(net, X);
    printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now() - time);

    int nboxes = 0;
    detection* dets = get_network_boxes(net, im.w, im.h, thresh, thresh, &nboxes, 1, &nboxes);
    printf("Detected '%d' obj, class %d\n", nboxes, dets->classes);

    for (int i = 0; i < nboxes; ++i)
        for (int j = 0; j < dets->classes; ++j)
            if (dets[i].prob[j] > thresh) {
                int x = (int) ((dets[i].bbox.x - dets[i].bbox.w/2) * im.w);
                int y = (int) ((dets[i].bbox.y - dets[i].bbox.h/2) * im.h);
                int w = (int) (dets[i].bbox.w * im.w);
                int h = (int) (dets[i].bbox.h * im.h);
                image cropped = crop_image(im, x, y, w, h);
                save_image(cropped, "prediction");
                show_image(cropped, "prediction", 1000);
            }

    free_detections(dets, nboxes);
    free_image(im);
    free_image(sized);

    return 0;
}
