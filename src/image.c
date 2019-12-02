#ifdef CLIENTIO
#include <clientio.h>
#endif // CLIENTIO

#include "image.h"
#include "ocr.h"
#include "utils.h"

void show_detections(const image im, detection *dets, const int num, const float thresh, char **names, const int classes)
{
    image **alphabet = load_alphabet();
    image canvas = copy_image(im);
    draw_detections(canvas, dets, num, thresh, names, alphabet, classes);
    fflush(stdout);
    show_image(canvas, "Detections", 0);
    free_image(canvas);
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
    int detected = num != 0;
    if (detected) {
        do_nms_sort(dets, num, classes, thresh);
        if (show)
            show_detections(im, dets, num, thresh, names, classes);
        detection best = best_detection(dets, num, thresh, 0);
        image cropped = crop_from_detection(im, best, extend_px);
        save_image(cropped, output);
        free_image(cropped);
    }
    free_image(im);
    free_image(sized);
    free_detections(dets, num);
    if (detected) {
        int speed_limit = recognize_number(input, show);
        printf("Speed limit: %d\n", speed_limit);
#ifdef CLIENTIO
        char message[24];
        size_t len = sizeof(message);
        snprintf(message, len, "CANN SPEED_LIMIT %d", speed_limit);
        int sockfd = create_connected_socket("127.0.0.1", 2222);
        send_message(sockfd, message, len);
        close_socket(sockfd);
#endif // CLIENTIO
    } else {
        puts("No detection.");
    }
}
