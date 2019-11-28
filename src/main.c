#include <stdlib.h>
#include <getopt.h>
#include <darknet.h>
#include <leptonica/allheaders.h>
#include <tesseract/capi.h>

_Noreturn void usage(const char *name)
{
    fprintf(stderr, "Usage: %s [-a] [-c] [-e enlarge_px] file\n", name);
    exit(1);
}

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

image crop_from_detection(image im, detection det, int enlarge_px)
{
    int dx = (int) ((det.bbox.x - (det.bbox.w/2)) * im.w) - enlarge_px;
    int dy = (int) ((det.bbox.y - (det.bbox.h/2)) * im.h) - enlarge_px;
    int w = (int) (det.bbox.w * im.w) + 2*enlarge_px;
    int h = (int) (det.bbox.h * im.h) + 2*enlarge_px;
    return crop_image(im, dx, dy, w, h);
}

int parse_number(const char *str)
{
    char *parsed = malloc(strlen(str) + 1);
    strcpy(parsed, str);
    char *source = parsed;
    char *dest = parsed;
    while ((source = strpbrk(source, "0123456789")))
        *dest++ = *source++;
    *dest = '\0';
    int num = atoi(parsed);
    free(parsed);
    return num;
}

int main(int argc, char **argv)
{

    int detections = 0;
    int crop = 0;
    int enlarge_px = 0;
    int opt;
    while ((opt = getopt(argc, argv, "ace:")) != -1)
        switch (opt) {
        case 'a': detections = 1; break;
        case 'c': crop = 1; break;
        case 'e': enlarge_px = atoi(optarg); break;
        case -1: puts("-1"); break;
        default: usage(argv[0]);
        }
    if (!argv[optind])
        usage(argv[0]);
    char *cfgfile = "cfg/saferauto.cfg";
    char *weightfile = "cfg/saferauto.weights";
    float thresh = 0.5;
    char *names[] = {"prohibitory", "danger", "mandatory", "stop", "yield"};
    int classes = (int) (sizeof(names) / sizeof(char *));
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    image im = load_image_color(argv[optind], 0, 0);
    image sized = resize_image(im, net->w, net->h);
    network_predict(net, sized.data);
    int num = 0;
    detection* dets = get_network_boxes(net, 1, 1, thresh, 0, NULL, 0, &num);
    do_nms_sort(dets, num, classes, thresh);
    if (detections)
        show_detections(im, dets, num, thresh, names, classes);
    detection best = best_detection(dets, num, thresh, 0);
    image cropped = crop_from_detection(im, best, enlarge_px);
    if (crop)
        show_image(cropped, "detection", 0);
    save_image(cropped, "detection");
    free_detections(dets, num);
    free_image(im);
    free_image(sized);
    free_image(cropped);
    TessBaseAPI *handle = TessBaseAPICreate();
    PIX *img = pixRead("detection.jpg");
    TessBaseAPIInit3(handle, NULL, "eng");
    TessBaseAPISetVariable(handle, "tessedit_char_whitelist", "0123456789");
    TessBaseAPISetImage2(handle, img);
    TessBaseAPISetSourceResolution(handle, 70);
    TessBaseAPIRecognize(handle, NULL);
    char *text = TessBaseAPIGetUTF8Text(handle);
    printf("OCR: %s\n", text);
    printf("Result: %d\n", parse_number(text));
    TessDeleteText(text);
    TessBaseAPIEnd(handle);
    TessBaseAPIDelete(handle);
    pixDestroy(&img);
}
