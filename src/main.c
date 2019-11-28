#include <getopt.h>
#include <stdlib.h>
#include <darknet.h>
#include <leptonica/allheaders.h>
#include <tesseract/capi.h>

_Noreturn void usage(const char *executable)
{
    fprintf(stderr, "Usage: %s [-a] [-c] [-e extend_px] file\n", executable);
    exit(1);
}

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

void yolo_detect(char* input, const int show_dets, const int show_crop, const int extend_px, const char *output)
{
    char *cfgfile = "cfg/saferauto.cfg";
    char *weightfile = "cfg/saferauto.weights";
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
    if (show_dets)
        show_detections(im, dets, num, thresh, names, classes);
    detection best = best_detection(dets, num, thresh, 0);
    image cropped = crop_from_detection(im, best, extend_px);
    if (show_crop)
        show_image(cropped, output, 0);
    save_image(cropped, output);
    free_detections(dets, num);
    free_image(im);
    free_image(sized);
    free_image(cropped);
}

void tesseract_ocr(const char *input)
{
    TessBaseAPI *handle = TessBaseAPICreate();
    PIX *img = pixRead(input);
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

int main(int argc, char **argv)
{

    int opt;
    int show_dets = 0;
    int show_crop = 0;
    int extend_px = 0;
    while ((opt = getopt(argc, argv, "cde:")) != -1)
        switch (opt) {
        case 'c': show_crop = 1; break;
        case 'd': show_dets = 1; break;
        case 'e': extend_px = atoi(optarg); break;
        case -1: puts("-1"); break;
        default: usage(argv[0]);
        }
    if (!argv[optind])
        usage(argv[0]);
    yolo_detect(argv[optind], show_dets, show_crop, extend_px, "prediction");
    tesseract_ocr("prediction.jpg");
}
