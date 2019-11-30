#ifdef CLIENTIO
#include <clientio.h>
#endif // CLIENTIO
#include <opencv2/highgui/highgui_c.h>

#include "ocr.h"
#include "utils.h"
#include "video.h"

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;
static network *net;
static image buff[3];
static image buff_letter[3];
static int buff_index;
static void *cap;
static double fps;
static float demo_thresh;
static float demo_hier;
static int demo_frame;
static int demo_index;
static float **predictions;
static float *avg;
static int demo_done;
static int demo_total;
static int demo_show;
static int demo_extend_px;
static const char* demo_output;

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c)
        return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

void my_embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k)
        for(y = 0; y < source.h; ++y)
            for(x = 0; x < source.w; ++x) {
                float val = get_pixel(source, x, y, k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
}

image my_make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

void my_letterbox_image_into(image im, int w, int h, image boxed)
{
    int new_w = im.w;
    int new_h = im.h;
    if ((w/im.w) < (h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    my_embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);
    free_image(resized);
}

void ipl_into_image(IplImage *src, image im)
{
    unsigned char *data = (unsigned char *) src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;
    for(i = 0; i < h; ++i)
        for(k = 0; k < c; ++k)
            for(j = 0; j < w; ++j)
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/(float) 255.;
}

image my_ipl_to_image(IplImage *src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
}

image my_get_image_from_stream(CvCapture *cap)
{
    IplImage* src = cvQueryFrame(cap);
    if (!src)
        return my_make_empty_image(0, 0, 0);
    image im = my_ipl_to_image(src);
    rgbgr_image(im);
    return im;
}

int size_network(network *net)
{
    int i;
    int count = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
            count += l.outputs;
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * (unsigned) l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for (j = 0; j < demo_frame; ++j)
        axpy_cpu(demo_total, ((float) 1.)/demo_frame, predictions[j], 1, avg, 1);
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
            memcpy(l.output, avg + count, sizeof(float) * (unsigned) l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

void *detect_in_thread(void *ptr)
{
    (void) ptr;
    int running = 1;
    float nms = (float) 0.4;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n", fps);

    if (nboxes) {
        if (nms > 0)
            do_nms_obj(dets, nboxes, l.classes, nms);

        image display = buff[(buff_index+2) % 3];

        detection best = best_detection(dets, nboxes, demo_thresh, 0);
        image cropped = crop_from_detection(display, best, demo_extend_px);
        save_image(cropped, demo_output);

        puts("Objects:\n");
        draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
        fflush(stdout);

        if (demo_show)
            show_image(cropped, demo_output, 1000);

        free_image(cropped);

        int speed_limit = recognize_number("pred.jpg", demo_show);
        printf("Speed limit: %d\n", speed_limit);
#ifdef CLIENTIO
        char message[16];
        size_t len = sizeof(message);
        snprintf(message, len, "CANN LIMIT %d", speed_limit);
        int sockfd = create_connected_socket("127.0.0.1", 2222);
        send_message(sockfd, message, len);
        close_socket(sockfd);
#endif // CLIENTIO
    } else {
        puts("No detection.");
    }

    free_detections(dets, nboxes);

    demo_index = (demo_index+1) % demo_frame;
    running = 0;
    return 0;
}

void *fetch_in_thread(void *ptr)
{
    (void) ptr;
    free_image(buff[buff_index]);
    buff[buff_index] = my_get_image_from_stream(cap);
    if (buff[buff_index].data == 0) {
        demo_done = 1;
        return 0;
    }
    my_letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    return 0;
}

void *display_in_thread(void)
{
    int c = show_image(buff[(buff_index+1) % 3], "Demo", 1);
    if (c != -1)
        c = c % 256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += (float) .02;
    } else if (c == 84) {
        demo_thresh -= (float) .02;
        if (demo_thresh <= (float) .02)
            demo_thresh = (float) .02;
    } else if (c == 83) {
        demo_hier += (float) .02;
    } else if (c == 81) {
        demo_hier -= (float) .02;
        if (demo_hier <= (float) .0)
            demo_hier = (float) .0;
    }
    return 0;
}

void detect_video(char *cfgfile, char *weightfile, const float thresh, const char *filename, char **names, const int classes, const int show, const int extend_px, const char *output)
{
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = 0;
    demo_show = show;
    demo_extend_px = extend_px;
    demo_output = output;

    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand((unsigned) time(NULL));

    int i;
    demo_total = size_network(net);
    demo_frame = 3;
    predictions = calloc((unsigned) demo_frame, sizeof(float *));
    for (i = 0; i < demo_frame; ++i)
        predictions[i] = calloc((unsigned) demo_total, sizeof(float));
    avg = calloc((unsigned) demo_total, sizeof(float));

    if (filename) {
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    } else {
        cap = cvCaptureFromCAM(0);
    }

    if (!cap)
        error("Couldn't connect to webcam.\n");

    buff[0] = my_get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);

    int count = 0;

    cvNamedWindow("Demo", CV_WINDOW_AUTOSIZE);

    double demo_time = what_time_is_it_now();

    while (!demo_done) {
        buff_index = (buff_index + 1) % 3;
        if (pthread_create(&fetch_thread, NULL, fetch_in_thread, NULL))
            error("Thread creation failed");
        if (pthread_create(&detect_thread, NULL, detect_in_thread, NULL))
            error("Thread creation failed");
        fps = 1. / (what_time_is_it_now()-demo_time);
        demo_time = what_time_is_it_now();
        display_in_thread();
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }
}
