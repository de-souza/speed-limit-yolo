#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "ocr.h"
#include "video.h"

int main(int argc, char **argv)
{
    int opt;
    int extend_px = 0;
    int show = 0;
    int video = 0;

    while ((opt = getopt(argc, argv, "e:sv")) != -1)
        switch (opt) {
        case 'e': extend_px = atoi(optarg); break;
        case 's': show = 1; break;
        case 'v': video = 1; break;
        default:
            fprintf(stderr, "Usage: %s [-e extend_px] [-sv] file\n", argv[0]);
            exit(1);
        }

    char *cfgfile = "cfg/saferauto.cfg";
    char *weightfile = "cfg/saferauto.weights";
    char *input = argv[optind];
    float thresh = 0.5;
    char *names[] = {"prohibitory", "danger", "mandatory", "stop", "yield"};
    int classes = (int) (sizeof(names) / sizeof(*names));

    if (!input || video) {
        detect_video(cfgfile, weightfile, thresh, input, names, classes, show, extend_px, "pred");
    } else {
        if (detect_image(cfgfile, weightfile, thresh, input, names, classes, show, extend_px, "pred"))
            printf("Detection: %d\n", recognize_number("pred.jpg", show));
        else
            puts("No detection.");
    }
}
