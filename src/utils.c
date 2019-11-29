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
