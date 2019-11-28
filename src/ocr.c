#include "ocr.h"

#include <string.h>
#include <leptonica/allheaders.h>
#include <tesseract/capi.h>

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

int recognize_number(const char *input, const int show_result)
{
    TessBaseAPI *handle = TessBaseAPICreate();
    PIX *img = pixRead(input);
    TessBaseAPIInit3(handle, NULL, "eng");
    TessBaseAPISetVariable(handle, "tessedit_char_whitelist", "0123456789");
    TessBaseAPISetImage2(handle, img);
    TessBaseAPISetSourceResolution(handle, 70);
    TessBaseAPIRecognize(handle, NULL);
    char *text = TessBaseAPIGetUTF8Text(handle);
    if (show_result)
        printf("OCR: %s\n", text);
    int result = parse_number(text);
    TessDeleteText(text);
    TessBaseAPIEnd(handle);
    TessBaseAPIDelete(handle);
    pixDestroy(&img);
    return result;
}
