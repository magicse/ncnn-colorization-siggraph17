#include <omp.h>
#include <string>
#include <vector>
#include <ostream>
#include <random>
#include <chrono>

#include <stdio.h>
#include <unistd.h>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <colornet.h>

int main(int argc, char** argv)
{
    int opt = 0;
    char *in_fname = NULL;
    char *out_fname = NULL;

    while ((opt = getopt(argc, argv, "i:o:")) != -1) {
            switch(opt){
                case 'i':
                    in_fname = optarg;
                    printf("\nInput option value=%s", in_fname);
                    break;
                case 'o':
                    out_fname = optarg;
                    printf("\nOutput option value=%s", out_fname);
                    break;
                case '?':
                    if (optopt == 'i'){
                        printf("\nMissing input file name");
                    } else if (optopt == 'o') {
                         printf("\nMissing output file name");
                    } else {
                        printf("\nInvalid option received");
                        printf("\nUsage: COLOR_GAN -i input file name -o output file name");
                    }
                    break;
            }

    }

    char *argv_my[] = {"bw_imge", in_fname, NULL };
    main_colorization(2, argv_my);
printf("\nFinal");
return 0;
}
