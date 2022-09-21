#include <omp.h>
#include "net.h"
#include <iostream>
#include <iomanip>

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <Sig17Slice.h>

static int colorization(const cv::Mat& bgr, const cv::Mat& out_image)
{
  ncnn::Net net;
  //net.opt.use_packing_layout = true;
  //net.opt.use_bf16_storage = true;
  //net.opt.num_threads = 1;
  net.opt.use_vulkan_compute = true;
  net.register_custom_layer("Sig17Slice", Sig17Slice_layer_creator);

  if (net.load_param("./models/siggraph17_color_sim.param"))
    exit(-1);
  if (net.load_model("./models/siggraph17_color_sim.bin"))
    exit(-1);
  //fixed input size for the pretrained network
  const int W_in = 256;
  const int H_in = 256;

  cv::Mat Base_img, lab, L, input_img;
  Base_img = bgr.clone();

  //normalize levels
  Base_img.convertTo(Base_img, CV_32F, 1.0/255);


  //Convert BGR to LAB color space format
  cvtColor(Base_img, lab, cv::COLOR_BGR2Lab);

  //Extract L channel
  cv::extractChannel(lab, L, 0);

  //Resize to input shape 256x256
  resize(L, input_img, cv::Size(W_in, H_in));

  //We subtract 50 from the L channel (for mean centering)
  //input_img -= 50;

  //convert to NCNN::MAT
  ncnn::Mat in_LAB_L(input_img.cols, input_img.rows, 1, (void*)input_img.data);
  in_LAB_L = in_LAB_L.clone();


  ncnn::Extractor ex = net.create_extractor();
  //set input, output lyers
  ex.input("input", in_LAB_L);

  //inference network
  ncnn::Mat out;
  //ex.extract("out_ab", out);
  ex.extract("out_ab", out);
  std::cout << "out matrix size " << out.w << " x "<< out.h <<"channels " << out.c <<std::endl;


  //create LAB material
  cv::Mat colored_LAB(out.h, out.w, CV_32FC2);
  //Extract ab channels from ncnn:Mat out
  memcpy((uchar*)colored_LAB.data, out.data, out.w * out.h * 2 * sizeof(float));

  //get separsted LAB channels a&b
  cv::Mat a(out.h, out.w, CV_32F, (float*)out.data);
  cv::Mat b(out.h, out.w, CV_32F, (float*)out.data + out.w * out.h);

  //Resize a, b channels to origina image size
  cv::resize(a, a, Base_img.size());
  cv::resize(b, b, Base_img.size());

  //merge channels, and convert back to BGR
  cv::Mat color, chn[] = {L, a, b};
  cv::merge(chn, 3, lab);
  cvtColor(lab, color, cv::COLOR_Lab2BGR);
  //normalize values to 0->255
  color.convertTo(color, CV_8UC3, 255);

  imshow("color", color);
  //imshow("original", Base_img);
  cv::imwrite("result_colored_out.png",color);
  cv::waitKey();
  return 0;
}

int main_colorization(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    cv::Mat out_image;
    colorization(m, out_image);

    return 0;
}
