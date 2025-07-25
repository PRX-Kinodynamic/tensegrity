#include <iostream>

#include <ros/ros.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

#include <tensegrity_utils/dbg_utils.hpp>
#include <tensegrity_utils/std_utils.hpp>
#include <tensegrity_utils/rosparams_utils.hpp>

using namespace cv;
const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;
int dilation_size = 0, erosion_size = 0;
bool new_image = false;
cv_bridge::CvImagePtr frame;
static void on_low_H_thresh_trackbar(int v_in, void*)
{
  // DEBUG_VARS(v_in, low_H);
  low_H = min(high_H - 1, v_in);
  // setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int v_in, void*)
{
  high_H = max(v_in, low_H + 1);
  // setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int v_in, void*)
{
  low_S = min(high_S - 1, v_in);
  // setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int v_in, void*)
{
  high_S = max(v_in, low_S + 1);
  // setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int v_in, void*)
{
  low_V = min(high_V - 1, v_in);
  // setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int v_in, void*)
{
  high_V = max(v_in, low_V + 1);
  // setTrackbarPos("High V", window_detection_name, high_V);
}
static void Dilation(int v_in, void*)
{
  dilation_size = min(v_in, 21);
}
static void Erosion(int v_in, void*)
{
  erosion_size = min(v_in, 21);
}

void image_callback(const sensor_msgs::ImageConstPtr message)
{
  PRINT_MSG_ONCE("First image received");
  // cv_bridge::CvImageConstPtr frame_in{ cv_bridge::toCvCopy(message) };
  frame = cv_bridge::toCvCopy(message);
  new_image = true;
}

int main(int argc, char* argv[])
{
  // VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
  ros::init(argc, argv, "Color_filter_manual_calib");
  ros::NodeHandle nh("~");
  namedWindow(window_capture_name);
  namedWindow(window_detection_name);
  namedWindow("Dilation");
  namedWindow("Erosion");

  std::string image_topic;
  PARAM_SETUP(nh, image_topic);

  ros::Subscriber image_subscriber{ nh.subscribe(image_topic, 1, image_callback) };
  // ros::Publisher publisher_gt{ nh.advertise<prx_models::Tree>(sln_tree_topic, 1, true) };
  //   createTrackbar( "Canny thresh:", source_window, nullptr, thresh_max, thresh_callback );
  // setTrackbarPos( "Canny thresh:", source_window, thresh);
  // Trackbars to set thresholds for HSV values
  createTrackbar("Low H", window_detection_name, nullptr, max_value_H, on_low_H_thresh_trackbar);
  createTrackbar("High H", window_detection_name, nullptr, max_value_H, on_high_H_thresh_trackbar);
  createTrackbar("Low S", window_detection_name, nullptr, max_value, on_low_S_thresh_trackbar);
  createTrackbar("High S", window_detection_name, nullptr, max_value, on_high_S_thresh_trackbar);
  createTrackbar("Low V", window_detection_name, nullptr, max_value, on_low_V_thresh_trackbar);
  createTrackbar("High V", window_detection_name, nullptr, max_value, on_high_V_thresh_trackbar);
  createTrackbar("Dilation: ", "Dilation", nullptr, 21, Dilation);
  createTrackbar("Erosion:", "Erosion", nullptr, 21, Erosion);
  Mat frame_HSV, frame_threshold, dilation_dst, erosion_dst;
  ros::Rate rate(10);
  while (ros::ok())
  {
    // cap >> frame;
    if (frame)
    {
      // break;
      // Convert from BGR to HSV colorspace
      cvtColor(frame->image, frame_HSV, COLOR_BGR2HSV);
      // Detect the object based on HSV Range Values
      inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
      // Show the frames
      imshow(window_capture_name, frame->image);
      imshow(window_detection_name, frame_threshold);

      Mat element = getStructuringElement(cv::MORPH_ELLIPSE, Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                          Point(dilation_size, dilation_size));
      Mat element_erosion = getStructuringElement(cv::MORPH_ELLIPSE, Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                  Point(erosion_size, erosion_size));
      erode(frame_threshold, erosion_dst, element_erosion);
      dilate(frame_threshold, dilation_dst, element);
      imshow("Dilation", dilation_dst);
      imshow("Erosion", erosion_dst);

      waitKey(100);
      new_image = false;
      // DEBUG_PRINT
    }
    // char key = (char)waitKey(30);
    // if (key == 'q' || key == 27)
    // {
    //   break;
    // }
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}