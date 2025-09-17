#include <atomic>
#include <chrono>
#include <thread>

#include <ros/ros.h>
#include <rosbag/bag.h>

#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64.h>

#include <sensor_msgs/Image.h>

#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <ackermann_msgs/AckermannDriveStamped.h>

#include <cv_bridge/cv_bridge.h>

#include <XmlRpcValue.h>

#include <tf2_msgs/TFMessage.h>
#include <opencv2/videoio.hpp>

#include <utils/rosparams_utils.hpp>
#include <utils/execution_status.hpp>
#include <interface/rosbag_record.hpp>
#include <interface/StampedMarkers.h>

class topic_to_video_t
{
public:
  topic_to_video_t(ros::NodeHandle& nh, const std::string video_name, const std::string image_topic, const int width,
                   const int height)
    : _output_video(video_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(width, height))
    , _size(width, height)
  {
    _timer = nh.createTimer(ros::Duration(1 / 30.0), &topic_to_video_t::write_frame, this);
    _image_subscriber = nh.subscribe(image_topic, 1, &topic_to_video_t::get_image, this);
  }

  void stop_writer()
  {
    _output_video.release();
  }

private:
  void get_image(const sensor_msgs::ImageConstPtr message)
  {
    const cv_bridge::CvImagePtr frame{ cv_bridge::toCvCopy(message) };
    cv::resize(frame->image, _frame, _size);
    // _frame = ;
  }

  void write_frame(const ros::TimerEvent& event)
  {
    if (_output_video.isOpened())
    {
      _output_video.write(_frame);
    }
  }

  cv::Size _size;
  ros::Timer _timer;
  cv::Mat _frame;
  cv::VideoWriter _output_video;
  ros::Subscriber _image_subscriber;
};
int main(int argc, char** argv)
{
  ros::init(argc, argv, "topic_to_video");

  ros::NodeHandle nh("~");

  std::string video_directory;
  std::string stop_topic;
  std::string image_topic;
  int width;
  int height;
  std::string video_id;

  ROS_PARAM_SETUP(nh, image_topic);
  ROS_PARAM_SETUP(nh, video_directory);
  ROS_PARAM_SETUP(nh, stop_topic);
  ROS_PARAM_SETUP(nh, width);
  ROS_PARAM_SETUP(nh, height);
  ROS_PARAM_SETUP(nh, video_id);
  // ros::Subscriber _rgb_subscriber = nh.subscribe(image_topic, 1, &aruco_wTc_nodelet_t::get_image, this);
  utils::execution_status_t execution_status(nh, stop_topic);

  std::ostringstream video_name;
  video_name << video_directory << "/v_" << utils::timestamp() << "_" << video_id << ".avi";
  ROS_INFO_STREAM("Video name: " << video_name.str());

  topic_to_video_t topic_to_video(nh, video_name.str(), image_topic, width, height);
  while (execution_status.ok())
  {
    ros::spinOnce();
  }

  topic_to_video.stop_writer();

  return 0;
}