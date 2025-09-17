#include <thread>
#include <ros/ros.h>
#include <rosgraph_msgs/Clock.h>
#include <std_msgs/Float64.h>

// #include <ml4kp_bridge/defs.h>
// #include <utils/rosparams_utils.hpp>
// #include <utils/dbg_utils.hpp>
// #include <interface/SetDuration.h>

struct sim_clock_t
{
  sim_clock_t(ros::NodeHandle& nh) : _now(0.0)
  {
    const std::string duration_topic_name{ "/sim_clock/duration" };

    // duration_service = nh.advertiseService(duration_service_name, &sim_clock_t::service_callback, this);
    // clock_publisher = nh.advertise<std_msgs::Float64>(duration_topic_name, 1);
    _clock_publisher = nh.advertise<rosgraph_msgs::Clock>("/clock", 1);
    _duration_subscriber = nh.subscribe(duration_topic_name, 1, &sim_clock_t::callback, this);

    // nanosecs = std::chrono::round<std::chrono::nanoseconds>(std::chrono::duration<double>{ sleep_dur });
    // msg.clock.sec = 0.0;
    // msg.clock.nsec = 0.0;
  }

  void callback(const std_msgs::Float64ConstPtr msg)
  {
    const ros::Duration clock_step(msg->data);
    _now += clock_step;
    _clock_msg.clock.sec = _now.sec;
    _clock_msg.clock.nsec = _now.nsec;
    _clock_publisher.publish(_clock_msg);
  }

  void step_and_publish()
  {
    // for (; steps > 0; steps--)
    // {
    //   now = now + d;
    //   msg.clock.sec = now.sec;
    //   msg.clock.nsec = now.nsec;
    //   clock_publisher.publish(msg);
    //   std::this_thread::sleep_for(nanosecs);
    // }
  }

  ros::Time _now;

  ros::Publisher _clock_publisher;
  ros::Subscriber _duration_subscriber;

  rosgraph_msgs::Clock _clock_msg;
};

int main(int argc, char** argv)
{
  const std::string node_name{ "SimClock" };
  ros::init(argc, argv, node_name);
  ros::NodeHandle nh("~");

  sim_clock_t sim_clock(nh);
  ros::spin();
  return 0;
}