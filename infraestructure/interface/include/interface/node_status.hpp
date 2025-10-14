#pragma once

// Ros
#include <ros/ros.h>

#include <interface/NodeStatus.h>
#include <tensegrity_utils/assert.hpp>
namespace interface
{

class node_status_t
{
  using This = node_status_t;
  using StatusType = uint8_t;
  // using interface::NodeStatus

  void init(ros::NodeHandle& nh, const std::string node_id)
  {
    _msg.status = NodeStatus::PREPARING;
    const std::string change_topic{ node_id + "/status/change" };
    const std::string current_topic{ node_id + "/status/current" };
    if (not _observer_only)
    {
      _status_publisher = nh.advertise<interface::NodeStatus>(current_topic, 1, true);
      _timer = nh.createTimer(ros::Rate(1.0), &This::update, this);
      _status_publisher.publish(_msg);
      _status_subscriber = nh.subscribe(change_topic, 1, &This::callback, this);
    }
    else
    {
      _status_subscriber = nh.subscribe(current_topic, 1, &This::callback, this);
    }
    // DEBUG_VARS(change_topic)
  }

public:
  // TODO: Should move it to private
  node_status_t(ros::NodeHandle& nh, bool observer_only = false) : _observer_only(observer_only)
  {
    std::string node_id;
    PARAM_SETUP(nh, node_id);
    init(nh, node_id);
  }

  node_status_t(ros::NodeHandle& nh, const std::string node_id, bool observer_only = false)
    : _observer_only(observer_only)
  {
    init(nh, node_id);
    // _msg.status = NodeStatus::PREPARING;
    // const std::string change_topic{ node_id + "/status/change" };
    // const std::string current_topic{ node_id + "/status/current" };
    // _status_publisher = nh.advertise<interface::NodeStatus>(current_topic, 1, true);
    // _status_subscriber = nh.subscribe(change_topic, 1, &This::callback, this);

    // _timer = nh.createTimer(ros::Rate(1.0), &This::update, this);
    // _status_publisher.publish(_msg);
  }
  // enum status_t
  // {
  //   PREPARING = 0,  // Initial status: The node is "up" but not ready to operate (i.e. is reading a config file)
  //   Ready = 1,      // The node is either ready to run or actively running
  //   Running = 2,    // The node is either ready to run or actively running
  //   Waiting = 3,    // The node was "running" but it is no longer running, i.e. paused. It may go back to running
  //   STOPPED = 4     // The node has been STOPPED and will exit soon. May be doing some cleanup (i.e. writing to
  //   file).
  // };

  ~node_status_t()
  {
    // TODO: Publish stopped, but constructor needs to be private
  }

  template <typename... Ts>
  static std::shared_ptr<node_status_t> create(ros::NodeHandle& nh, Ts... args)
  {
    return std::make_shared<node_status_t>(nh, args...);
  }

  // static std::shared_ptr<node_status_t> create(ros::NodeHandle& nh, const std::string node_id)
  // {
  //   return std::make_shared<node_status_t>(nh, node_id);
  // }

  void update(const ros::TimerEvent& t)
  {
    // if (_msg.status == NodeStatus::FINISH)
    // {
    //   ros::shutdown();
    // }
    _status_publisher.publish(_msg);
  }

  // TODO: add option to call custom function to check status
  virtual void callback(const interface::NodeStatusConstPtr msg)
  {
    try
    {
      status_change(msg->status);
    }
    catch (std::exception)
    {
      PRINT_MSG("[node_status_t] invalid message received.");
    }
  }

  friend std::ostream& operator<<(std::ostream& ost, const node_status_t& obj)
  {
    std::string str;
    switch (obj._msg.status)
    {
      case NodeStatus::PREPARING:
        str = "PREPARING";
        break;
      case NodeStatus::READY:
        str = "READY";
        break;
      case NodeStatus::RUNNING:
        str = "RUNNING";
        break;
      case NodeStatus::WAITING:
        str = "WAITING";
        break;
      case NodeStatus::STOPPED:
        str = "STOPPED";
        break;
      case NodeStatus::FINISH:
        str = "FINISH";
        break;
      default:
        TENSEGRITY_THROW("[node_status_t] Status unknown");
    }
    ost << str;
    return ost;
  }

  friend std::ostream& operator<<(std::ostream& ost, const std::shared_ptr<node_status_t>& obj)
  {
    ost << *obj;
    return ost;
  }

  inline StatusType status() const
  {
    return _msg.status;
  }

  inline void status(const StatusType new_status)
  {
    status_change(new_status);
  }

private:
  void status_change(const StatusType new_status)
  {
    const StatusType current{ _msg.status };
    // StatusType next;
    // switch (current)
    // {
    //   case NodeStatus::PREPARING:
    //     next = status_preparing(new_status);
    //     break;
    //   case NodeStatus::READY:
    //     next = status_ready(new_status);
    //     break;
    //   case NodeStatus::RUNNING:
    //     next = status_running(new_status);
    //     break;
    //   case NodeStatus::WAITING:
    //     next = status_waiting(new_status);
    //     break;
    //   case NodeStatus::STOPPED:
    //     next = status_stopped(new_status);
    //     break;
    //   default:
    //     TENSEGRITY_THROW("[node_status_t] Status unknown");
    // }

    // DEBUG_VARS(new_status)
    _msg.status = new_status;
    if (not _observer_only)
      _status_publisher.publish(_msg);
  }

  StatusType status_preparing(const StatusType new_status)
  {
    return new_status;  // All valid on PREPARING
  }

  StatusType status_ready(const StatusType new_status)
  {
    switch (new_status)
    {
      case NodeStatus::PREPARING:
        TENSEGRITY_THROW("[node_status_t] Current status is Running; requested PREPARING");
        return NodeStatus::READY;
      case NodeStatus::STOPPED:
        TENSEGRITY_THROW("[node_status_t] Current status is STOPPED; requested PREPARING");
        return NodeStatus::STOPPED;
      default:
        return new_status;
    }
  }

  StatusType status_running(const StatusType new_status)
  {
    switch (new_status)
    {
      case NodeStatus::PREPARING:
        TENSEGRITY_THROW("[node_status_t] Current status is Running; requested PREPARING");
        return NodeStatus::PREPARING;
      case NodeStatus::STOPPED:
        TENSEGRITY_THROW("[node_status_t] Current status is STOPPED; requested PREPARING");
        return NodeStatus::STOPPED;
      default:
        return new_status;
    }
  }

  StatusType status_waiting(const StatusType new_status)
  {
    switch (new_status)
    {
      case NodeStatus::PREPARING:
        TENSEGRITY_THROW("[node_status_t] Current status is Running; requested PREPARING");
        return NodeStatus::PREPARING;
      case NodeStatus::STOPPED:
        TENSEGRITY_THROW("[node_status_t] Current status is STOPPED; requested PREPARING");
        return NodeStatus::STOPPED;
      default:
        return new_status;
    }
  }

  StatusType status_stopped(const StatusType new_status)
  {
    switch (new_status)
    {
      case NodeStatus::STOPPED:
        return NodeStatus::STOPPED;
      default:
        TENSEGRITY_THROW("[node_status_t] Current status is STOPPED. No status change allow");
        return NodeStatus::STOPPED;
    }
  }

  StatusType status_finished(const StatusType new_status)
  {
    return NodeStatus::FINISH;
  }

  interface::NodeStatus _msg;

  bool _observer_only;

  ros::Timer _timer;
  ros::Publisher _status_publisher;
  ros::Subscriber _status_subscriber;
};
}  // namespace interface