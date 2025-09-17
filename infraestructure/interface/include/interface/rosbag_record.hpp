#pragma once
#include <any>
#include <tensegrity_utils/rosparams_utils.hpp>
#include <tensegrity_utils/std_utils.hpp>
// #include <tensegrity_utils/execution_status.hpp>

namespace interface
{
void init_bag(rosbag::Bag* bag, const std::string rosbag_directory)
{
  std::ostringstream bag_name;
  bag_name << rosbag_directory << "/b_" << tensegrity::utils::timestamp() << ".bag";
  bag->open(bag_name.str(), rosbag::bagmode::Write);
  ROS_INFO_STREAM("Bag name: " << bag_name.str());
}

template <typename Msg>
class queued_callback_t
{
public:
  using TupleQueue = std::queue<std::tuple<std::string, ros::Time, typename Msg::ConstPtr>>;
  queued_callback_t() : _t0(ros::Time::now()){};
  queued_callback_t(const std::string topic_name) : _topic_name(topic_name)
  {
    std::cout << "topic_name: " << _topic_name << std::endl;
  };
  static inline std::mutex _queue_mutex;
  static inline TupleQueue _queue;

  void callback(const ros::MessageEvent<Msg const>& event)
  {
    const ros::Time t_now{ ros::Time::now() };
    if (t_now > _t0)
    {
      const std::string topic{ event.getConnectionHeader().at("topic") };
      _queue.push(std::make_tuple(topic, t_now, event.getMessage()));
    }
  }

private:
  std::string _topic_name;
  const ros::Time _t0;
};

template <typename Msg>
class queues_t
{
public:
  using QCallback = queued_callback_t<Msg>;
  using Subscribers = std::vector<ros::Subscriber>;

  bool register_topic(const std::string& topic_name, const std::string topic_type, const std::string expected_type,
                      Subscribers& subscribers, ros::NodeHandle& nh)
  {
    bool status{ false };
    if (topic_type == expected_type)  // Must be a nicer way of checking MsgType/topic_type == expected
    {
      _queues.emplace_back(topic_name);
      subscribers.push_back(nh.subscribe(topic_name, 100, &QCallback::callback, &_queues.back()));
      status = true;
    }
    return status;
  }

  std::size_t size() const
  {
    return _queues.size();
  }

  QCallback operator[](const std::size_t& idx) const
  {
    return _queues[idx];
  }

  QCallback& operator[](const std::size_t& idx)
  {
    return _queues[idx];
  }

private:
  std::vector<QCallback> _queues;
};

}  // namespace interface