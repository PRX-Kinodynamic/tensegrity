#pragma once

// Copy and other interface utils
namespace interface
{

// Copy implementations:
// 1) copy(Obj, Obj) --> Implemented in another, obj-specific file
// 2) copy(Obj, Ptr)
// 3) copy(Ptr, Obj)
// 4) copy(Ptr, Ptr)

// Version 2
template <typename Out, std::enable_if_t<not tensegrity::utils::is_any_ptr<Out>::value, bool> = true,  // no-lint
          typename In, std::enable_if_t<tensegrity::utils::is_any_ptr<In>::value, bool> = true>
inline void copy(Out& msg, const In state)
{
  copy(msg, *state);
}

// Version 3
template <typename In, std::enable_if_t<tensegrity::utils::is_any_ptr<In>::value, bool> = true,  // no-lint
          typename Out, std::enable_if_t<not tensegrity::utils::is_any_ptr<Out>::value, bool> = true>
inline void copy(In msg, const Out& state)
{
  copy(*msg, state);
}

// Version 4
template <typename In, std::enable_if_t<tensegrity::utils::is_any_ptr<In>::value, bool> = true,  // no-lint
          typename Out, std::enable_if_t<tensegrity::utils::is_any_ptr<Out>::value, bool> = true>
inline void copy(In msg, const Out state)
{
  copy(*msg, *state);
}

template <typename Msg>
void to_file(const Msg& msg, const std::string filename, const std::ios_base::openmode mode = std::ofstream::trunc)
{
  std::ofstream ofs;
  ofs.open(filename.c_str(), mode);
  to_file(msg, ofs);
  ofs.close();
}

inline void to_file(const std_msgs::Header& msg, std::ofstream& ofs)
{
  ofs << msg.stamp << " ";
  ofs << msg.seq << " ";
  ofs << msg.frame_id << " ";
}
}  // namespace interface