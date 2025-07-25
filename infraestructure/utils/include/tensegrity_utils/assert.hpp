#pragma once

#include <exception>
#include <string>
#include <sstream>
#include <unistd.h>
#include <iostream>
#include <execinfo.h>

#include <tensegrity_utils/constants.hpp>

namespace tensegrity
{
namespace utils
{

class assert_t : public std::exception
{
private:
  const char* expression;
  const char* file;
  int line;
  std::string message;
  std::string report;

public:
  class stream_t
  {
    std::ostringstream stream;

  public:
    operator std::string() const
    {
      return stream.str();
    }

    template <typename T>
    stream_t& operator<<(const T& value)
    {
      stream << value;
      return *this;
    }
  };

  void logger()
  {
    std::cerr << report << std::endl;
  }

  assert_t(const char* expression, const char* file, int line, const std::string& message)
    : expression(expression), file(file), line(line), message(message)
  {
    std::ostringstream out_stream;

    if (!message.empty())
    {
      out_stream << message << ": ";
    }

    out_stream << "Assertion '" << expression << "'";
    out_stream << " failed in file '" << file << "' line " << line;
    report = out_stream.str();
    logger();
  }

  assert_t(const std::string& message) : message(message)
  {
    std::ostringstream out_stream;

    if (!message.empty())
    {
      out_stream << message << ": ";
    }
    report = out_stream.str();
  }

  assert_t()
  {
  }

  assert_t get_backtrace(std::string message = "", size_t size = 10)
  {
    char** strings;
    void* buffer[size];
    std::ostringstream out_stream;

    // get void*'s for all entries on the stack
    int bt_size = backtrace(buffer, size);

    strings = backtrace_symbols(buffer, bt_size);
    if (!message.empty())
    {
      out_stream << message << ": " << std::endl;
    }

    for (int i = 0; i < bt_size; ++i)
    {
      // TODO: Print the line number
      out_stream << strings[i] << std::endl;
    }
    report = out_stream.str();
    return *this;
  }

  virtual const char* what() const throw()
  {
    return report.c_str();
  }

  const char* get_expression() const throw()
  {
    return expression;
  }

  const char* get_file() const throw()
  {
    return file;
  }

  int get_line() const throw()
  {
    return line;
  }

  const char* get_message() const throw()
  {
    return message.c_str();
  }

  ~assert_t() throw()
  {
  }
};
}  // namespace utils
}  // namespace tensegrity

#define TENSEGRITY_ASSERT(EXPRESSION, MESSAGE)                                                                         \
  if (!(EXPRESSION))                                                                                                   \
  {                                                                                                                    \
    using namespace tensegrity::utils;                                                                                 \
    throw assert_t(#EXPRESSION, __FILE__, __LINE__, (assert_t::stream_t() << MESSAGE));                                \
  }

#define TENSEGRITY_THROW(MESSAGE)                                                                                      \
  {                                                                                                                    \
    using namespace tensegrity::utils;                                                                                 \
    throw assert_t((assert_t::stream_t() << "[" << __FILE__ << ":" << __PRETTY_FUNCTION__ << ":" << __LINE__ << "] "   \
                                         << MESSAGE));                                                                 \
  }
#define TENSEGRITY_THROW_QUIET(MESSAGE)                                                                                \
  {                                                                                                                    \
    using namespace tensegrity::utils;                                                                                 \
    throw assert_t((assert_t::stream_t() << MESSAGE));                                                                 \
  }

/**
 * throw error printing the bactrace
 * @param  __VA_ARGS__ At most two arguments: First the message, second the max size of the stack to print.
 * @return             Throws the error printing the backtrace
 */
#define TENSEGRITY_THROW_BACKTRACE(...)                                                                                \
  {                                                                                                                    \
    using namespace tensegrity::utils;                                                                                 \
    throw assert_t().get_backtrace(__VA_ARGS__);                                                                       \
  }

#define TENSEGRITY_WARN(MESSAGE)                                                                                       \
  {                                                                                                                    \
    using namespace tensegrity::constants::color;                                                                      \
    std::cerr << yellow << "[PRX WARN] " << __PRETTY_FUNCTION__ << ":" << __LINE__ << " ";                             \
    std::cerr << MESSAGE << normal << std::endl;                                                                       \
  }

#define TENSEGRITY_WARN_COND(EXPRESSION, MESSAGE)                                                                      \
  if (!(EXPRESSION))                                                                                                   \
  {                                                                                                                    \
    TENSEGRITY_WARN(#EXPRESSION << " " << MESSAGE)                                                                     \
  }
