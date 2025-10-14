#include <ros/init.h>
#include <gtest/gtest.h>
#include <tensegrity_utils/gnn.hpp>

namespace mock
{
template <typename Vector>
struct metric
{
  double operator()(const Vector& a, const Vector& b) const
  {
    double sum{ 0 };
    for (int i = 0; i < a.size(); ++i)
    {
      sum += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(sum);
  }
};

}  // namespace mock

TEST(GNN, empty_gnn)
{
  using State = std::array<double, 2>;
  using Metric = mock::metric<State>;
  // using Node = utils::nearest_neighbors::node_t<State>;
  // using Container = std::vector<Node>;
  using Gnn = utils::graph_nearest_neighbors_t<State, Metric>;

  Gnn gnn{};

  EXPECT_TRUE(gnn.size() == 0);
}

TEST(GNN, add_node_to_gnn)
{
  using State = std::array<double, 2>;
  using Metric = mock::metric<State>;
  using Gnn = utils::graph_nearest_neighbors_t<State, Metric>;
  using NodePtr = Gnn::NodePtr;

  Gnn gnn{};

  const State p0{ { 0, 0 } };

  gnn.add_node(p0);
  // Node& n0{ gnn.back() };
  // gnn_queries.add_node(n0, gnn);

  EXPECT_TRUE(gnn.size() == 1);
}

TEST(GNN, query_gnn_with_single_node)
{
  using State = std::array<double, 2>;
  using Metric = mock::metric<State>;
  using Gnn = utils::graph_nearest_neighbors_t<State, Metric>;
  using NodePtr = Gnn::NodePtr;

  Gnn gnn{};

  const State p0{ { 0, 0 } };
  const State p0p{ { 0, 1 } };  // P0^\prime

  gnn.emplace(p0);

  double dist0, dist0p;
  const State result0{ gnn.single_query(p0, dist0) };
  const State result0p{ gnn.single_query(p0p, dist0p) };

  const std::size_t expected_idx{ 0 };
  const double expected_dist0{ 0.0 };
  const double expected_dist0p{ 1.0 };

  EXPECT_EQ(result0, p0);
  EXPECT_DOUBLE_EQ(dist0, expected_dist0);

  EXPECT_EQ(result0p, p0);
  EXPECT_DOUBLE_EQ(dist0p, expected_dist0p);
}

TEST(GNN, query_gnn_with_single_node_high_dim)
{
  using State = std::array<double, 10>;
  using Metric = mock::metric<State>;
  using Gnn = utils::graph_nearest_neighbors_t<State, Metric>;
  using NodePtr = Gnn::NodePtr;

  Gnn gnn{};

  const State p0{ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
  State p0p{ p0 };  // P0^\prime
  p0p[0] = 1.0;

  gnn.emplace(p0);

  double dist0, dist0p;
  const State result0{ gnn.single_query(p0, dist0) };
  const State result0p{ gnn.single_query(p0p, dist0p) };

  // const std::size_t expected_idx{ Node::index(n0) };
  const double expected_dist0{ 0.0 };
  const double expected_dist0p{ 1.0 };

  EXPECT_EQ(result0, p0);
  EXPECT_DOUBLE_EQ(dist0, expected_dist0);

  EXPECT_EQ(result0p, p0p);
  EXPECT_DOUBLE_EQ(dist0p, expected_dist0p);
}

TEST(GNN, query_gnn_with_multiple_nodes)
{
  using State = std::array<double, 2>;
  using Metric = mock::metric<State>;
  using Gnn = utils::graph_nearest_neighbors_t<State, Metric>;
  using NodePtr = Gnn::NodePtr;

  Gnn gnn{};

  // Creating a square
  const State p0{ { 0, 0 } };
  const State p1{ { 0, 1 } };
  const State p2{ { 1, 0 } };
  const State p3{ { 1, 1 } };

  gnn.emplace(p0);
  gnn.emplace(p1);
  gnn.emplace(p2);
  gnn.emplace(p3);

  double dist0, dist0p, dist1;
  const State result0{ gnn.single_query(p0, dist0) };
  const State result0p{ gnn.single_query(State({ 0.49, 0.49 }), dist0p) };  // closest to P0
  const State result1{ gnn.single_query(State({ 2.0, 2.0 }), dist1) };      // closest to P0

  const double expected_dist0{ 0.0 };
  const double expected_dist0p{ std::sqrt(2.0 * 0.49 * 0.49) };
  const double expected_dist1{ std::sqrt(2.0) };

  EXPECT_EQ(result0, p0);
  EXPECT_DOUBLE_EQ(dist0, expected_dist0);

  EXPECT_EQ(result0p, p0);
  EXPECT_DOUBLE_EQ(dist0p, expected_dist0p);

  EXPECT_EQ(result1, p3);
  EXPECT_DOUBLE_EQ(dist1, expected_dist0p);
}
int main(int argc, char** argv)
{
  // ros::Time::init();
  ros::init(argc, argv, "gnn_test");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}