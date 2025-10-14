#pragma once

// #include "prx/utilities/defs.hpp"
// #include "prx/utilities/spaces/space.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <tensegrity_utils/assert.hpp>

namespace utils
{
// #define MAX_KK 2000
// #define INIT_NODE_SIZE 1000
// #define INIT_CAP_NEIGHBORS 200

template <typename Value, typename DistanceFunction>
class graph_nearest_neighbors_t;

// Assuming State has "-" operator and norm() function i.e from Eigen::Vector
template <typename State>
struct eucledian_metric_t
{
  double operator()(const State& a, const State& b) const
  {
    return (a - b).norm();
  }
};
/**
 * @brief <b> Proximity node for the graph-based distance metric.</b>
 *
 * Proximity node for the graph-based distance metric.
 *
 * @author Kostas Bekris, Edgar Granados
 */
template <typename Value, typename DistanceFunction>
class proximity_node_t
{
public:
  using Index = std::size_t;
  using Neighbors = std::vector<Index>;
  /**
   * @brief Constructor
   */
  proximity_node_t() : proximity_node_t(Value(), 200)
  {
  }

  proximity_node_t(const Value value, const int initial_max_neighbors)
    : _neighbors(), _added_index(0), _cap_neighbors(initial_max_neighbors), _added_to_metric(false), _value(value)
  {
  }

  virtual ~proximity_node_t()
  {
  }

  /**
   * Gets the position of the node in the data structure. Used for fast deletion.
   * @brief Gets the position of the node in the data structure.
   * @return The index value.
   */
  Index get_index() const
  {
    return _prox_index;
  }
  Index get_prox_index() const
  {
    return _prox_index;
  }

  /**
   * Returns the stored neighbors.
   * @brief Returns the stored neighbors.
   * @param nr_neigh Storage for the number of neighbors returned.
   * @return The neighbor indices.
   */
  const Neighbors& get_neighbors() const
  {
    return _neighbors;
  }

  Neighbors get_neighbors()
  {
    return _neighbors;
  }

  const Value& value() const
  {
    return _value;
  }

  Index value_id() const
  {
    return _value_id;
  }

protected:
  Index _added_index;
  /**
   * Adds a node index into this node's neighbor list.
   * @brief Adds a node index into this node's neighbor list.
   * @param node The index to add.
   */
  void add_neighbor(const Index node_id)
  {
    _neighbors.push_back(node_id);
  }

  /**
   * Deletes a node index from this node's neighbor list.
   * @brief Deletes a node index from this node's neighbor list.
   * @param node The index to delete.
   */
  void delete_neighbor(const Index node)
  {
    const auto it = std::find(_neighbors.begin(), _neighbors.end(), node);
    TENSEGRITY_ASSERT(it != _neighbors.end(), "Node not found");
    _neighbors.erase(it);
    // std::erase(_neighbors, node);
  }

  /**
   * Replaces a node index from this node's neighbor list.
   * @brief Replaces a node index from this node's neighbor list.
   * @param prev The index to look for.
   * @param new_index The index to replace with.
   */
  void replace_neighbor(Index prev, Index new_index)
  {
    const auto it = std::find(_neighbors.begin(), _neighbors.end(), prev);
    TENSEGRITY_ASSERT(it != _neighbors.end(), "Node not found");
    *it = new_index;
  }

  void remove_all_neighbors()
  {
    _added_index = 0;
  }

  /**
   * Sets the position of the node in the data structure. Used for fast deletion.
   * @brief Sets the position of the node in the data structure.
   * @param indx The index value.
   */
  void set_index(const Index& indx)
  {
    _prox_index = indx;
  }

  // void set_value(const Value value)
  // {
  //
  //   _value = value;
  //
  // }

  bool _added_to_metric;

  /**
   * @brief Index in the data structure. Serves as an identifier to other nodes.
   */
  Index _prox_index;

  /**
   * @brief The max number of neighbors.
   */
  Index _cap_neighbors;

  /**
   * @brief The current number of neighbors.
   */
  // Index _nr_neighbors;

  /**
   * @brief The neighbor list for this node.
   */
  Neighbors _neighbors;

  /**
   * @brief Value stored in this node
   */
  Value _value;

  /**
   * @brief Id associated to _value i.e. Will not be changed by the GNN.
   */
  Index _value_id;

  friend class graph_nearest_neighbors_t<Value, DistanceFunction>;
};

/**
 * A proximity structure based on graph literature. Each node maintains a list of neighbors.
 * When performing queries, the graph is traversed to determine other locally close nodes.
 * @brief <b> A proximity structure based on graph literature. </b>
 * @author Kostas Bekris, Edgar Granados
 */
template <typename Value, typename DistanceFunction>
class graph_nearest_neighbors_t
{
  using Index = std::size_t;
  using Neighbors = std::vector<Index>;

public:
  using Node = proximity_node_t<Value, DistanceFunction>;
  using NodePtr = std::shared_ptr<Node>;

  double node_distance(const NodePtr s1, const NodePtr s2)
  {
    return _distance_function(s1->value(), s2->value());
  }

  /**
   * @brief Constructor
   * @param state The first node to add to the structure.
   */
  graph_nearest_neighbors_t(DistanceFunction distFn, const int init_node_size = 1000, const int max_kk = 2000)
    : _distance_function(distFn)
    , _added_node_id(0)
    , _nodes()
    , _cap_nodes(init_node_size)
    , _second_nodes(max_kk)
    , _second_distances(max_kk)
    , _query_node(std::make_shared<Node>())
    , _max_kk(max_kk)
    , _preallocated_nodes()
    , _uniform_zero_one(0.0, std::nextafter(1.0, std::numeric_limits<double>::max()))
    , _curr_value_idx(0)

  {
  }

  graph_nearest_neighbors_t(const int init_node_size = 1000, const int max_kk = 2000)
    : graph_nearest_neighbors_t(DistanceFunction(), init_node_size, max_kk)
  {
    // nr_nodes = 0;
  }

  ~graph_nearest_neighbors_t()
  {
  }

  /**
   * Adds a node to the proximity structure
   * @brief Adds a node to the proximity structure
   * @param node The node to insert.
   */
  void add_node(const Value& value, Index value_id)
  {
    NodePtr graph_node{ get_graph_node(value, value_id) };
    const Index k{ percolation_threshold() };

    const Index new_k{ find_k_close(graph_node, _second_nodes, _second_distances, k) };

    // if (nr_nodes >= cap_nodes - 1)
    // {
    //   cap_nodes = 2 * cap_nodes;
    //   nodes = (proximity_node_t**)realloc(nodes, cap_nodes * sizeof(proximity_node_t*));
    // }
    graph_node->set_index(_nodes.size());

    _nodes.push_back(graph_node);

    // nr_nodes++;

    for (int i = 0; i < new_k; i++)
    {
      graph_node->add_neighbor(_second_nodes[i]->get_prox_index());
      _second_nodes[i]->add_neighbor(graph_node->get_prox_index());
    }
    graph_node->_added_to_metric = true;
  }
  void add_node(const Value& value)
  {
    add_node(value, _curr_value_idx);
    _curr_value_idx++;
  }

  void emplace(const Value& value, const Index value_id)
  {
    add_node(value, value_id);
  }
  void emplace(const Value& value)
  {
    add_node(value);
  }

  /**
   * @brief Removes a node from the structure.
   * @param node
   */
  void remove_node(NodePtr graph_node)
  {
    // Index nr_neighbors;
    const Neighbors neighbors{ graph_node->get_neighbors() };
    for (auto n_idx : neighbors)
    {
      _nodes[n_idx]->delete_neighbor(graph_node->get_prox_index());
    }
    graph_node->remove_all_neighbors();

    const Index index{ graph_node->get_prox_index() };
    const std::size_t& nr_nodes{ _nodes.size() };

    if (index < nr_nodes - 1)
    {
      // Swap the node to remove with the last one, update the last one and its neighbors. Remove the last one
      std::swap(_nodes[index], _nodes[nr_nodes - 1]);
      // _nodes[index] = _nodes[nr_nodes - 1];
      _nodes[index]->set_index(index);

      const Neighbors curr_neighbors{ _nodes[index]->get_neighbors() };
      for (auto n_idx : curr_neighbors)
      {
        _nodes[n_idx]->replace_neighbor(nr_nodes - 1, index);
      }
    }
    _nodes.pop_back();
    graph_node->_added_to_metric = false;

    // if (nr_nodes < (_cap_nodes - 1) / 2)
    // {
    //   _cap_nodes *= 0.5;
    //   _nodes.resize(_cap_nodes);
    // }
  }

  void clear()
  {
    for (auto idx : _nodes)
    {
      _nodes[idx]->_neighbors.clear();
    }
    _nodes.clear();
  }

  /**
   * Get function for number of nodes
   * @return Number of nodes in graph
   */
  std::size_t size()
  {
    return _nodes.size();
  }

  std::size_t get_nr_nodes()
  {
    return _nodes.size();
  }

  void single_query(const Value& value, double& distance, NodePtr& node)
  {
    _query_node->_value = value;
    // double distance;
    node = find_closest(_query_node, distance);
  }
  Value single_query(const Value& value, double& distance)
  {
    NodePtr node;
    single_query(value, distance, node);
    return node->value();
  }
  Value single_query(const Value& value)
  {
    NodePtr node;
    double distance;

    single_query(value, distance, node);

    return node->value();
  }

  std::vector<NodePtr> multi_query(const Value& value, const int k)
  {
    _query_node->_value = value;
    find_k_close(_query_node, _second_nodes, _second_distances, k);
    std::vector<NodePtr> ret(_second_nodes.begin(), _second_nodes.end());
    return std::move(ret);
  }

  void radius_and_closest_query(std::vector<NodePtr>& out_nodes, std::vector<double>& out_distances, const Value& value,
                                const double rad)
  {
    out_nodes.clear();
    out_distances.clear();
    _query_node->_value = value;

    const int new_k{ find_delta_close_and_closest(_query_node, _second_nodes, _second_distances, rad) };

    if (new_k > 0)
    {
      out_nodes.insert(out_nodes.end(), _second_nodes.begin(), _second_nodes.begin() + new_k);
      out_distances.insert(out_distances.end(), _second_distances.begin(), _second_distances.begin() + new_k);
    }
  }

  std::vector<NodePtr> radius_and_closest_query(const Value& value, const double rad)
  {
    std::vector<NodePtr> out_nodes;
    std::vector<double> out_distances;
    radius_and_closest_query(out_nodes, out_distances, value, rad);
    return out_nodes;
    // _query_node->_value = value;
    // int new_k{ find_delta_close_and_closest(_query_node, _second_nodes, _second_distances, rad) };
    // if (new_k > 0)
    // {
    //   std::vector<NodePtr> ret(_second_nodes.begin(), _second_nodes.begin() + new_k);
    //   return std::move(ret);
    // }
    // else
    // {
    //   return {};
    // }
  }

  NodePtr operator[](const std::size_t idx)
  {
    return _nodes[idx];
  }

  NodePtr front()
  {
    return _nodes.front();
  }

  // typename std::vector<NodePtr>::Iterator begin()
  // {
  //   return _nodes.begin();
  // }

  // typename std::vector<NodePtr>::Iterator end()
  // {
  //   return _nodes.end();
  // }

protected:
  NodePtr get_graph_node(const Value& value, const Index& value_id)
  {
    if (_preallocated_nodes.size() == 0)
    {
      for (int i = 0; i < _max_kk; ++i)
      {
        _preallocated_nodes.push_back(std::make_shared<Node>());
      }
    }

    NodePtr node{ _preallocated_nodes.back() };
    _preallocated_nodes.pop_back();

    node->_value = value;
    node->_value_id = value_id;

    return node;
  }

  DistanceFunction _distance_function;

  /**
   * Returns the closest node in the data structure.
   * @brief Returns the closest node in the data structure.
   * @param state The query point.
   * @param distance The resulting distance between the closest point and the query point.
   * @return The closest point.
   */
  NodePtr find_closest(NodePtr state, double& distance)
  {
    Index min_index{ std::numeric_limits<Index>::max() };
    return basic_closest_search(state, distance, min_index);
  }

  /**
   * Find the k closest nodes to the query point.
   * @brief Find the k closest nodes to the query point.
   * @param state The query state.
   * @param close_nodes The returned close nodes.
   * @param distances The corresponding distances to the query point.
   * @param k The number to return.
   * @return The number of nodes actually returned.
   */
  Index find_k_close(NodePtr state, std::vector<NodePtr>& close_nodes, std::vector<double>& distances, Index k)
  {
    const Index nr_nodes{ _nodes.size() };
    if (nr_nodes == 0)
    {
      return 0;
    }

    if (k > _max_kk)
    {
      k = _max_kk;
    }
    else if (k >= nr_nodes)  // k is greater than the stored nodes ==> return everything
    {
      for (int i = 0; i < nr_nodes; i++)
      {
        close_nodes[i] = _nodes[i];
        distances[i] = node_distance(_nodes[i], state);
      }
      sort_proximity_nodes(close_nodes, distances, 0, nr_nodes - 1);
      return nr_nodes;
    }

    clear_added();

    Index min_index = -1;
    distances.emplace_back();
    close_nodes.push_back(basic_closest_search(state, distances[0], min_index));
    _nodes[min_index]->_added_index = _added_node_id;

    min_index = 0;
    Index nr_elements = 1;
    // double max_distance = distances[0];

    /* Find the neighbors of the closest node if they are not already in the set of k-closest nodes.
    If the distance to any of the neighbors is less than the distance to the k-th closest element,
    then replace the last element with the neighbor and resort the list. In order to decide the next
    node to pivot about, it is either the next node on the list of k-closest
    */
    do
    {
      // long unsigned nr_neighbors;
      Neighbors neighbors{ _nodes[close_nodes[min_index]->get_prox_index()]->get_neighbors() };
      Index lowest_replacement{ nr_elements };

      for (int j = 0; j < neighbors.size(); j++)
      {
        NodePtr the_neighbor = _nodes[neighbors[j]];
        if (does_node_exist(the_neighbor) == false)
        {
          the_neighbor->_added_index = _added_node_id;

          const double distance{ node_distance(the_neighbor, state) };
          bool to_resort = false;
          if (nr_elements < k)
          {
            close_nodes[nr_elements] = the_neighbor;
            distances[nr_elements] = distance;
            nr_elements++;
            to_resort = true;
          }
          else if (distance < distances[k - 1])
          {
            close_nodes[k - 1] = the_neighbor;
            distances[k - 1] = distance;
            to_resort = true;
          }

          if (to_resort)
          {
            int test = resort_proximity_nodes(close_nodes, distances, nr_elements - 1);
            lowest_replacement = (test < lowest_replacement ? test : lowest_replacement);
          }
        }
      }

      /* In order to decide the next node to pivot about,
      it is either the next node on the list of k-closest (min_index)
      or one of the new neighbors in the case that it is closer than nodes already checked.
      */
      if (min_index < lowest_replacement)
      {
        min_index++;
      }
      else
      {
        min_index = lowest_replacement;
      }
    } while (min_index < nr_elements);

    return nr_elements;
  }

  /**
   * Find all nodes within a radius and the closest node.
   * @brief Find all nodes within a radius and the closest node.
   * @param state The query state.
   * @param close_nodes The returned close nodes.
   * @param distances The corresponding distances to the query point.
   * @param delta The radius to search within.
   * @return The number of nodes returned.
   */
  int find_delta_close_and_closest(NodePtr state, std::vector<NodePtr>& close_nodes, std::vector<double>& distances,
                                   double delta)
  {
    const std::size_t nr_nodes{ _nodes.size() };
    if (nr_nodes == 0)
    {
      return 0;
    }

    clear_added();

    Index min_index = -1;
    close_nodes[0] = basic_closest_search(state, distances[0], min_index);

    if (distances[0] > delta)
    {
      return 1;
    }

    _nodes[min_index]->_added_index = _added_node_id;

    int nr_points = 1;
    for (int counter = 0; counter < nr_points; counter++)
    {
      // long unsigned nr_neighbors;
      const Neighbors neighbors{ close_nodes[counter]->get_neighbors() };
      for (int j = 0; j < neighbors.size(); j++)
      {
        NodePtr the_neighbor = _nodes[neighbors[j]];
        if (does_node_exist(the_neighbor) == false)
        {
          the_neighbor->_added_index = _added_node_id;
          double distance = node_distance(the_neighbor, state);
          if (distance < delta && nr_points < _max_kk)
          {
            close_nodes[nr_points] = the_neighbor;
            distances[nr_points] = distance;
            nr_points++;
          }
        }
      }
    }

    if (nr_points > 0)
    {
      sort_proximity_nodes(close_nodes, distances, 0, nr_points - 1);
    }

    return nr_points;
  }

  /**
   * Find all nodes within a radius.
   * @brief Find all nodes within a radius.
   * @param state The query state.
   * @param close_nodes The returned close nodes.
   * @param distances The corresponding distances to the query point.
   * @param delta The radius to search within.
   * @return The number of nodes returned.
   */
  int find_delta_close(NodePtr state, std::vector<NodePtr> close_nodes, std::vector<double> distances,
                       const double delta)
  {
    const std::size_t& nr_nodes{ _nodes.size() };
    if (nr_nodes == 0)
    {
      return 0;
    }

    clear_added();

    Index min_index = -1;
    close_nodes[0] = basic_closest_search(state, distances[0], min_index);

    if (distances[0] > delta)
    {
      return 0;
    }

    _nodes[min_index]->added_index = _added_node_id;

    int nr_points = 1;
    for (int counter = 0; counter < nr_points; counter++)
    {
      // long unsigned nr_neighbors;
      const Neighbors neighbors{ close_nodes[counter]->get_neighbors() };
      for (int j = 0; j < neighbors.size(); j++)
      {
        NodePtr the_neighbor = _nodes[neighbors[j]];
        if (does_node_exist(the_neighbor) == false)
        {
          the_neighbor->added_index = _added_node_id;
          double distance = node_distance(the_neighbor, state);
          if (distance < delta && nr_points < _max_kk)
          {
            close_nodes[nr_points] = the_neighbor;
            distances[nr_points] = distance;
            nr_points++;
          }
        }
      }
    }

    if (nr_points > 0)
    {
      sort_proximity_nodes(close_nodes, distances, 0, nr_points - 1);
    }

    return nr_points;
  }

  Index random_uniform(const Index min, const Index max)
  {
    const double val{ _uniform_zero_one(_global_generator) };
    const Index r{ static_cast<Index>(val * (max - min) + min) };
    return r;
  }

  /**
   * Determine the number of nodes to sample for initial populations in queries.
   * @brief Determine the number of nodes to sample for initial populations in queries.
   * @return The number of random nodes to initially select.
   */
  inline Index sampling_function()
  {
    if (_nodes.size() < 200)
      return _nodes.size();
    else
      return 200;
  }

  /**
   * Given the number of nodes, get the number of neighbors required for connectivity (in the limit).
   * @brief Given the number of nodes, get the number of neighbors required for connectivity (in the limit).
   * @return
   */
  inline Index percolation_threshold()
  {
    if (_nodes.size() > 12)
      return (2.0 * std::log(_nodes.size()));
    else
      return _nodes.size();
  }

  /**
   * Sorts a list of proximity_node_t's. Performed using a quick sort operation.
   * @param close_nodes The list to sort.
   * @param distances The distances that determine the ordering.
   * @param low The lower index.
   * @param high The upper index.
   */
  void sort_proximity_nodes(std::vector<NodePtr>& close_nodes, std::vector<double>& distances, const int low,
                            const int high)
  {
    {
      if (low < high)
      {
        int left, right, pivot;
        double pivot_distance = distances[low];
        NodePtr pivot_node = close_nodes[low];

        double temp;
        NodePtr temp_node;

        pivot = left = low;
        right = high;
        while (left < right)
        {
          while (left <= high && distances[left] <= pivot_distance)
          {
            left++;
          }
          while (distances[right] > pivot_distance)
          {
            right--;
          }
          if (left < right)
          {
            temp = distances[left];
            distances[left] = distances[right];
            distances[right] = temp;

            temp_node = close_nodes[left];
            close_nodes[left] = close_nodes[right];
            close_nodes[right] = temp_node;
          }
        }
        distances[low] = distances[right];
        distances[right] = pivot_distance;

        close_nodes[low] = close_nodes[right];
        close_nodes[right] = pivot_node;

        sort_proximity_nodes(close_nodes, distances, low, right - 1);
        sort_proximity_nodes(close_nodes, distances, right + 1, high);
      }
    }
  }

  /**
   * Performs sorting over a list of nodes. Assumes all nodes before index are sorted.
   * @param close_nodes The list to sort.
   * @param distances The distances that determine the ordering.
   * @param index The index to start from.
   */
  int resort_proximity_nodes(std::vector<NodePtr>& close_nodes, std::vector<double>& distances, int index_)
  {
    double temp;
    NodePtr temp_node;

    while (index_ > 0 && distances[index_] < distances[index_ - 1])
    {
      temp = distances[index_];
      distances[index_] = distances[index_ - 1];
      distances[index_ - 1] = temp;

      temp_node = close_nodes[index_];
      close_nodes[index_] = close_nodes[index_ - 1];
      close_nodes[index_ - 1] = temp_node;

      index_--;
    }
    return index_;
  }

  /**
   * Helper function for determining existance in a list.
   * @brief Helper function for determining existance in a list.
   * @param query_node The node to search for.
   */
  bool does_node_exist(NodePtr query_node)
  {
    return query_node->_added_index == _added_node_id;
  }

  /**
   * The basic search process for finding the closest node to the query state.
   * @brief Find the closest node to the query state.
   * @param state The query state.
   * @param distance The corresponding distance to the query point.
   * @param node_index The index of the returned node.
   * @return The closest node.
   */
  NodePtr basic_closest_search(NodePtr state, double& the_distance, Index& the_index)
  {
    if (_nodes.size() == 0)
    {
      return nullptr;
    }

    const std::size_t nr_samples{ sampling_function() };
    double min_distance{ std::numeric_limits<double>::max() };

    Index min_index = -1;
    bool node_found{ false };
    for (int i = 0; i < nr_samples; i++)
    {
      const Index index_{ random_uniform(0, _nodes.size()) };  // rand() % nr_nodes;
      const double distance{ node_distance(_nodes[index_], state) };
      if (distance < min_distance)
      {
        min_distance = distance;
        min_index = index_;
        node_found = true;
      }
    }

    TENSEGRITY_ASSERT(node_found, "Error: GNN couldn't find a close neighbor");
    Index old_min_index{ min_index };
    do
    {
      old_min_index = min_index;
      // long unsigned nr_neighbors;
      const Neighbors neighbors{ _nodes[min_index]->get_neighbors() };
      // for (int j = 0; j < nr_neighbors; j++)
      for (auto n_idx : neighbors)
      {
        const double distance{ node_distance(_nodes[n_idx], state) };
        if (distance < min_distance)
        {
          min_distance = distance;
          min_index = n_idx;
        }
      }
    } while (old_min_index != min_index);

    the_distance = min_distance;
    the_index = min_index;
    return _nodes[min_index];
  }

  void clear_added()
  {
    _added_node_id++;
  }

  /**
   * @brief The nodes being stored.
   */
  std::vector<NodePtr> _nodes;

  /**
   * @brief The current number of nodes being stored.
   */
  // Index nr_nodes;

  /**
   * @brief The maximum number of nodes that can be stored.
   */
  Index _cap_nodes;

  /**
   * @brief Temporary storage for query functions.
   */
  std::vector<NodePtr> _second_nodes;

  /**
   * @brief Temporary storage for query functions.
   */
  std::vector<double> _second_distances;

  // std::vector<proximity_node_t*> added_nodes;

  Index _added_node_id;

  NodePtr _query_node;

  int _max_kk;
  std::vector<NodePtr> _preallocated_nodes;
  // abstract_node_t* query_node;
  Index _curr_value_idx;

  std::mt19937_64 _global_generator;
  std::uniform_real_distribution<double> _uniform_zero_one;
};
}  // namespace utils