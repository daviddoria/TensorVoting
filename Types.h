#ifndef TYPES_H
#define TYPES_H

// ITK
#include "itkImage.h"

typedef itk::Image<unsigned char, 2> ImageType;

// Boost
#include <boost/graph/adjacency_list.hpp>

// Graph properties
typedef float WeightType;
typedef boost::property<boost::edge_weight_t, WeightType> EdgeWeightProperty;

// The property to store the id of the point at which a vertex resides
struct VertexProperty
{
  int PointId;
};

// Directed graph with vertex and edge properties. Vertex type selector must be vecS because we rely on using integer indexing
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
			      VertexProperty, EdgeWeightProperty> Graph;

// Directed graph with only edge properties
//typedef boost::adjacency_list < boost::listS, boost::vecS, boost::directedS,
  //boost::no_property, EdgeWeightProperty > Graph;

// Directed graph with only edge properties
//typedef boost::adjacency_list < boost::listS, boost::vecS, boost::directedS,
  //boost::no_property, EdgeWeightProperty > Graph;
  
// Undirected graph
//typedef boost::adjacency_list < boost::listS, boost::vecS, boost::undirectedS,
  //boost::no_property, EdgeWeightProperty > Graph;

#endif