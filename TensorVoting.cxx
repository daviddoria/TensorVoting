/*=========================================================================
 *
 *  Copyright David Doria 2011 daviddoria@gmail.com
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// This algorithm is based on "Using Aerial Lidar Data to Segment And Model Buildings" by Oliver Wang

// VTK
#include <vtkActor.h>
#include <vtkPolyLine.h>
#include <vtkCellArray.h>
#include <vtkGraphToPolyData.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkKdTreePointLocator.h>
#include <vtkLine.h>
#include <vtkMath.h>
#include <vtkProperty.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkXMLPolyDataWriter.h>

// ITK
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegionConstIterator.h>

// Custom
#include "Helpers.h"
#include "Types.h"

// Boost
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/adjacency_list.hpp>

std::vector<unsigned int> RoughOrdering(vtkPolyData* points);

void Visualize(vtkPolyData* graph, vtkPolyData* path);

std::vector<unsigned int> OutlineApproximation(vtkPolyData* points, float straightnessErrorTolerance);

void WriteGraph(Graph& g, vtkPolyData* polydata, std::string filename);

float StraightnessError(Graph g, vtkPolyData* points, unsigned int start, unsigned int end);

std::vector<unsigned int> GetShortestClosedLoop(Graph& g);

int main(int argc, char *argv[])
{
  if(argc < 2)
    {
    std::cerr << "Required arguments: filename [straightnessErrorTolerance]" << std::endl;
    return EXIT_FAILURE;
    }
    
  std::string fileName = argv[1];
  std::cout << "Reading " << fileName << std::endl;
  
  float straightnessErrorTolerance = 1.0;
  if(argc == 3)
  {
    std::stringstream ss;
    ss << argv[2];
    ss >> straightnessErrorTolerance;
  }
  
  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(fileName);
  reader->Update();
  
  std::vector<itk::Index<2> > pixelList = Helpers::BinaryImageToPixelList(reader->GetOutput());
  std::cout << "pixelList has " << pixelList.size() << " points." << std::endl;
  
  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
  Helpers::PixelListToPolyData(pixelList, polydata);
  std::cout << "polydata has " << polydata->GetNumberOfPoints() << " points." << std::endl;
  
  Helpers::WritePoints(polydata, "points.vtp");
  
  std::vector<unsigned int> shortestPath = OutlineApproximation(polydata, straightnessErrorTolerance);
  std::cout << "shortestPath has " << shortestPath.size() << " points." << std::endl;
  
  std::cout << "shortestPath:" << std::endl;
  Helpers::OutputVector(shortestPath);

  Helpers::WritePathAsLines(shortestPath, polydata, "OutlineApproximation.vtp");
  //WriteGraph(g, polydata, "OutlineApproximation.vtp");
  
  //Visualize(polydata, path);
 
  return EXIT_SUCCESS;
}

std::vector<unsigned int> RoughOrdering(vtkPolyData* allPoints)
{
  // This function starts at point index 0, and then greedily finds the closest point which has not yet been
  // found. This should hopefully make a reasonable outline of the points.
  // OLD: The loop is closed - that is the first and last elements of the output vector are the same vertex
  // NEW: The loop is open - that is, if you connect all vertices i to i+1, you will not reach the beginning (you would need to additionally
  //      connect the last point to the first point.
  
  std::cout << "The input has " << allPoints->GetNumberOfPoints() << " points." << std::endl;
  
  // Track if each point has been found yet or not
  std::vector<bool> used(allPoints->GetNumberOfPoints(), false);
    
  // Start at point id 0
  unsigned int currentPointId = 0;
  
  // Track the order in which the points were found
  std::vector<unsigned int> pointOrder;
  
  // Loop until all points have been used
  while(Helpers::CountFalse(used) > 1)
    {
    // Mark the current point as used
    //std::cout << "marking " << currentPointId << " as used." << std::endl;
    used[currentPointId] = true;
  
    //std::cout << "There are " << Helpers::CountFalse(used) << " points remaining." << std::endl;
  
    // Save the current point in the point order tracking
    pointOrder.push_back(currentPointId);
  
    // Create a vtkPoints of the remaining points (one of which will be the closest to the current point)
    vtkSmartPointer<vtkPoints> remainingPoints = 
      vtkSmartPointer<vtkPoints>::New();
    
    // Save the mapping from the unused points id to the original point ids
    std::map <unsigned int, unsigned int> idMap;

    // Keep track of which unused point we are on
    unsigned int unusedPointId = 0; 
    for(unsigned int i = 0; i < allPoints->GetNumberOfPoints(); ++i)
      {
      if(used[i])
	{
	continue;
	}
	
      // Store the id mapping in the map
      idMap.insert(std::pair<unsigned int, unsigned int>(unusedPointId, i));
      //std::cout << "adding map entry from " << unusedPointId << " to " << i << std::endl;
      unusedPointId++;
    
      // Add the point to the remainingPoints
      double p[3];
      allPoints->GetPoint(i, p);
      remainingPoints->InsertNextPoint(p);
      }
    
    //std::cout << "remainingPoints has " << remainingPoints->GetNumberOfPoints() << " points." << std::endl;
    
    vtkSmartPointer<vtkPolyData> remainingPolyData = 
      vtkSmartPointer<vtkPolyData>::New();
    remainingPolyData->SetPoints(remainingPoints);
      
    // Create the tree
    vtkSmartPointer<vtkKdTreePointLocator> kDTree = 
      vtkSmartPointer<vtkKdTreePointLocator>::New();
    kDTree->SetDataSet(remainingPolyData);
    kDTree->BuildLocator();
    
    double currentPoint[3];
    allPoints->GetPoint(currentPointId, currentPoint);
    //std::cout << "CurrentPoint is " << currentPoint[0] << " " << currentPoint[1] << " " << currentPoint[2] << std::endl;
    
    // Find the closest points to TestPoint
    unsigned int closestRemainingPointId = kDTree->FindClosestPoint(currentPoint);
    //std::cout << "closestRemainingPointId is " << closestRemainingPointId << std::endl;
    
    // Set the current point to the closest point
    //std::cout << "Changing currentPointId from " << currentPointId << " to " << idMap[closestRemainingPointId] << std::endl;
    currentPointId = idMap[closestRemainingPointId];
    }
    
  // Add the last point to the list, as there is no decision to be made about which one is closest!
  pointOrder.push_back(currentPointId);
  
  // Add the first point to the list again to close the loop
  //pointOrder.push_back(pointOrder[0]);

  return pointOrder;
}

std::vector<unsigned int> OutlineApproximation(vtkPolyData* points, float straightnessErrorTolerance)
{
  // Inputs: graphPolyData
  // Outputs: std::vector<unsigned int> path
  
  std::vector<unsigned int> roughOrder = RoughOrdering(points);
  //std::cout << "Size of 'roughOrder': " << roughOrder.size() << std::endl;
  
  //std::cout << "Rough order: " << std::endl;
  //Helpers::OutputVector(roughOrder);
  
  Helpers::WritePathAsLines(roughOrder, points, "rough.vtp");
  
  // Create a graph from the initial rough order
  Graph g;
  
  // We must add vertices from the original rough outline twice (the reason for this is explained later)!
  
  for(unsigned int loopCounter = 0; loopCounter < 2; ++loopCounter)
    {
    for(unsigned int i = 0; i < roughOrder.size(); ++i)
      {
      Graph::vertex_descriptor v = boost::add_vertex(g);
      g[v].PointId = roughOrder[i];
      }
    }
  
  //std::cout << "Size of 'vertices': " << boost::num_vertices(g) << std::endl;
    
  // Add weighted edges between adjacent vertices
  for(unsigned int i = 0; i < boost::num_vertices(g) - 1; ++i)
    {
    unsigned int currentVertexId = i;
    unsigned int nextVertexId = i+1;
  
    float distance = Helpers::GetDistanceBetweenPoints(points, g[currentVertexId].PointId, g[nextVertexId].PointId);

    EdgeWeightProperty weight(distance);
    boost::add_edge(currentVertexId, nextVertexId, weight, g);
    //std::cout << "Added edge between vertices " << currentVertexId << " and " << nextVertexId
	//      << " which corresponds to points " << g[currentVertexId].PointId << " and " << g[nextVertexId].PointId << std::endl;
    }
  
  // Close the second loop
  boost::add_edge(boost::num_vertices(g) - 1, 0, 
		  Helpers::GetDistanceBetweenPoints(points, g[boost::num_vertices(g) - 1].PointId, g[0].PointId), g);
  
  WriteGraph(g, points, "OutlineGraph.vtp");
  
  // Add all other edges which pass the straightness test
  
  for(unsigned int start = 0; start < boost::num_vertices(g); ++start)
    {
    for(unsigned int end = start+1; end < boost::num_vertices(g); ++end)
      {
      float error = StraightnessError(g, points, start, end);
      if(error < straightnessErrorTolerance)
	{
	// Add an edge between start and end
	double startPoint[3];
	double endPoint[3];
	
	points->GetPoint(g[start].PointId, startPoint);
	points->GetPoint(g[end].PointId, endPoint);
	float distance = sqrt(vtkMath::Distance2BetweenPoints(startPoint, endPoint));

	EdgeWeightProperty weight(distance);
	boost::add_edge(start, end, weight, g);
	//std::cout << "Added edge between " << g[start].PointId << " and " << g[end].PointId << std::endl;
	}
      }
    }

  WriteGraph(g, points, "StraigtnessGraph.vtp");

  std::vector<unsigned int> approximateOutline = GetShortestClosedLoop(g);
  
  return approximateOutline;
}

void Visualize(vtkPolyData* graph, vtkPolyData* path)
{
  std::cout << "GraphPolyData has " << graph->GetNumberOfCells() << " cells." << std::endl;
  
  // Create a mapper and actor
  vtkSmartPointer<vtkPolyDataMapper> pathMapper = 
    vtkSmartPointer<vtkPolyDataMapper>::New();
  pathMapper->SetInputConnection(path->GetProducerPort());
 
  vtkSmartPointer<vtkActor> pathActor = 
    vtkSmartPointer<vtkActor>::New();
  pathActor->SetMapper(pathMapper);
  pathActor->GetProperty()->SetColor(1,0,0); // Red
  pathActor->GetProperty()->SetLineWidth(4);
    
  // Create a mapper and actor
  vtkSmartPointer<vtkPolyDataMapper> mapper = 
    vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(graph->GetProducerPort());
 
  vtkSmartPointer<vtkActor> actor = 
    vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
 
  // Create a renderer, render window, and interactor
  vtkSmartPointer<vtkRenderer> renderer = 
    vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow = 
    vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = 
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);
 
  // Add the actor to the scene
  renderer->AddActor(actor);
  renderer->AddActor(pathActor);
  renderer->SetBackground(.3, .6, .3); // Background color green
 
  
  vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = 
    vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
  renderWindowInteractor->SetInteractorStyle( style );
 
  // Render and interact
  renderWindow->Render();
  renderWindowInteractor->Start();
}

float StraightnessError(Graph g, vtkPolyData* points, unsigned int startId, unsigned int endId)
{
  // This function finds the sum of the distances from each point between vertices[startId] and vertices[endId]
  // to the line formed between order[start] and order[end].
  
  double startPoint[3];
  points->GetPoint(g[startId].PointId, startPoint);
  
  double endPoint[3];
  points->GetPoint(g[endId].PointId, endPoint);
  
  float totalDistance = 0.;
  
  unsigned int numberOfPoints = 0;
  for(unsigned int i = startId+1; i < endId; ++i)
    {
    double currentPoint[3];
    points->GetPoint(g[i].PointId, currentPoint);
  
    totalDistance += vtkLine::DistanceToLine(currentPoint, startPoint, endPoint);
    numberOfPoints++;
    }
  
  //return distance; // sum, as in original paper
  return totalDistance/static_cast<float>(numberOfPoints); // average, makes more sense
}


std::vector<unsigned int> GetShortestClosedLoop(Graph& g)
{
  unsigned int numberOfPoints = boost::num_vertices(g)/2;
  //std::cout << "boost::num_vertices(g)/2 = " << numberOfPoints << std::endl;
  
  float shortestPathDistance = std::numeric_limits<float>::max();
  std::vector<unsigned int> shortestPath;
  
  for(unsigned int i = 0; i < numberOfPoints; ++i)
    {
    float distance = Helpers::GetShortestPathDistance(g, i, i + numberOfPoints);
    std::cout << "Distance between " << i << " and " << i + numberOfPoints << " is " << distance << std::endl;
  
    if(distance < shortestPathDistance)
      {
      shortestPath = Helpers::GetShortestPath(g, i, i + numberOfPoints);
      }
    }
    
  return shortestPath;
}

void WriteGraph(Graph& g, vtkPolyData* points, std::string filename)
{
  vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
    
  typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
  IndexMap index = get(boost::vertex_index, g);
  
  typedef boost::graph_traits<Graph>::edge_iterator edge_iter;
  std::pair<edge_iter, edge_iter> edgePair;
  for(edgePair = boost::edges(g); edgePair.first != edgePair.second; ++edgePair.first)
    {
    vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
    //unsigned int source = index[boost::source(*edgePair.first, g)];
    //unsigned int target = index[boost::target(*edgePair.first, g)];
    
    unsigned int sourcePointId = g[boost::source(*edgePair.first, g)].PointId;
    unsigned int targetPointId = g[boost::target(*edgePair.first, g)].PointId;

    line->GetPointIds()->SetId(0,sourcePointId);
    line->GetPointIds()->SetId(1,targetPointId);
    lines->InsertNextCell(line);
    
    //std::cout << "Adding line between " << sourcePointId << " and " << targetPointId << std::endl;
    }
  std::cout << std::endl;
  
  // Create a polydata to store everything in
  vtkSmartPointer<vtkPolyData> polyData = 
    vtkSmartPointer<vtkPolyData>::New();
  
  // Add the points to the dataset
  polyData->SetPoints(points->GetPoints());
  
  // Add the lines to the dataset
  polyData->SetLines(lines);
  
  vtkSmartPointer<vtkXMLPolyDataWriter> writer =
    vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName(filename.c_str());
  writer->SetInput(polyData);
  writer->Write();
   
}
