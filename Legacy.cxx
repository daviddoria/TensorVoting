void VTKGraph()
{  
  std::cout << "There are " << graph->GetNumberOfVertices() << " vertices." << std::endl;
  std::cout << "There are " << graph->GetNumberOfEdges() << " edges." << std::endl;
  
  // Associate physical locations with the vertices
  graph->SetPoints(graphPolyData->GetPoints());

  // Convert the graph to a polydata
  vtkSmartPointer<vtkGraphToPolyData> graphToPolyData = 
    vtkSmartPointer<vtkGraphToPolyData>::New();
  graphToPolyData->SetInput(graph);
  graphToPolyData->Update();
  
  graphPolyData->ShallowCopy(graphToPolyData->GetOutput());
  
  std::cout << "There are " << graphToPolyData->GetOutput()->GetNumberOfPoints() << " points." << std::endl;
  
  vtkSmartPointer<vtkDijkstraGraphGeodesicPath> dijkstra = 
    vtkSmartPointer<vtkDijkstraGraphGeodesicPath>::New();
  dijkstra->SetInputConnection(graphToPolyData->GetOutputPort());
  dijkstra->SetStartVertex(0);
  dijkstra->SetEndVertex(0);
  dijkstra->Update();
  
  path->ShallowCopy(dijkstra->GetOutput());
}


std::vector<unsigned int> FindShortestPath(vtkPolyData* points)
{
  // Inputs: graphPolyData
  // Outputs: std::vector<unsigned int> path
  
  std::vector<unsigned int> roughOrder = RoughOrdering(points);
  std::cout << "Size of 'roughOrder': " << roughOrder.size() << std::endl;
  
  OutputPath(roughOrder, points, "rough.vtp");
  
  // Create a graph from the initial rough order
  Graph g;
  
  // Add all of the vertices from the original rough outline
  std::vector<vertex_descriptor> vertices;
  for(unsigned int i = 0; i < roughOrder.size(); ++i)
    {
    vertices.push_back(boost::add_vertex(g));
    }
  std::cout << "Size of 'vertices': " << vertices.size() << std::endl;
  // This technique doesn't allow the link between the first and last vertex to be modified. Need to use the "double-wrap" technique.
  
  // Add weighted edges
  for(unsigned int i = 0; i < roughOrder.size() - 1; ++i)
    {
    unsigned int currentVertexId = i;
    unsigned int nextVertexId = i+1;
  
    // In the case that we are on the last vertex, the destination is the first vertex (0)
    /*
    // We don't need to handle this specially because roughOrder is already a closed loop
    if(nextVertexId == roughOrder.size() - 1)
      {
      nextVertexId = 0;
      }
    */

    double currentPoint[3];
    double nextPoint[3];
    
    points->GetPoint(currentVertexId, currentPoint);
    points->GetPoint(nextVertexId, nextPoint);
    float distance = sqrt(vtkMath::Distance2BetweenPoints(currentPoint, nextPoint));

    EdgeWeightProperty weight(distance);
    boost::add_edge(vertices[currentVertexId], vertices[nextVertexId], weight, g);
    std::cout << "Added edge between " << currentVertexId << " and " << nextVertexId << std::endl;
    }
    
  // Add all other edges which pass the straightness test
  for()
    {
    float error = StraightnessError(roughOrder, points, unsigned int startId, unsigned int endId);
    }

  std::vector<unsigned int> order = Helpers::GetShortestPath(g, vertices[0], vertices[vertices.size()-1]);
  
  return order;
}
