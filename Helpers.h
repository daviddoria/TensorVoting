#ifndef HELPERS_H
#define HELPERS_H

#include "Types.h"

#include "itkImage.h"

#include <vector>

#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

namespace Helpers
{
//void CreateFullyConnectedGraph(vtkMutableUndirectedGraph* graph, unsigned int numberOfPoints);

std::vector<itk::Index<2> > BinaryImageToPixelList(ImageType::Pointer image);
void PixelListToPolyData(std::vector<itk::Index<2> > pixelList, vtkSmartPointer<vtkPolyData> polydata);
unsigned int FindKeyByValue(std::map <unsigned int, unsigned int> myMap, unsigned int value);
unsigned int CountFalse(std::vector<bool>);

std::vector<unsigned int> GetShortestPath(Graph& g, Graph::vertex_descriptor start, Graph::vertex_descriptor end);
float GetShortestPathDistance(Graph& g, Graph::vertex_descriptor start, Graph::vertex_descriptor end);

} // end namespace Helpers
#endif