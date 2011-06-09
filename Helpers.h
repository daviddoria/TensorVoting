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

void WritePoints(vtkPolyData*, std::string filename);
void WritePathAsPolyLine(std::vector<unsigned int> order, vtkPolyData* graphPolyData, std::string filename);
void WritePathAsLines(std::vector<unsigned int> order, vtkPolyData* graphPolyData, std::string filename);

std::vector<unsigned int> GetShortestPath(Graph& g, Graph::vertex_descriptor start, Graph::vertex_descriptor end);
float GetShortestPathDistance(Graph& g, Graph::vertex_descriptor start, Graph::vertex_descriptor end);

float GetDistanceBetweenPoints(vtkPolyData*, vtkIdType a, vtkIdType b);

void OutputVector(std::vector<unsigned int> &v);

} // end namespace Helpers
#endif