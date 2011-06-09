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

/*
 By definition, a point on the boundary should have a large region in some
 direction where no other points exist. We test each point by considering a
 circle of radius R centered at the point. We determine the largest angular region of 
 the circle where no points exist. If this angular region is larger than a 
 threshold (we have chosen 70 degrees), the point is a boundary point.
*/

// STL
#include <algorithm>
#include <cmath>
#include <sstream>

// VTK
#include <vtkIdList.h>
#include <vtkKdTreePointLocator.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkXMLPolyDataWriter.h>

int main(int argc, char *argv[])
{
  // Verify arguments
  if(argc < 3)
    {
    std::cerr << "Required arguments: input.vtp output.vtp [radius] [angleThreshold]" << std::endl;
    return EXIT_FAILURE;
    }
    
  // Parse arguments
  std::string inputFileName = argv[1];
  
  std::string outputFileName = argv[2];
  
  float radius = 1.0;
  
  if(argc > 3)
    {
    std::stringstream ss;
    ss << argv[3];
    ss >> radius;
    }
    
  float angleThreshold = 70.;
  
  if(argc > 4)
    {
    std::stringstream ss;
    ss << argv[4];
    ss >> angleThreshold;
    }
    
  // Display arguments
  std::cout << "Input file: " << inputFileName << std::endl;
  std::cout << "Output file: " << outputFileName << std::endl;
  std::cout << "Radius: " << radius << std::endl;
  
  vtkSmartPointer<vtkXMLPolyDataReader> reader =
    vtkSmartPointer<vtkXMLPolyDataReader>::New();
  reader->SetFileName(inputFileName.c_str());
  reader->Update();
  
  // Create a tree
  vtkSmartPointer<vtkKdTreePointLocator> pointTree = 
    vtkSmartPointer<vtkKdTreePointLocator>::New();
  pointTree->SetDataSet(reader->GetOutput());
  pointTree->BuildLocator();
  
  vtkSmartPointer<vtkPoints> boundaryPoints =
    vtkSmartPointer<vtkPoints>::New();
  
  // For each point, perform the test
  for(vtkIdType pointId = 0; pointId < reader->GetOutput()->GetNumberOfPoints(); ++pointId)
    {
    // Get the current point
    double currentPoint[3];
    reader->GetOutput()->GetPoint(pointId, currentPoint);

    // Find the points within the specified radius of the current point
    vtkSmartPointer<vtkIdList> result = 
      vtkSmartPointer<vtkIdList>::New();
    
    pointTree->FindPointsWithinRadius(radius, currentPoint, result);

    if(result->GetNumberOfIds() == 0) // There were no points found within 'radius' of the current point
      {
      continue; // The point is an outlier, skip it
      }
      
    // Compute the angle from the current point to every point found within 'radius'
    std::vector<float> angles;
    
    for(vtkIdType i = 0; i < result->GetNumberOfIds(); i++)
      {
      // Get the ith neighbor of pointId
      vtkIdType neighborId = result->GetId(i);
      double neighbor[3];
      reader->GetOutput()->GetPoint(neighborId, neighbor);

      float angle = atan2(neighbor[0]-currentPoint[0], neighbor[1]-currentPoint[1]); // Get the angle from the currentPoint to p
      angles.push_back(angle);
      } // end neighbor loop
    
    // Sort the angles
    std::sort(angles.begin(), angles.end());
    
    // Append the smallest angle to the end of the list of angles for simplicity of the next loop.
    angles.push_back(angles[0]);
    
    // Determine if the point is a boundary point.
    // Compute the difference between adjacent angles. If the different is > 180, subtract from 360 to get the inner distance.
    // For example, if the two angles are 350 and 10, the angle between them should be 20, not 340.
    // If the point is a boundary point, add it to the output point set.
    for(unsigned int i = 0; i < angles.size() - 1; ++i) // This loop uses the [last-1]th element, hence the -1.
      {
      float difference = fabs(angles[i] - angles[i+1]);
      if(difference > 180.)
	{
	difference = 360. - difference;
	}
      if(difference > angleThreshold) // If the angle between two vectors is greater than the specified threshold, the point is a boundary point.
	{
	boundaryPoints->InsertNextPoint(currentPoint);
	}
      } // end angle difference loop

    } // end main point loop

  vtkSmartPointer<vtkPolyData> boundaryPolyData =
    vtkSmartPointer<vtkPolyData>::New();
  boundaryPolyData->SetPoints(boundaryPoints);
  
  vtkSmartPointer<vtkVertexGlyphFilter> glyphFilter =
    vtkSmartPointer<vtkVertexGlyphFilter>::New();
  glyphFilter->SetInputConnection(boundaryPolyData->GetProducerPort());
  glyphFilter->Update();
  
  vtkSmartPointer<vtkXMLPolyDataWriter> writer =
    vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName(outputFileName.c_str());
  writer->SetInputConnection(glyphFilter->GetOutputPort());
  writer->Write();
  
  return EXIT_SUCCESS;
}
