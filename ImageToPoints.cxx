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

// This program reads a binary image and converts the black pixels to points
// in a vtkPolyData. Drawing in an image is an easier way to create a data set
// for testing FindBoundary and BoundingPolygon than specifying 3D points
// manually.

// VTK
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

// ITK
#include <itkImage.h>
#include <itkImageFileReader.h>

// Custom
#include "Helpers.h"

int main(int argc, char *argv[])
{
  if(argc < 3)
    {
    std::cerr << "Required arguments: inputFileName.png outputFileName.vtp" << std::endl;
    return EXIT_FAILURE;
    }
    
  std::string inputFileName = argv[1];
  std::cout << "Reading " << inputFileName << std::endl;

  std::string outputFileName = argv[2];
  
  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFileName);
  reader->Update();
  
  std::vector<itk::Index<2> > pixelList = Helpers::BinaryImageToPixelList(reader->GetOutput());
  std::cout << "pixelList has " << pixelList.size() << " points." << std::endl;
  
  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
  Helpers::PixelListToPolyData(pixelList, polydata);
  std::cout << "polydata has " << polydata->GetNumberOfPoints() << " points." << std::endl;
  
  Helpers::WritePoints(polydata, outputFileName.c_str());
  
  return EXIT_SUCCESS;
}
