#ifndef HELPERS_H
#define HELPERS_H

#include "Types.h"

#include "itkImage.h"

#include <vector>

namespace Helpers
{

  template <typename T>
  typename T::Pointer readImage(const std::string& filename)
  {
    std::cout << "Reading " << filename << std::endl;
    typedef typename itk::ImageFileReader<T> ReaderType;
    typename ReaderType::Pointer reader = ReaderType::New();

    ReaderType::GlobalWarningDisplayOff();
    reader->SetFileName(filename);
    try
    {
      reader->Update();
    }
    catch(itk::ExceptionObject &err)
    {
      std::cout << "ExceptionObject caught!" <<std::endl;
      std::cout << err << std::endl;
      //return EXIT_FAILURE;
    }
    std::cout << "Done." << std::endl;
    return reader->GetOutput();
  }




  template <typename T>
  int writeImage(typename T::Pointer im, const std::string& filename)
  {
    std::cout << "Writing " << filename << std::endl;
    typedef typename itk::ImageFileWriter<T> WriterType;

    typename WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(im);
    try
    {
      writer->Update();
    }
    catch(itk::ExceptionObject &err)
    {
      std::cout << "ExceptionObject caught!" <<std::endl;
      std::cout << err << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Done." << std::endl;
    return EXIT_SUCCESS;
  }

} // end namespace Helpers
#endif