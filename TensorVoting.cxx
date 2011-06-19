/* Copyright 2010 Brad King
Distributed under the Boost Software License, Version 1.0.
(See accompanying file rtvl_license_1_0.txt or copy at
http://www.boost.org/LICENSE_1_0.txt) */

// Run with:
// TensorVoting <input filename> <voting field parameter (5-10) > < dense voting field range (5-10)> < Output file name >
// The "foreground" pixels in the input image must be white, while the background pixels must be black

// ITK
#include "itkArray.h"
#include "itkListSample.h"
#include "itkVector.h"
#include "itkKdTreeGenerator.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkNeighborhood.h"
#include "itkNeighborhoodIterator.h"
#include "itkDerivativeImageFilter.h"
#include "itkZeroCrossingImageFilter.h"

// VXL
#include <rtvl/rtvl_tensor.hxx>
#include <rtvl/rtvl_vote.hxx>
#include <rtvl/rtvl_votee.hxx>
#include <rtvl/rtvl_voter.hxx>
#include <rtvl/rtvl_weight_original.hxx>
#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_matrix_fixed.h>
#include <vcl_vector.h>
#include <vcl_iostream.h>
#include <vcl_fstream.h>

// STL
#include <cmath>

// Custom
#include "Helpers.h"


int main(int argc, char **argv)
{

  //For Saliency Map
  typedef double InputPixelType;
  typedef itk::Image<InputPixelType,2> SaliencyMap;
  typedef itk::ImageRegionConstIterator<SaliencyMap> IteratorType;
  typedef itk::ConstNeighborhoodIterator< SaliencyMap > NeighborhoodIteratorType;

  typedef unsigned char OutputPixelType;
  typedef itk::Image<OutputPixelType,2> OutputMap;
  typedef itk::ImageFileWriter< OutputMap >  WriterType;
  typedef itk::ImageRegionIterator<OutputMap> OutIteratorType;
  typedef itk::ImageRegionConstIterator< OutputMap > ConstIteratorType;
  typedef itk::ImageRegionIterator< OutputMap > OIteratorType;


  OutputMap::Pointer im_input;
  im_input = Helpers::readImage<OutputMap>(argv[1]);
  OutputMap::SizeType size = im_input->GetLargestPossibleRegion().GetSize();
  OutputMap::IndexType pixelIndex;

  OutputMap::IndexType start;
  OutputMap::RegionType region;
  start[0] = 0; // first index on X
  start[1] = 0; // first index on Y

  region.SetSize(size);
  region.SetIndex( start );

  double max = -1;

  //TVL initializations
  vcl_vector< vnl_matrix_fixed<double, 2,2> > votee_matrices(size[0]*size[1]);
  vcl_vector< vnl_matrix_fixed<double, 2,2> > votee_matrices_initial;
  vnl_matrix_fixed<double,2,2> zero_matrix;
  vcl_vector< vnl_vector_fixed<double, 2> > inlocations;
  vcl_vector< vnl_vector_fixed<double, 2> > seeds;
  vcl_vector< vnl_vector_fixed<double, 2> > outlocations(size[0]*size[1]);
  vcl_vector< vnl_vector_fixed<double, 2> > outlocations_initial;
  vnl_vector_fixed<double,2> zero_location(0.0);
  vnl_vector_fixed<double, 2> neighbor;
  vnl_vector_fixed<double,1> zeroloc(0.0);
  vcl_vector< vnl_vector_fixed<double, 1> > sals;

  int ctr1;
  ctr1=0;

  vcl_cout<<"Noting the input locations "<<vcl_endl;

  OIteratorType inpIt(im_input,region);

  for(inpIt.GoToBegin();!inpIt.IsAtEnd(); ++inpIt)
  {
    double currpix = inpIt.Get();
    if(currpix>0)
    {
      inlocations.push_back(zero_location);
      pixelIndex = inpIt.GetIndex();
      inlocations[ctr1][0] = pixelIndex[0];
      inlocations[ctr1][1] = pixelIndex[1];
      votee_matrices_initial.push_back(zero_matrix);
      outlocations_initial.push_back(zero_location);
      outlocations_initial[ctr1][0] = pixelIndex[0];
      outlocations_initial[ctr1][1] = pixelIndex[1];
      votee_matrices_initial[ctr1][0][0] = 1;
      votee_matrices_initial[ctr1][0][1] = 0;
      votee_matrices_initial[ctr1][1][0] = 0;
      votee_matrices_initial[ctr1][1][1] = 1;
      ctr1 = ctr1+1;
    }
  }


  //while (! myfile.eof() )
  //{
  //      for(int i =0; i<2 ; i++)
  //      {
  //              myfile>>a[i];
  //      }
  //      inlocations.push_back(zero_location);
  //      inlocations[ctr1][0] = a[0];
  //      inlocations[ctr1][1] = a[1];
  //      votee_matrices_initial.push_back(zero_matrix);
  //      outlocations_initial.push_back(zero_location);
  //      outlocations_initial[ctr1][0] = a[0];
  //      outlocations_initial[ctr1][1] = a[1];

  //      votee_matrices_initial[ctr1][0][0] = 1;
  //      votee_matrices_initial[ctr1][0][1] = 0;
  //      votee_matrices_initial[ctr1][1][0] = 0;
  //      votee_matrices_initial[ctr1][1][1] = 1;
  //      ctr1 = ctr1+1;
  //}
  //
  //myfile.close();


  vcl_cout<<"Finished reading the seeds file "<<vcl_endl;

  long p = 0;

  for(unsigned int counter1 = 0; counter1 < size[0]; counter1++)
  {
    for(unsigned int counter2 = 0; counter2 < size[1]; counter2++)
    {
      votee_matrices[p][0][0] = 0;
      votee_matrices[p][0][1] = 0;
      votee_matrices[p][1][0] = 0;
      votee_matrices[p][1][1] = 0;
      p = p+1;
    }
  }


  //Initial Ball voting
  vnl_matrix_fixed<double, 2,2> voter_matrix;
  voter_matrix(0,0) = 1;
  voter_matrix(0,1) = 0;
  voter_matrix(1,0) = 0;
  voter_matrix(1,1) = 1;
  // Use "rtvl_tensor" to decompose the matrix.
  rtvl_tensor<2> voter_tensor_initial(voter_matrix);
  rtvl_weight_original<2> tvw(atoi(argv[2]));


  vcl_cout<<"Ball Voting in progress......."<<vcl_endl;

  for(unsigned int counter = 0; counter< inlocations.size() ; counter++)
  {
    vcl_cout<<counter<<vcl_endl;
    // Use "rtvl_voter" to encapsulate a token (location + input tensor).
    rtvl_voter<2> voter(inlocations[counter], voter_tensor_initial);
    //Sparse Ball Voting
    for(unsigned int vcounter = 0; vcounter< outlocations_initial.size(); vcounter++)
    {
      // Use "rtvl_votee" to encapsulate a site (location + output tensor).
      rtvl_votee<2> votee(outlocations_initial[vcounter], votee_matrices_initial[vcounter]);
      // Compute one vote.
      rtvl_vote(voter, votee, tvw);
    }
  }

  //During dense voting if the votee happens to be an initial token,
  // its matrix should be an updated value.

  for(unsigned int vcounter = 0; vcounter< outlocations_initial.size(); vcounter++)
  {
    long ctrx = ( outlocations_initial[vcounter][0])*size[1] + (outlocations_initial[vcounter][1]) ;
    votee_matrices[ctrx] = votee_matrices_initial[vcounter];
  }

  vcl_cout<<"Finished Ball Voting !"<<vcl_endl;
  vcl_cout<<"Creating Output Image and Iterators........"<<vcl_endl;


  SaliencyMap::Pointer Image = SaliencyMap::New();
  region.SetSize(size);
  region.SetIndex( start );
  Image->SetRegions( region );
  Image->Allocate();


  itk::NeighborhoodIterator<SaliencyMap>::IndexType loc;
  NeighborhoodIteratorType::RadiusType radius;

  vcl_cout<<"Am here !"<<vcl_endl;
  //radius.Fill();
  radius[0] = (atoi(argv[3])) ;//argv [2]
  radius[1] = (atoi(argv[3]));//argv [2]


  //"N"eighborhood "It"erators.
  itk::Neighborhood<double, 2> nhood;
  itk::NeighborhoodIterator<SaliencyMap> NIt(radius, Image, Image->GetRequestedRegion());
  vcl_cout<<"Finished creating Output Image  and Iterators!"<<vcl_endl;
  vcl_cout<<"Performing Dense Voting & Creating the Saliency Map........"<<vcl_endl;

  ctr1=0;
  int ctr=0;
  for(unsigned int counter = 0; counter< inlocations.size() ; counter++)
  {
    vcl_cout<<counter<<vcl_endl;


    //Encode the new information into a new tensor which will be used for dense voting.
    rtvl_tensor<2> voter_tensor(votee_matrices_initial[counter]);
    //Remove the ballness of the tensor !
    voter_tensor.remove_ballness(1);
    // Use "rtvl_voter" to encapsulate a token (location + input tensor).
    rtvl_voter<2> voter(inlocations[counter], voter_tensor);

    loc[0] = inlocations[counter][0];
    loc[1] = inlocations[counter][1];

    itk::Offset<2> off_set;

    NIt.SetLocation(loc);
    nhood = NIt.GetNeighborhood();
    for(unsigned int i = 0; i < nhood.Size(); ++i)
    {
      off_set = nhood.GetOffset(i);
      neighbor[0] = loc[0] + off_set[0];
      neighbor[1] = loc[1] + off_set[1];


      if(neighbor[0]<1) continue;
      if(neighbor[1]<1) continue;


      if(neighbor[0]>=size[0]) continue;
      if(neighbor[1]>=size[1]) continue;


      pixelIndex[0] = neighbor[0];
      pixelIndex[1] = neighbor[1];


      ctr = ( neighbor[0])*size[1] + (neighbor[1]);
      rtvl_votee<2> votee(neighbor, votee_matrices[ctr]);

      // Compute one vote.
      rtvl_vote(voter, votee, tvw);
      rtvl_tensor<2> votee_tensor(votee_matrices[ctr]);
      Image->SetPixel( pixelIndex, votee_tensor.saliency(0));
      if(max<votee_tensor.saliency(0))
      {
        max =   votee_tensor.saliency(0);
      }
    }
  }

  IteratorType salIt(Image,region);

  for(inpIt.GoToBegin(),salIt.GoToBegin();!inpIt.IsAtEnd(); ++inpIt,++salIt)
  {
    double currpix = salIt.Get();
    inpIt.Set(floor((currpix/max)*255));
  }

  Helpers::writeImage<OutputMap>(im_input,argv[4]);

  vcl_cout<<"Finished"<<vcl_endl;
  return 0;
}








































































































///* Copyright 2010 Brad King
//Distributed under the Boost Software License, Version 1.0.
//(See accompanying file rtvl_license_1_0.txt or copy at
//http://www.boost.org/LICENSE_1_0.txt) */
//
//#include "rtvl_hello_2D_image.h"
//#define MAX(a,b) (((a) > (b))?(a):(b))
//#define MIN(a,b) (((a) < (b))?(a):(b))
//
//template <typename T>
//typename T::Pointer readImage(const char *filename)
//{
//      printf("Reading %s ... ",filename);
//      typedef typename itk::ImageFileReader<T> ReaderType;
//      typename ReaderType::Pointer reader = ReaderType::New();
//
//      ReaderType::GlobalWarningDisplayOff();
//      reader->SetFileName(filename);
//      try
//      {
//              reader->Update();
//      }
//      catch(itk::ExceptionObject &err)
//      {
//              std::cout << "ExceptionObject caught!" <<std::endl;
//              std::cout << err << std::endl;
//              //return EXIT_FAILURE;
//      }
//      printf("Done.\n");
//      return reader->GetOutput();
//}
//
//
//
//
//template <typename T>
//int writeImage(typename T::Pointer im, const char* filename)
//{
//      printf("Writing %s ... ",filename);
//      typedef typename itk::ImageFileWriter<T> WriterType;
//
//      typename WriterType::Pointer writer = WriterType::New();
//      writer->SetFileName(filename);
//      writer->SetInput(im);
//      try
//      {
//              writer->Update();
//      }
//      catch(itk::ExceptionObject &err)
//      {
//              std::cout << "ExceptionObject caught!" <<std::endl;
//              std::cout << err << std::endl;
//              return EXIT_FAILURE;
//      }
//      printf("Done.\n");
//      return EXIT_SUCCESS;
//}
//
//
//
//
//int main()
//{
//
//      InputImageType::Pointer im_input;
//      DoubleImageType::Pointer gradx;
//      DoubleImageType::Pointer grady;
//      DoubleImageType::IndexType start;
//      DoubleImageType::RegionType region;
//      DoubleImageType::IndexType pixelIndex;
//
//      InputImageType::IndexType start2;
//      InputImageType::RegionType region2;
//      InputImageType::IndexType pixelIndex2;
//
//      //SaliencyMap::IndexType maxIndex;
//      start[0] = 0; // first index on X
//      start[1] = 0; // first index on Y
//
//      start2[0] = 0; // first index on X
//      start2[1] = 0; // first index on Y
//
//
//      im_input = readImage<InputImageType>("C:\\Lidar_TV\\daviddoria-TensorVoting-c7d0b33\\daviddoria-TensorVoting-c7d0b33\\data\\rectangleNoisy_8bit_inverted.tif");
//      InputImageType::SizeType size = im_input->GetLargestPossibleRegion().GetSize();
//
//
//      //TVL initializations
//      //vcl_vector< vnl_vector_fixed<double, 2> > neighbors;
//      vcl_vector< vnl_matrix_fixed<double, 2,2> > votee_matrices(size[0]*size[1]);
//      vcl_vector< vnl_matrix_fixed<double, 2,2> > votee_matrices_initial;
//      vnl_matrix_fixed<double, 2,2> zero_matrix;
//      zero_matrix.fill(0);
//      vcl_vector< vnl_vector_fixed<double, 2> > inlocations;
//      vcl_vector< vnl_vector_fixed<double, 2> > outlocations(size[0]*size[1]);
//      //vcl_vector< vnl_vector_fixed<double, 2> > outlocations_initial;
//      vnl_vector_fixed<double,2> zero_location(0.0);
//      vnl_vector_fixed<double, 2> neighbor;
//
//
//      DoubleImageType::Pointer SaliencyMap = DoubleImageType::New();
//      InputImageType::Pointer canny_output = InputImageType::New();
//    region.SetSize(size);
//    region.SetIndex( start );
//      SaliencyMap->SetRegions( region );
//      SaliencyMap->Allocate();
//
//
//      GFilterType::Pointer filterx = GFilterType::New();
//      GFilterType::Pointer filtery = GFilterType::New();
//      filterx->SetOrder(1);
//      filtery->SetOrder(1);
//      filterx->SetInput(im_input);
//      filtery->SetInput(im_input);
//      filterx->SetDirection(0);
//
//      gradx = filterx->GetOutput();
//      filterx->Update();
//      //Derivative along y
//    filtery->SetDirection(1);
//      filtery->Update();
//      grady = filtery->GetOutput();
//
//    int ctr1 = 0;
//
//
//      IteratorType inpIt(im_input,region);
//      DIteratorType gradxIt(gradx,region);
//      DIteratorType gradyIt(grady,region);
//      DIteratorType salIt(SaliencyMap,region);
//
//
//      for(inpIt.GoToBegin(),gradxIt.GoToBegin(),gradyIt.GoToBegin();!inpIt.IsAtEnd(); ++inpIt,++gradxIt,++gradyIt)
//      {
//              double currpix = inpIt.Get();
//              inlocations.push_back(zero_location);
//          pixelIndex = inpIt.GetIndex();
//              inlocations[ctr1][0] = pixelIndex[0];
//          inlocations[ctr1][1] = pixelIndex[1];
//              votee_matrices.at(ctr1) = zero_matrix;
//              if(currpix>0)
//              {
//                      //double pix1 = gradxIt.Get();
//                      //double pix2 = gradyIt.Get();
//
//                      //double lambda = sqrt(pix1*pix1+pix2*pix2);
//                      //votee_matrices[ctr1][0][0] = lambda*(pix1)*(pix1);
//                      //votee_matrices[ctr1][0][1] = lambda*(pix1)*(pix2) ;
//                      //votee_matrices[ctr1][1][0] = lambda*(pix1)*(pix2) ;
//                      //votee_matrices[ctr1][1][1] = lambda*(pix2)*(pix2) ;
//                      //std::cout<<lambda<<std::endl;
//                      votee_matrices[ctr1][0][0] = 1;
//                      votee_matrices[ctr1][0][1] = 0 ;
//                      votee_matrices[ctr1][1][0] = 0;
//                      votee_matrices[ctr1][1][1] = 1;
//              }
//              ctr1 = ctr1+1;
//      }
//
//      rtvl_weight_original<2> tvw(4);
//      NeighborhoodIteratorType::IndexType loc;
//      NeighborhoodIteratorType::RadiusType radius;
//      radius.Fill(10.0);
//
//      //"N"eighborhood "It"erators.
//    itk::Neighborhood<InputPixelType, 2> nhood;
//    itk::NeighborhoodIterator<InputImageType> NIt(radius, im_input, im_input->GetRequestedRegion());
//
//
//      vcl_cout<<"Sparse Voting in progress......."<<vcl_endl;
//
//      int counter = 0;
//
//      for(inpIt.GoToBegin();!inpIt.IsAtEnd(); ++inpIt)
//      {
//              double currpix = inpIt.Get();
//
//              if(currpix>0)
//              {
//              // Use "rtvl_tensor" to decompose the matrix.
//                      rtvl_tensor<2> voter_tensor_sparse(votee_matrices[counter]);
//                      rtvl_voter<2> voter(inlocations[counter], voter_tensor_sparse);
//                      loc[0] = inlocations[counter][0];
//                      loc[1] = inlocations[counter][1];
//
//                      itk::Offset<2> off_set;
//                      int ctr;
//
//                      NIt.SetLocation(loc);
//                      nhood = NIt.GetNeighborhood();
//
//                      for(int i = 0; i<nhood.Size(); ++i)
//                      {
//                              off_set = nhood.GetOffset(i);
//                              neighbor[0] = loc[0] + off_set[0];
//                              neighbor[1] = loc[1] + off_set[1];
//
//                              if(neighbor[0]<0) continue;
//                              if(neighbor[1]<0) continue;
//
//                              if(neighbor[0]>size[0]-1) continue;
//                              if(neighbor[1]>size[1]-1) continue;
//
//                              pixelIndex[0] = neighbor[0];
//                              pixelIndex[1] = neighbor[1];
//
//                              if(im_input->GetPixel(pixelIndex)>0)
//                              {
//                                      ctr = (neighbor[0])*size[1] + (neighbor[1]);
//                                      rtvl_votee<2> votee(neighbor, votee_matrices[ctr]);
//                                      // Compute one vote.
//                                      rtvl_vote(voter, votee, tvw);
//                                      rtvl_tensor<2> votee_tensor(votee_matrices[ctr]);
//                              }
//                      }
//              }
//              counter = counter + 1;
//      }
//
//       vcl_cout<<"Finished Sparse Voting !"<<vcl_endl;
//       radius.Fill(10.0);
//
//
//      ////"N"eighborhood "It"erators.
//      //itk::Neighborhood<double, 2> nhood;
//      //itk::NeighborhoodIterator<DoubleImageType> NIt(radius, SaliencyMap, SaliencyMap->GetRequestedRegion());
//      vcl_cout<<"Finished creating Saliency  and Iterators!"<<vcl_endl;
//    vcl_cout<<"Performing Dense Voting & Creating the Saliency Map........"<<vcl_endl;
//
//      long max = -1;
//      double maxval =-1;
//      //Reset ctr1
//      ctr1=0;
//
//      for( int counter = 0; counter< inlocations.size() ; counter++)
//      {
//              vcl_cout<<counter<<vcl_endl;
//
//              //Encode the new information into a new tensor which will be used for dense voting.
//              rtvl_tensor<2> voter_tensor(votee_matrices[counter]);
//              //Remove the ballness of the tensor !
//              //voter_tensor.remove_ballness(1);
//              // Use "rtvl_voter" to encapsulate a token (location + input tensor).
//              rtvl_voter<2> voter(inlocations[counter], voter_tensor);
//
//              loc[0] = inlocations[counter][0];
//              loc[1] = inlocations[counter][1];
//
//              itk::Offset<2> off_set;
//              int ctr;
//
//              NIt.SetLocation(loc);
//              nhood = NIt.GetNeighborhood();
//      for (int i = 0; i<nhood.Size(); ++i)
//                      {
//                              off_set = nhood.GetOffset(i);
//                              neighbor[0] = loc[0] + off_set[0];
//                              neighbor[1] = loc[1] + off_set[1];
//
//                  if(neighbor[0]<0) continue;
//                          if(neighbor[1]<0) continue;
//
//                              if(neighbor[0]>size[0]-1) continue;
//                              if(neighbor[1]>size[1]-1) continue;
//
//                              pixelIndex[0] = neighbor[0];
//                              pixelIndex[1] = neighbor[1];
//
//                              ctr = (neighbor[0])*size[1] + (neighbor[1]);
//                              rtvl_votee<2> votee(neighbor, votee_matrices[ctr]);
//
//                              // Compute one vote.
//                              rtvl_vote(voter, votee, tvw);
//                              rtvl_tensor<2> votee_tensor(votee_matrices[ctr]);
//                          SaliencyMap->SetPixel( pixelIndex, votee_tensor.saliency(0));
//
//                              if(votee_tensor.saliency(0) > maxval)
//                                      maxval = votee_tensor.saliency(0);
//                      }
//      }
//
//              for(inpIt.GoToBegin(),salIt.GoToBegin();!inpIt.IsAtEnd(); ++inpIt,++salIt)
//              {
//                      double currpix = salIt.Get();
//                      inpIt.Set(floor((currpix/maxval)*255));
//              }
//
//
//              writeImage<InputImageType>(im_input,"C:/output_1.tif");
//
//              return 0;
//      }
//
////
////pixelIndex  = maxIndex;
////    radius[0] = 1;
////    radius[1] = 1;
////    radius[2] = 1;
////
////    itk::NeighborhoodIterator<VectorImage> VIt(radius,QImage,QImage->GetRequestedRegion());
////
////    NeighborhoodIteratorType::OffsetType offset1 = {{-1,-1,-1}};
////    NeighborhoodIteratorType::OffsetType offset2 = {{1,-1,-1}};
////    NeighborhoodIteratorType::OffsetType offset3 = {{-1,1,-1 }};
////    NeighborhoodIteratorType::OffsetType offset4 = {{1,1,-1}};
////    NeighborhoodIteratorType::OffsetType offset5 = {{-1,-1,1}};
////    NeighborhoodIteratorType::OffsetType offset6 = {{1,-1,1}};
////    NeighborhoodIteratorType::OffsetType offset7 = {{-1,1,1 }};
////    NeighborhoodIteratorType::OffsetType offset8 = {{1,1,1}};
////
////    VIt.NeedToUseBoundaryConditionOn();
////    p= 0;
////    while(1)
////    {
////            //loc = pixelIndex;
////            loc[0]= 204;
////            loc[1]= 92;
////            loc[2]= 20;
////
////            pixelIndex[0]= 204;
////            pixelIndex[1]= 92;
////            pixelIndex[2]= 20;
////
////            int ctrx = ( pixelIndex[0])*260*35 + (pixelIndex[1])*35 + pixelIndex[2];
////            pixelIndex[0] = loc[0] + offset1[0];
////            pixelIndex[1] = loc[1] + offset1[1];
////            pixelIndex[2] = loc[2] + offset1[2];
////
////            projection = QImage->GetPixel(pixelIndex);
////                                            /*if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
////            vcl_cout<<projection<<vcl_endl;
////                            /*}*/
////
////
////            pixelIndex[0] = loc[0] + offset2[0];
////            pixelIndex[1] = loc[1] + offset2[1];
////            pixelIndex[2] = loc[2] + offset2[2];
////
////            projection = QImage->GetPixel(pixelIndex);
////
////                            /*              if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
////            vcl_cout<<projection<<vcl_endl;
////                            /*}*/
////
////
////            pixelIndex[0] = loc[0] + offset3[0];
////            pixelIndex[1] = loc[1] + offset3[1];
////            pixelIndex[2] = loc[2] + offset3[2];
////
////            projection = QImage->GetPixel(pixelIndex);
////
////                                    /*      if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
////            vcl_cout<<projection<<vcl_endl;
////                            //}
////
////            pixelIndex[0] = loc[0] + offset4[0];
////            pixelIndex[1] = loc[1] + offset4[1];
////            pixelIndex[2] = loc[2] + offset4[2];
////
////            projection = QImage->GetPixel(pixelIndex);
////
////                                            /*if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
////            vcl_cout<<projection<<vcl_endl;
////                            //}
////
////            pixelIndex[0] = loc[0] + offset5[0];
////            pixelIndex[1] = loc[1] + offset5[1];
////            pixelIndex[2] = loc[2] + offset5[2];
////
////            projection = QImage->GetPixel(pixelIndex);
////
////                                    /*      if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
////            vcl_cout<<projection<<vcl_endl;
////                            //}
////
////            pixelIndex[0] = loc[0] + offset6[0];
////            pixelIndex[1] = loc[1] + offset6[1];
////            pixelIndex[2] = loc[2] + offset6[2];
////
////            projection = QImage->GetPixel(pixelIndex);
////
////                                            /*if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
////            vcl_cout<<projection<<vcl_endl;
////                            //}
////
////            pixelIndex[0] = loc[0] + offset7[0];
////            pixelIndex[1] = loc[1] + offset7[1];
////            pixelIndex[2] = loc[2] + offset7[2];
////
////            projection = QImage->GetPixel(pixelIndex);
////
////                                            /*if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
////            vcl_cout<<projection<<vcl_endl;
////                            //}
////
////
////            pixelIndex[0] = loc[0] + offset8[0];
////            pixelIndex[1] = loc[1] + offset8[1];
////            pixelIndex[2] = loc[2] + offset8[2];
////
////            projection = QImage->GetPixel(pixelIndex);
////
////                                            /*if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
////            vcl_cout<<projection<<vcl_endl;
////                            //}
////            system("pause");
////            rtvl_tensor<3> votee_tensor_tangent(votee_matrices[ctrx]);
////            /*vnl_vector_fixed <double,3> tangent = votee_tensor_tangent.basis(2);
////            if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){
////                    vcl_cout<<floor(tangent[0]+0.5)<<vcl_endl;
////                    vcl_cout<<floor(tangent[1])<<vcl_endl;
////                    vcl_cout<<floor(tangent[2])<<vcl_endl;
////                    system("pause");
////            }*/
////            }
//
//      //InputImageType::IndexType start;
//      //InputImageType::RegionType region;
//      //InputImageType::IndexType pixelIndex;
//
//      //start[0] = 0; // first index on X
//      //start[1] = 0; // first index on Y
//
//
// //   DoubleImageType::Pointer gradx = DoubleImageType::New();
// //   region.SetSize(size);
// //   region.SetIndex( start );
//      //gradx->SetRegions( region );
//      //gradx->Allocate();
//
//
//
//
//
//
//
//
//
//      //for( int counter = 0; counter< inlocations.size() ; counter++)
//      //{
//      //      pixelIndex[0] = inlocations[counter][0];
//      //      pixelIndex[1] = inlocations[counter][1];
//      //
//      //      if(im_input->GetPixel(pixelIndex)>0)
//      //      {
//      //      // Use "rtvl_tensor" to decompose the matrix.
//      //      rtvl_tensor<2> voter_tensor_sparse(votee_matrices[counter]);
//      //      rtvl_voter<2> voter(inlocations[counter], voter_tensor_sparse);
//      //
//      //      loc[0] = inlocations[counter][0];
//      //      loc[1] = inlocations[counter][1];
//
//      //      itk::Offset<2> off_set;
//      //      int ctr;
//      //
//      //      NIt.SetLocation(loc);
//      //      nhood = NIt.GetNeighborhood();
// //           for (int i = 0; i<nhood.Size(); ++i)
//      //               {
//      //                off_set = nhood.GetOffset(i);
//      //                neighbor[0] = loc[0] + off_set[0];
// //             neighbor[1] = loc[1] + off_set[1];
//      //
//      //                if(neighbor[0]<1) neighbor[0] =0;
//      //                if(neighbor[1]<1) neighbor[1] =0;
//      //
//      //                if(neighbor[0]>size[0]) neighbor[0] =size[0];
//      //                if(neighbor[1]>size[1]) neighbor[1] =size[1];
//      //
//      //                pixelIndex[0] = neighbor[0];
//      //                pixelIndex[1] = neighbor[1];
//      //
//      //                if(im_input->GetPixel(pixelIndex)>0)
//      //                {
//      //                ctr = (neighbor[0])*size[1] + (neighbor[1]);
//      //                rtvl_votee<2> votee(neighbor, votee_matrices[ctr]);
//      //                // Compute one vote.
// //             rtvl_vote(voter, votee, tvw);
//      //                }
//      //              }
//      //      }
//      //}
//
//              //  //During dense voting if the votee happens to be an initial token,
//       // // its matrix should be an updated value.
//       //for(int vcounter = 0; vcounter< inlocations.size(); vcounter++)
//              //{
//              //      int ctrx = ( inlocations[vcounter][0])*size[1] + (inlocations[vcounter][1]);
//              //      votee_matrices[ctrx] = votee_matrices_initial[vcounter];
//              //}
//
//
//
//
//      //for(int i=0;i<size[0];i++)
//      //  {
//      //      for(int j =0; j<size[1] ; j++)
//      //               {
//      //                       pixelIndex[0] = i;
//      //                       pixelIndex[1] = j;
//      //                       double currpix = im_input->GetPixel(pixelIndex);
//      //
//      //                       inlocations.push_back(zero_location);
//      //                       inlocations[ctr1][0] = i;
//      //                       inlocations[ctr1][1] = j;
//      //                       votee_matrices.push_back(zero_matrix);
//      //                       if(currpix>0)
//      //                       {
//      //                       double pix1 = gradx->GetPixel(pixelIndex);
//      //                       double pix2 = grady->GetPixel(pixelIndex);
//      //                       double lambda = sqrt(pix1*pix1+pix2*pix2);
//      //                       votee_matrices[ctr1][0][0] = lambda*(gradx->GetPixel(pixelIndex))*(gradx->GetPixel(pixelIndex));
//      //                       votee_matrices[ctr1][0][1] = lambda*(gradx->GetPixel(pixelIndex))*(grady->GetPixel(pixelIndex)) ;
//      //                       votee_matrices[ctr1][1][0] = lambda*(gradx->GetPixel(pixelIndex))*(grady->GetPixel(pixelIndex));
//      //                       votee_matrices[ctr1][1][1] = lambda*(grady->GetPixel(pixelIndex))*(grady->GetPixel(pixelIndex));
//      //                       }
//      //                        ctr1 = ctr1+1;
//      //               }
//      //
//      //}
//
//
