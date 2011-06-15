/*=========================================================================
 *
 *  Copyright Raghav Padmanabhan, David Doria 2011
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

//Including the necessary header files
#include <stdio.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <vector>
#include <algorithm> // not sure if this is required ; check again later
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <string.h>
#include <iostream>
#include <list>
#include <itkArray.h>
#include "itkListSample.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCannyEdgeDetectionImageFilter.h"


#include "itkNeighborhood.h"
#include "itkNeighborhoodIterator.h"
#include "itkDerivativeImageFilter.h"
#include "itkZeroCrossingImageFilter.h"
#include <rtvl/rtvl_tensor.hxx>
#include <rtvl/rtvl_vote.hxx>
#include <rtvl/rtvl_votee.hxx>
#include <rtvl/rtvl_voter.hxx>
#include <rtvl/rtvl_weight_original.hxx>
#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_matrix_fixed.h>
//#include <vnl/math.h>
#include <vcl_vector.h>
#include <vcl_iostream.h>
#include <vcl_fstream.h>
#include <math.h>
//
//
//#include "itkBinaryThinningImageFilter3D.h"
//#include "itkBinaryThinningImageFilter3D.txx"
//

// Define the types that will be used in the program
// Define the types that will be used in the program
typedef unsigned char InputPixelType;
typedef double DoublePixelType;
typedef unsigned char OutputPixelType;
const unsigned int Dimension = 2;

//typedef itk::Image<short int,3> LabelImageType; // Check again if the image should be of type "short int"
//Remove later if these will not be required !

typedef itk::Image<InputPixelType,Dimension> InputImageType;
typedef itk::Image<DoublePixelType,Dimension> DoubleImageType;
typedef itk::Image<OutputPixelType,Dimension> OutputImageType;
//typedef itk::Image<short int,Dimension> LabelImageType;

//LabelImageType is the output of GetYousefSegmented program.
//Reason for using short int:
//After segmentation every pixel belonging to an object will have an id.
//The number of objects in an image might exceed 256 and in all probability will be less than 65k.
//So, short int will suffice.

//typedef itk::ImageRegionConstIterator<InputImageType> ConstIteratorType;
//typedef itk::ImageRegionConstIterator<SaliencyMap> SalMapIteratorType;

//Filters used
//typedef itk::DerivativeImageFilter<InputImageType,SaliencyMap >  DFilterType;
//typedef itk::ZeroCrossingImageFilter<SaliencyMap,OutputImageType >  ZFilterType;
typedef itk::CastImageFilter< InputImageType, DoubleImageType> CastToRealFilterType;
typedef itk::CannyEdgeDetectionImageFilter<DoubleImageType,DoubleImageType> cFilterType;
typedef itk::RescaleIntensityImageFilter<DoubleImageType, InputImageType > RescaleFilter;
typedef itk::DerivativeImageFilter<InputImageType, DoubleImageType >  GFilterType;

typedef itk::ImageFileReader< InputImageType >  ReaderType;
//WriterType writes labeled images
typedef itk::ImageFileWriter< DoubleImageType >  WriterType;
typedef itk::NeighborhoodIterator< DoubleImageType > NeighborhoodIteratorType;





//typedef itk::ImageRegionIterator<LabelImageType> LabelIteratorType;
//typedef itk::ImageFileWriter< intType >  WriterType2;
//
//
//typedef unsigned char OutputPixelType;
//typedef itk::Image<OutputPixelType,3> OutputMap;
//typedef itk::ImageFileWriter< SaliencyMap >  WriterType;
//typedef itk::ImageRegionIterator<OutputMap> OutIteratorType;
//typedef itk::ImageRegionConstIterator< OutputMap > ConstIteratorType;
//typedef itk::ImageRegionIterator< OutputMap > IteratorTypez;
//typedef itk::BinaryThinningImageFilter3D< OutputMap, OutputMap > ThinningFilterType;

/* Copyright 2010 Brad King
Distributed under the Boost Software License, Version 1.0.
(See accompanying file rtvl_license_1_0.txt or copy at
http://www.boost.org/LICENSE_1_0.txt) */

#include "rtvl_hello_2D_image.h"

int main()
{

        InputImageType::Pointer im_input;
        DoubleImageType::Pointer gradx;
        DoubleImageType::Pointer grady;
        DoubleImageType::IndexType start;
        DoubleImageType::RegionType region;
        DoubleImageType::IndexType pixelIndex;

        InputImageType::IndexType start2;
        InputImageType::RegionType region2;
        InputImageType::IndexType pixelIndex2;



        //SaliencyMap::IndexType maxIndex;
        start[0] = 0; // first index on X
        start[1] = 0; // first index on Y

        start2[0] = 0; // first index on X
        start2[1] = 0; // first index on Y


        //Read the input image
ReaderType::Pointer reader = ReaderType::New();
reader->SetFileName("C:/lab/VXL_TV/bin/contrib/rpl/rtvl/debug/Image.tif");
try
        {
                reader->Update();
        }
catch(itk::ExceptionObject &err)
        {
                std::cerr << "ExceptionObject caught!" <<std::endl;
                std::cerr << err << std::endl;
                //return EXIT_FAILURE;
        }
        printf("Done.\n");

        im_input = reader->GetOutput();
    InputImageType::SizeType size = im_input->GetLargestPossibleRegion().GetSize();


                //TVL initializations
        //vcl_vector< vnl_vector_fixed<double, 2> > neighbors;
        vcl_vector< vnl_matrix_fixed<double, 2,2> > votee_matrices(size[0]*size[1]);
        vcl_vector< vnl_matrix_fixed<double, 2,2> > votee_matrices_initial;
        vnl_matrix_fixed<double,2,2> zero_matrix;
        vcl_vector< vnl_vector_fixed<double, 2> > inlocations;
        vcl_vector< vnl_vector_fixed<double, 2> > outlocations(size[0]*size[1]);
        //vcl_vector< vnl_vector_fixed<double, 2> > outlocations_initial;
        vnl_vector_fixed<double,2> zero_location(0.0);
        vnl_vector_fixed<double, 2> neighbor;




        DoubleImageType::Pointer SaliencyMap = DoubleImageType::New();
        InputImageType::Pointer canny_output = InputImageType::New();
    region.SetSize(size);
    region.SetIndex( start );
        SaliencyMap->SetRegions( region );
        SaliencyMap->Allocate();
    //canny_output->SetRegions( region );
    //canny_output->Allocate();


        GFilterType::Pointer filterx = GFilterType::New();
        GFilterType::Pointer filtery = GFilterType::New();
        filterx->SetOrder(1);
        filtery->SetOrder(1);
        filterx->SetInput(im_input);
        filtery->SetInput(im_input);
        filterx->SetDirection(0);

        gradx = filterx->GetOutput();
        filterx->Update();
        //Derivative along y
    filtery->SetDirection(1);
        filtery->Update();
        grady = filtery->GetOutput();

         CastToRealFilterType::Pointer toCanny = CastToRealFilterType::New();
         RescaleFilter::Pointer rescale = RescaleFilter::New();
         cFilterType::Pointer cFilter = cFilterType::New();
         WriterType::Pointer writer = WriterType::New();

         rescale->SetOutputMinimum(   0 );
         rescale->SetOutputMaximum( 255 );

         toCanny->SetInput(im_input);
         cFilter->SetInput( toCanny->GetOutput() );
         cFilter->SetVariance(2.0);
         cFilter->SetThreshold(2.0);
         rescale->SetInput( cFilter->GetOutput() );
         rescale->Update();
         canny_output = rescale->GetOutput();
         int ctr1 = 0;

        for(int i=0;i<size[0];i++)
          {
                for(int j =0; j<size[1] ; j++)
                         {
                                 pixelIndex[0] = i;
                                 pixelIndex[1] = j;
                                 double currpix = canny_output->GetPixel(pixelIndex);

                                 inlocations.push_back(zero_location);
                                 inlocations[ctr1][0] = i;
                                 inlocations[ctr1][1] = j;
                                 votee_matrices.push_back(zero_matrix);
                                 if(currpix>0)
                                 {
                                 double pix1 = gradx->GetPixel(pixelIndex);
                                 double pix2 = grady->GetPixel(pixelIndex);
                                 double lambda = sqrt(pix1*pix1+pix2*pix2);
                                 votee_matrices[ctr1][0][0] = lambda*(gradx->GetPixel(pixelIndex))*(gradx->GetPixel(pixelIndex));
                                 votee_matrices[ctr1][0][1] = lambda*(gradx->GetPixel(pixelIndex))*(grady->GetPixel(pixelIndex)) ;
                                 votee_matrices[ctr1][1][0] = lambda*(gradx->GetPixel(pixelIndex))*(grady->GetPixel(pixelIndex));
                                 votee_matrices[ctr1][1][1] = lambda*(grady->GetPixel(pixelIndex))*(grady->GetPixel(pixelIndex));
                                 }
                                  ctr1 = ctr1+1;
                }


        }

        rtvl_weight_original<2> tvw(20);

        NeighborhoodIteratorType::IndexType loc;
        NeighborhoodIteratorType::RadiusType radius;
        radius.Fill(25.0);
         //"N"eighborhood "It"erators.
    itk::Neighborhood<InputPixelType, 2> nhood;
    itk::NeighborhoodIterator<InputImageType> NIt(radius, canny_output, canny_output->GetRequestedRegion());

        vcl_cout<<"Sparse Voting in progress......."<<vcl_endl;

        for( int counter = 0; counter< inlocations.size() ; counter++)
        {
                pixelIndex[0] = inlocations[counter][0];
                pixelIndex[1] = inlocations[counter][1];

                if(canny_output->GetPixel(pixelIndex)>0)
                {
                // Use "rtvl_tensor" to decompose the matrix.
                rtvl_tensor<2> voter_tensor_sparse(votee_matrices[counter]);
                rtvl_voter<2> voter(inlocations[counter], voter_tensor_sparse);

                loc[0] = inlocations[counter][0];
                loc[1] = inlocations[counter][1];

                itk::Offset<2> off_set;
                int ctr;

                NIt.SetLocation(loc);
                nhood = NIt.GetNeighborhood();
        for (int i = 0; i<nhood.Size(); ++i)
                         {
                          off_set = nhood.GetOffset(i);
                          neighbor[0] = loc[0] + off_set[0];
              neighbor[1] = loc[1] + off_set[1];

                          if(neighbor[0]<1) neighbor[0] =0;
                          if(neighbor[1]<1) neighbor[1] =0;

                          if(neighbor[0]>size[0]) neighbor[0] =size[0];
                          if(neighbor[1]>size[1]) neighbor[1] =size[1];

                          pixelIndex[0] = neighbor[0];
                          pixelIndex[1] = neighbor[1];

                          if(canny_output->GetPixel(pixelIndex)>0)
                          {
                          ctr = (neighbor[0])*size[1] + (neighbor[1]);
                          rtvl_votee<2> votee(neighbor, votee_matrices[ctr]);
                          // Compute one vote.
              rtvl_vote(voter, votee, tvw);
                          }
                        }
                }
        }

                //  //During dense voting if the votee happens to be an initial token,
         // // its matrix should be an updated value.
         //for(int vcounter = 0; vcounter< inlocations.size(); vcounter++)
                //{
                //      int ctrx = ( inlocations[vcounter][0])*size[1] + (inlocations[vcounter][1]);
                //      votee_matrices[ctrx] = votee_matrices_initial[vcounter];
                //}

         vcl_cout<<"Finished Sparse Voting !"<<vcl_endl;



        //// Zero Crossing Images:
        //OutputMap::Pointer zeroImagex = OutputMap::New();
        //zeroImagex->SetRegions( region );
        //zeroImagex->Allocate();
        //
        //OutputMap::Pointer zeroImagey = OutputMap::New();
        //zeroImagey->SetRegions( region );
        //zeroImagey->Allocate();

        //// Zero Crossing Image:
        //OutputMap::Pointer zeroImagef = OutputMap::New();
        //zeroImagef->SetRegions( region );
        //zeroImagef->Allocate();

        //OutputMap::Pointer curve = OutputMap::New();
        //curve->SetRegions( region );
        //curve->Allocate();


        radius.Fill(10.0);


        ////"N"eighborhood "It"erators.
        //itk::Neighborhood<double, 2> nhood;
        //itk::NeighborhoodIterator<DoubleImageType> NIt(radius, SaliencyMap, SaliencyMap->GetRequestedRegion());
        vcl_cout<<"Finished creating Saliency  and Iterators!"<<vcl_endl;
    vcl_cout<<"Performing Dense Voting & Creating the Saliency Map........"<<vcl_endl;

        long max = -1;
        //Reset ctr1
        ctr1=0;

        for( int counter = 0; counter< inlocations.size() ; counter++)
        {
                vcl_cout<<counter<<vcl_endl;
                //Encode the new information into a new tensor which will be used for dense voting.
                rtvl_tensor<2> voter_tensor(votee_matrices[counter]);
                //Remove the ballness of the tensor !
                voter_tensor.remove_ballness(1);
                // Use "rtvl_voter" to encapsulate a token (location + input tensor).
                rtvl_voter<2> voter(inlocations[counter], voter_tensor);

                loc[0] = inlocations[counter][0];
                loc[1] = inlocations[counter][1];

                itk::Offset<2> off_set;
                int ctr;

                NIt.SetLocation(loc);
                nhood = NIt.GetNeighborhood();
        for (int i = 0; i<nhood.Size(); ++i)
                         {
                          off_set = nhood.GetOffset(i);
                          neighbor[0] = loc[0] + off_set[0];
              neighbor[1] = loc[1] + off_set[1];


                          if(neighbor[0]<1) neighbor[0] =0;
                          if(neighbor[1]<1) neighbor[1] =0;

                          if(neighbor[0]>size[0]) neighbor[0] =size[0];
                          if(neighbor[1]>size[1]) neighbor[1] =size[1];

                          pixelIndex[0] = neighbor[0];
                          pixelIndex[1] = neighbor[1];

                          ctr = ( neighbor[0])*size[1] + (neighbor[1]);
                          rtvl_votee<2> votee(neighbor, votee_matrices[ctr]);

                          // Compute one vote.
                                rtvl_vote(voter, votee, tvw);
                                rtvl_tensor<2> votee_tensor(votee_matrices[ctr]);
                            SaliencyMap->SetPixel( pixelIndex, votee_tensor.saliency(0));
            }
        }

         vcl_cout<< "Finished Dense Voting ! " <<vcl_endl;
         writer->SetFileName("OutputImage.vtk");
         writer->SetInput(SaliencyMap);

        try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught!!!! !" << std::endl;
    std::cerr << err << std::endl;
    }
        vcl_cout<<"Finished"<<vcl_endl;

        return 0;
}

//
//pixelIndex    = maxIndex;
//      radius[0] = 1;
//      radius[1] = 1;
//      radius[2] = 1;
//
//      itk::NeighborhoodIterator<VectorImage> VIt(radius,QImage,QImage->GetRequestedRegion());
//
//      NeighborhoodIteratorType::OffsetType offset1 = {{-1,-1,-1}};
//      NeighborhoodIteratorType::OffsetType offset2 = {{1,-1,-1}};
//      NeighborhoodIteratorType::OffsetType offset3 = {{-1,1,-1 }};
//      NeighborhoodIteratorType::OffsetType offset4 = {{1,1,-1}};
//      NeighborhoodIteratorType::OffsetType offset5 = {{-1,-1,1}};
//      NeighborhoodIteratorType::OffsetType offset6 = {{1,-1,1}};
//      NeighborhoodIteratorType::OffsetType offset7 = {{-1,1,1 }};
//      NeighborhoodIteratorType::OffsetType offset8 = {{1,1,1}};
//
//      VIt.NeedToUseBoundaryConditionOn();
//      p= 0;
//      while(1)
//      {
//              //loc = pixelIndex;
//              loc[0]= 204;
//              loc[1]= 92;
//              loc[2]= 20;
//
//              pixelIndex[0]= 204;
//              pixelIndex[1]= 92;
//              pixelIndex[2]= 20;
//
//              int ctrx = ( pixelIndex[0])*260*35 + (pixelIndex[1])*35 + pixelIndex[2];
//              pixelIndex[0] = loc[0] + offset1[0];
//              pixelIndex[1] = loc[1] + offset1[1];
//              pixelIndex[2] = loc[2] + offset1[2];
//
//              projection = QImage->GetPixel(pixelIndex);
//                                              /*if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
//              vcl_cout<<projection<<vcl_endl;
//                              /*}*/
//
//
//              pixelIndex[0] = loc[0] + offset2[0];
//              pixelIndex[1] = loc[1] + offset2[1];
//              pixelIndex[2] = loc[2] + offset2[2];
//
//              projection = QImage->GetPixel(pixelIndex);
//
//                              /*              if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
//              vcl_cout<<projection<<vcl_endl;
//                              /*}*/
//
//
//              pixelIndex[0] = loc[0] + offset3[0];
//              pixelIndex[1] = loc[1] + offset3[1];
//              pixelIndex[2] = loc[2] + offset3[2];
//
//              projection = QImage->GetPixel(pixelIndex);
//
//                                      /*      if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
//              vcl_cout<<projection<<vcl_endl;
//                              //}
//
//              pixelIndex[0] = loc[0] + offset4[0];
//              pixelIndex[1] = loc[1] + offset4[1];
//              pixelIndex[2] = loc[2] + offset4[2];
//
//              projection = QImage->GetPixel(pixelIndex);
//
//                                              /*if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
//              vcl_cout<<projection<<vcl_endl;
//                              //}
//
//              pixelIndex[0] = loc[0] + offset5[0];
//              pixelIndex[1] = loc[1] + offset5[1];
//              pixelIndex[2] = loc[2] + offset5[2];
//
//              projection = QImage->GetPixel(pixelIndex);
//
//                                      /*      if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
//              vcl_cout<<projection<<vcl_endl;
//                              //}
//
//              pixelIndex[0] = loc[0] + offset6[0];
//              pixelIndex[1] = loc[1] + offset6[1];
//              pixelIndex[2] = loc[2] + offset6[2];
//
//              projection = QImage->GetPixel(pixelIndex);
//
//                                              /*if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
//              vcl_cout<<projection<<vcl_endl;
//                              //}
//
//              pixelIndex[0] = loc[0] + offset7[0];
//              pixelIndex[1] = loc[1] + offset7[1];
//              pixelIndex[2] = loc[2] + offset7[2];
//
//              projection = QImage->GetPixel(pixelIndex);
//
//                                              /*if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
//              vcl_cout<<projection<<vcl_endl;
//                              //}
//
//
//              pixelIndex[0] = loc[0] + offset8[0];
//              pixelIndex[1] = loc[1] + offset8[1];
//              pixelIndex[2] = loc[2] + offset8[2];
//
//              projection = QImage->GetPixel(pixelIndex);
//
//                                              /*if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){*/
//              vcl_cout<<projection<<vcl_endl;
//                              //}
//              system("pause");
//      rtvl_tensor<3> votee_tensor_tangent(votee_matrices[ctrx]);
//              /*vnl_vector_fixed <double,3> tangent = votee_tensor_tangent.basis(2);
//              if(loc[0]== 204 && loc[1]== 92 && loc[2]== 20){
//                      vcl_cout<<floor(tangent[0]+0.5)<<vcl_endl;
//                      vcl_cout<<floor(tangent[1])<<vcl_endl;
//                      vcl_cout<<floor(tangent[2])<<vcl_endl;
//                      system("pause");
//              }*/
//              }

        //InputImageType::IndexType start;
        //InputImageType::RegionType region;
        //InputImageType::IndexType pixelIndex;

        //start[0] = 0; // first index on X
        //start[1] = 0; // first index on Y


 //   DoubleImageType::Pointer gradx = DoubleImageType::New();
 //   region.SetSize(size);
 //   region.SetIndex( start );
        //gradx->SetRegions( region );
        //gradx->Allocate();    