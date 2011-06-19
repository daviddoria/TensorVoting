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


int main(int argc, char*argv[])
{
  // Verify arguments
  if(argc < 5)
  {
    std::cerr << "Required arguments: InputFilename VotingFieldParameter DenseVotingFieldRange OutputFileName" << std::endl;
    return EXIT_FAILURE;
  }

  // Parse arguments
  std::string inputFileName = argv[1];

  int votingFieldParameter = 0;
  std::stringstream strVotingFieldParameter(argv[2]);
  strVotingFieldParameter >> votingFieldParameter;

  int denseVotingFieldRange = 0;
  std::stringstream strDenseVotingFieldRange(argv[3]);
  strDenseVotingFieldRange >> denseVotingFieldRange;

  std::string outputFileName = argv[4];

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
  im_input = Helpers::readImage<OutputMap>(inputFileName);
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
  
  vnl_vector_fixed<double, 2> neighbor;
  
  vcl_vector< vnl_vector_fixed<double, 1> > sals;

  vcl_cout << "Noting the input locations " << vcl_endl;

  OIteratorType inputIterator(im_input,region);

  for(inputIterator.GoToBegin();!inputIterator.IsAtEnd(); ++inputIterator)
  {
    if(inputIterator.Get() != 0) // This is one of the "foreground" points
    {
      pixelIndex = inputIterator.GetIndex();

      vnl_vector_fixed<double,2> location;
      location[0] = pixelIndex[0];
      location[1] = pixelIndex[1];

      inlocations.push_back(location);
      outlocations_initial.push_back(location);      

      vnl_matrix_fixed<double, 2,2> voteeMatrix;
      voteeMatrix[0][0] = 1;
      voteeMatrix[0][1] = 0;
      voteeMatrix[1][0] = 0;
      voteeMatrix[1][1] = 1;
      votee_matrices_initial.push_back(voteeMatrix);
    }
  }

  vcl_cout << "Finished reading the seeds file" << vcl_endl;

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
  rtvl_weight_original<2> tvw(votingFieldParameter);

  vcl_cout << "Ball Voting in progress......." << vcl_endl;

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

  vcl_cout << "Finished Ball Voting!" << vcl_endl;
  vcl_cout << "Creating Output Image and Iterators........" << vcl_endl;

  SaliencyMap::Pointer Image = SaliencyMap::New();
  region.SetSize(size);
  region.SetIndex( start );
  Image->SetRegions( region );
  Image->Allocate();

  itk::NeighborhoodIterator<SaliencyMap>::IndexType loc;
  NeighborhoodIteratorType::RadiusType radius;

  vcl_cout<<"Am here !"<<vcl_endl;
  //radius.Fill();
  radius[0] = denseVotingFieldRange;
  radius[1] = denseVotingFieldRange;

  //"N"eighborhood "It"erators.
  itk::Neighborhood<double, 2> nhood;
  itk::NeighborhoodIterator<SaliencyMap> NIt(radius, Image, Image->GetRequestedRegion());
  vcl_cout<<"Finished creating Output Image  and Iterators!"<<vcl_endl;
  vcl_cout<<"Performing Dense Voting & Creating the Saliency Map........"<<vcl_endl;

  int ctr=0;
  for(unsigned int counter = 0; counter< inlocations.size() ; counter++)
  {
    vcl_cout << counter << vcl_endl;

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

      if(neighbor[0]<1 || neighbor[1]<1 || neighbor[0]>=size[0] || neighbor[1]>=size[1])
      {
        continue;
      }

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
        max = votee_tensor.saliency(0);
      }
    }
  }

  IteratorType salIt(Image,region);

  for(inputIterator.GoToBegin(),salIt.GoToBegin();!inputIterator.IsAtEnd(); ++inputIterator,++salIt)
  {
    double currpix = salIt.Get();
    inputIterator.Set(floor((currpix/max)*255));
  }

  Helpers::writeImage<OutputMap>(im_input, outputFileName);

  vcl_cout << "Finished" << vcl_endl;

  return EXIT_SUCCESS;
}
