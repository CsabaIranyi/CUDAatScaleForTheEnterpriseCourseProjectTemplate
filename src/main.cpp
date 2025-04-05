
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

bool printfNPPinfo(int argc, char *argv[])
{
    // NPP Library version
    const NppLibraryVersion *libVer = nppGetLibVersion();
    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    // CUDA Driver and Runtime version
    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

int main(int argc, char *argv[])
{
  printf("Starting image Gauss filter program\n\n");

  try
  {
    // Input file name
    std::string sFilename;
    char *filePath;

    // Find and select desired CUDA device
    findCudaDevice(argc, (const char **)argv);

    // Show NPP versions
    if (printfNPPinfo(argc, argv) == false)
    {
      exit(EXIT_SUCCESS);
    }

    // Check input file argument
    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    }
    else
    {
      // Default test file
      filePath = sdkFindFilePath("Lena.pgm", argv[0]);
    }

    if (filePath)
    {
      sFilename = filePath;
    }
    else
    {
      sFilename = "Lena.pgm";
    }
    printf("\nSource image file: %s\n", sFilename.data());

    // Check input image file
    std::ifstream infile(sFilename.data(), std::ifstream::in);
    if (infile.good())
    {
      printf("Check source image file: success\n");
      infile.close();
    }
    else
    {
      printf("Check source image file: failed\n");
      infile.close();
      exit(EXIT_FAILURE);
    }

    // Construct output filename automatically
    std::string sResultFilename = sFilename;
    std::string::size_type dot = sResultFilename.rfind('.');
    if (dot != std::string::npos)
    {
      sResultFilename = sResultFilename.substr(0, dot);
    }
    sResultFilename += "_gaussian.pgm";

    // Check and apply explicit output filename argument
    if (checkCmdLineFlag(argc, (const char **)argv, "output"))
    {
      char *outputFilePath;
      getCmdLineArgumentString(argc, (const char **)argv, "output", &outputFilePath);
      sResultFilename = outputFilePath;
    }
    printf("Destination image file: %s\n", sResultFilename.data());

    // Check and apply gauss filter mask size argument
    NppiMaskSize gaussMaskSize = NPP_MASK_SIZE_11_X_11;
    if (checkCmdLineFlag(argc, (const char **)argv, "mask"))
    {
      char *maskSize;
      getCmdLineArgumentString(argc, (const char **)argv, "mask", &maskSize);
      std::string sGaussMaskSizeStr{maskSize};

      // 3 X 3 filter mask size, leaving space for more N X 1 type enum values.
      if (sGaussMaskSizeStr == "3")
      {
        gaussMaskSize = NppiMaskSize::NPP_MASK_SIZE_3_X_3;
        printf("Mask size: 3 X 3\n");
      }
      // 5 X 5 filter mask size.
      else if (sGaussMaskSizeStr == "5")
      {
        gaussMaskSize = NppiMaskSize::NPP_MASK_SIZE_5_X_5;
        printf("Mask size: 5 X 5\n");
      }
      // 7 X 7 filter mask size.
      else if (sGaussMaskSizeStr == "7")
      {
        gaussMaskSize = NppiMaskSize::NPP_MASK_SIZE_7_X_7;
        printf("Mask size: 7 X 7\n");
      }
      // 9 X 9 filter mask size.
      else if (sGaussMaskSizeStr == "9")
      {
        gaussMaskSize = NppiMaskSize::NPP_MASK_SIZE_9_X_9;
        printf("Mask size: 9 X 9\n");
      }
      // 11 X 11 filter mask size.
      else if (sGaussMaskSizeStr == "11")
      {
        gaussMaskSize = NppiMaskSize::NPP_MASK_SIZE_11_X_11;
        printf("Mask size: 11 X 11\n");
      }
      // 13 X 13 filter mask size.
      else if (sGaussMaskSizeStr == "13")
      {
        gaussMaskSize = NppiMaskSize::NPP_MASK_SIZE_13_X_13;
        printf("Mask size: 13 X 13\n");
      }
      // 15 X 15 filter mask size.
      else if (sGaussMaskSizeStr == "15")
      {
        gaussMaskSize = NppiMaskSize::NPP_MASK_SIZE_15_X_15;
        printf("Mask size: 15 X 15\n");
      }
    }

    // CPU source image data
    npp::ImageCPU_8u_C1 hostSrc;

    // Load input image from filesystem
    printf("Load source image\n");
    npp::loadImage(sFilename, hostSrc);

    // GPU source image data
    npp::ImageNPP_8u_C1 deviceSrc(hostSrc);

    // Image size
    const NppiSize srcSize = {(int)deviceSrc.width(), (int)deviceSrc.height()};
    const NppiPoint srcOffset = {0, 0};
    const NppiSize roiSize = {(int)deviceSrc.width(), (int)deviceSrc.height()};

    // GPU result image data
    npp::ImageNPP_8u_C1 deviceDst(roiSize.width, roiSize.height);

    // Apply Single channel 8-bit unsigned Gauss filter with border control
    printf("Apply Gauss filter\n");
    NPP_CHECK_NPP(nppiFilterGaussBorder_8u_C1R(deviceSrc.data(),
                                               deviceSrc.pitch(),
                                               srcSize,
                                               srcOffset,
                                               deviceDst.data(),
                                               deviceDst.pitch(),
                                               roiSize,
                                               gaussMaskSize,
                                               NppiBorderType::NPP_BORDER_REPLICATE));

    // CPU result image data
    npp::ImageCPU_8u_C1 hostDst(deviceDst.size());

    // Copy result data from GPU
    printf("Copy result from GPU\n");
    deviceDst.copyTo(hostDst.data(), hostDst.pitch());

    // Save result image to filesystem
    printf("Save destination image\n");
    saveImage(sResultFilename, hostDst);

    // Free allocated data
    nppiFree(deviceSrc.data());
    nppiFree(deviceDst.data());
    nppiFree(hostSrc.data());
    nppiFree(hostDst.data());

    exit(EXIT_SUCCESS);
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
