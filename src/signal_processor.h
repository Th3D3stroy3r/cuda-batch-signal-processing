#ifndef CUDA_SIGNAL_PROJECT_SRC_SIGNAL_PROCESSOR_H_
#define CUDA_SIGNAL_PROJECT_SRC_SIGNAL_PROCESSOR_H_

#include <cuda_runtime.h>
#include <string>
#include <vector>

struct Config {
  std::string input_dir = "data/input";
  std::string output_dir = "data/output";
  int num_files = 200;
  int signal_length = 1024;
  int filter_window = 7;
};

#define CHECK_CUDA(val) CheckCuda((val), #val, __FILE__, __LINE__)
inline void CheckCuda(cudaError_t err, const char* const func,
                      const char* const file, const int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error at %s:%d code=%d(%s) \"%s\"\n", file, line,
            static_cast<unsigned int>(err), cudaGetErrorString(err), func);
    exit(EXIT_FAILURE);
  }
}

Config ParseCommandLineArgs(int argc, char** argv);
void GenerateSyntheticSignals(const Config& config);
bool ReadSignalCsv(const std::string& filepath, std::vector<float>* signal);
bool WriteSignalCsv(const std::string& filepath,
                    const std::vector<float>& signal);
void RunBatchGpuSignalProcessing(const Config& config);

#endif  // CUDA_SIGNAL_PROJECT_SRC_SIGNAL_PROCESSOR_H_
