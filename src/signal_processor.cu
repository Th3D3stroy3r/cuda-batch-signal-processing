#include "signal_processor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
constexpr int kThreadsPerBlock = 256;
constexpr int kNumStreams = 4;
constexpr float kPi = 3.14159265358979323846f;
}  // namespace

__global__ void MovingAverageDenoiseKernel(const float* __restrict__ d_input,
                                           float* __restrict__ d_output,
                                           int signal_length,
                                           int filter_window) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < signal_length) {
    int half_win = filter_window / 2;
    float sum = 0.0f;
    int count = 0;

    for (int offset = -half_win; offset <= half_win; ++offset) {
      int neighbor_idx = idx + offset;
      if (neighbor_idx >= 0 && neighbor_idx < signal_length) {
        sum += d_input[neighbor_idx];
        ++count;
      }
    }
    d_output[idx] = sum / static_cast<float>(count);
  }
}

Config ParseCommandLineArgs(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input_dir" && i + 1 < argc) {
      config.input_dir = argv[++i];
    } else if (arg == "--output_dir" && i + 1 < argc) {
      config.output_dir = argv[++i];
    } else if (arg == "--num_files" && i + 1 < argc) {
      config.num_files = std::stoi(argv[++i]);
    } else if (arg == "--signal_length" && i + 1 < argc) {
      config.signal_length = std::stoi(argv[++i]);
    } else if (arg == "--filter_window" && i + 1 < argc) {
      config.filter_window = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: ./signal_processor.exe [--input_dir DIR] "
                << "[--output_dir DIR] [--num_files N] "
                << "[--signal_length LEN] [--filter_window WIN]\n";
      exit(EXIT_SUCCESS);
    }
  }
  return config;
}

void GenerateSyntheticSignals(const Config& config) {
  std::cout << "[Data Generator] Creating " << config.num_files
            << " noisy signal CSV files (" << config.signal_length
            << " samples each) in '" << config.input_dir << "/'...\n";
  srand(42);
  for (int file_idx = 0; file_idx < config.num_files; ++file_idx) {
    std::ostringstream filepath;
    filepath << config.input_dir << "/signal_" << std::setw(3)
             << std::setfill('0') << file_idx << ".csv";

    std::vector<float> noisy_signal(config.signal_length);
    float freq = 2.0f + static_cast<float>(file_idx % 5);

    for (int i = 0; i < config.signal_length; ++i) {
      float t = static_cast<float>(i) / static_cast<float>(config.signal_length);
      float clean_val = std::sin(2.0f * kPi * freq * t);
      float noise = ((static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f) * 0.35f;
      noisy_signal[i] = clean_val + noise;
    }
    WriteSignalCsv(filepath.str(), noisy_signal);
  }
  std::cout << "[Data Generator] Successfully generated " << config.num_files
            << " input CSV signal files.\n";
}

bool ReadSignalCsv(const std::string& filepath, std::vector<float>* signal) {
  std::ifstream file(filepath);
  if (!file.is_open()) return false;

  signal->clear();
  std::string line, token;
  if (std::getline(file, line)) {
    std::istringstream ss(line);
    while (std::getline(ss, token, ',')) {
      if (!token.empty()) {
        signal->push_back(std::stof(token));
      }
    }
  }
  return !signal->empty();
}

bool WriteSignalCsv(const std::string& filepath,
                    const std::vector<float>& signal) {
  std::ofstream file(filepath);
  if (!file.is_open()) return false;

  file << std::fixed << std::setprecision(5);
  for (size_t i = 0; i < signal.size(); ++i) {
    if (i > 0) file << ",";
    file << signal[i];
  }
  file << "\n";
  return true;
}

void RunBatchGpuSignalProcessing(const Config& config) {
  int device_id = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDevice(&device_id));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

  std::cout << "=========================================================\n";
  std::cout << "CUDA Batch Signal Processing Pipeline\n";
  std::cout << "GPU Device: " << prop.name << " (Compute " << prop.major << "."
            << prop.minor << ")\n";
  std::cout << "Batch Size: " << config.num_files << " signal files\n";
  std::cout << "Signal Length: " << config.signal_length << " samples/file\n";
  std::cout << "Filter Window: " << config.filter_window << "-tap Moving Average\n";
  std::cout << "CUDA Streams: " << kNumStreams << " concurrent streams\n";
  std::cout << "=========================================================\n";

  size_t bytes_per_signal = config.signal_length * sizeof(float);

  cudaStream_t streams[kNumStreams];
  float* h_pinned_in[kNumStreams];
  float* h_pinned_out[kNumStreams];
  float* d_in[kNumStreams];
  float* d_out[kNumStreams];

  for (int s = 0; s < kNumStreams; ++s) {
    CHECK_CUDA(cudaStreamCreate(&streams[s]));
    CHECK_CUDA(cudaHostAlloc((void**)&h_pinned_in[s], bytes_per_signal,
                             cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&h_pinned_out[s], bytes_per_signal,
                             cudaHostAllocDefault));
    CHECK_CUDA(cudaMalloc((void**)&d_in[s], bytes_per_signal));
    CHECK_CUDA(cudaMalloc((void**)&d_out[s], bytes_per_signal));
  }

  cudaEvent_t start_event, stop_event;
  CHECK_CUDA(cudaEventCreate(&start_event));
  CHECK_CUDA(cudaEventCreate(&stop_event));
  CHECK_CUDA(cudaEventRecord(start_event, 0));

  int blocks_per_grid =
      (config.signal_length + kThreadsPerBlock - 1) / kThreadsPerBlock;
  int processed_count = 0;

  for (int base_idx = 0; base_idx < config.num_files; base_idx += kNumStreams) {
    int active_streams = 0;

    for (int s = 0; s < kNumStreams && (base_idx + s) < config.num_files; ++s) {
      int file_idx = base_idx + s;
      std::ostringstream in_path;
      in_path << config.input_dir << "/signal_" << std::setw(3)
              << std::setfill('0') << file_idx << ".csv";

      std::vector<float> temp_signal;
      if (!ReadSignalCsv(in_path.str(), &temp_signal)) {
        std::cerr << "Failed to read input file: " << in_path.str() << "\n";
        continue;
      }

      for (int i = 0; i < config.signal_length; ++i) {
        h_pinned_in[s][i] = temp_signal[i];
      }

      CHECK_CUDA(cudaMemcpyAsync(d_in[s], h_pinned_in[s], bytes_per_signal,
                                 cudaMemcpyHostToDevice, streams[s]));

      MovingAverageDenoiseKernel<<<blocks_per_grid, kThreadsPerBlock, 0,
                                   streams[s]>>>(
          d_in[s], d_out[s], config.signal_length, config.filter_window);

      CHECK_CUDA(cudaMemcpyAsync(h_pinned_out[s], d_out[s], bytes_per_signal,
                                 cudaMemcpyDeviceToHost, streams[s]));
      ++active_streams;
    }

    for (int s = 0; s < active_streams; ++s) {
      int file_idx = base_idx + s;
      CHECK_CUDA(cudaStreamSynchronize(streams[s]));

      std::vector<float> out_signal(config.signal_length);
      for (int i = 0; i < config.signal_length; ++i) {
        out_signal[i] = h_pinned_out[s][i];
      }

      std::ostringstream out_path;
      out_path << config.output_dir << "/denoised_signal_" << std::setw(3)
               << std::setfill('0') << file_idx << ".csv";
      WriteSignalCsv(out_path.str(), out_signal);
      ++processed_count;

      if (file_idx < 5 || file_idx % 25 == 0 ||
          file_idx == config.num_files - 1) {
        std::cout << "[Stream " << s << "] Processed file " << std::setw(3)
                  << (file_idx + 1) << "/" << config.num_files << ": "
                  << out_path.str() << " | Sample[0]: " << h_pinned_in[s][0]
                  << " -> " << h_pinned_out[s][0] << "\n";
      }
    }
  }

  CHECK_CUDA(cudaEventRecord(stop_event, 0));
  CHECK_CUDA(cudaEventSynchronize(stop_event));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));

  std::cout << "=========================================================\n";
  std::cout << "Batch Execution Complete!\n";
  std::cout << "Total Files Processed in Single Run: " << processed_count
            << " / " << config.num_files << "\n";
  std::cout << "Total Data Samples Filtered on GPU: "
            << (processed_count * config.signal_length) << "\n";
  std::cout << "Total GPU Pipeline Elapsed Time: " << elapsed_ms << " ms\n";
  std::cout << "=========================================================\n";

  CHECK_CUDA(cudaEventDestroy(start_event));
  CHECK_CUDA(cudaEventDestroy(stop_event));
  for (int s = 0; s < kNumStreams; ++s) {
    CHECK_CUDA(cudaFree(d_in[s]));
    CHECK_CUDA(cudaFree(d_out[s]));
    CHECK_CUDA(cudaFreeHost(h_pinned_in[s]));
    CHECK_CUDA(cudaFreeHost(h_pinned_out[s]));
    CHECK_CUDA(cudaStreamDestroy(streams[s]));
  }
}

int main(int argc, char** argv) {
  Config config = ParseCommandLineArgs(argc, argv);
  GenerateSyntheticSignals(config);
  RunBatchGpuSignalProcessing(config);
  return 0;
}
