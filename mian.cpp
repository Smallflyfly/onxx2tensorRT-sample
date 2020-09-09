#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
//using namespace samplesCommon;
//using namespace nvinfer1;
//using namespace nvonnxparser;

const string gsampleNmae = "mask_onxx";

class SampleOnxx
{
	template <typename T>
	using SampleUniquePtr = unique_ptr<T, samplesCommon::InferDeleter>;

public:
	SampleOnxx(const samplesCommon::OnnxSampleParams& params)
		: mParams(params)
		, mEngine(nullptr)
	{
	}
	// brief Function builds the network engine
	bool build();

	// brief Runs the TensorRT inference engine for this sample
	bool infer();

private:
	samplesCommon::OnnxSampleParams mParams; // parameters for the samples
	nvinfer1::Dims mInputDims; //input dimensions of input to the network
	nvinfer1::Dims mOutputDims; //output dimensions of output to the network
	int mNumber{ 0 };

	shared_ptr<nvinfer1::ICudaEngine> mEngine; // tensorRT engine used to run the network

	// brief Parses an ONNX model for MNIST and creates a TensorRT network
	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
		SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
		SampleUniquePtr<nvonnxparser::IParser>& parser);

	// brief Reads the input  and stores the result in a managed buffer
	bool processInput(const samplesCommon::BufferManager& buffers);

	// brief Classifies digits and verify result
	bool verifyOutput(const samplesCommon::BufferManager& buffers);



};

samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
	samplesCommon::OnnxSampleParams params;
	if (args.dataDirs.empty())
	{
		params.dataDirs.push_back("data/mnist/");
		params.dataDirs.push_back("data/samples/mnist/");
	}
	else
	{
		params.dataDirs = args.dataDirs;
	}
	params.onnxFileName = "mnist.onnx";
	params.inputTensorNames.push_back("Input3");
	params.batchSize = 1;
	params.outputTensorNames.push_back("Plus214_Output_0");
	params.dlaCore = args.useDLACore;
	params.int8 = args.runInInt8;
	params.fp16 = args.runInFp16;

	return params;
}

int main(int argc, char** argv)
{
	//cout << "fang" << endl;
	samplesCommon::Args args;
	bool argsOk = parseArgs(args, argc, argv);
	auto sampleTest = gLogger.defineTest(gsampleNmae, argc, argv);
	gLogger.reportTestStart(sampleTest);
	
	SampleOnxx sample(initializeSampleParams(args));
	gLogInfo << "Run it" << endl;

	if (!sample.build())
	{
		return gLogger.reportFail(sampleTest);
	}
	if (!sample.infer())
	{
		return gLogger.reportFail(sampleTest);
	}

	return gLogger.reportPass(sampleTest);
	//return 0;
}

bool SampleOnxx::build()
{
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
	if (!builder)
	{
		return false;
	}

	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		return false;
	}

	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		return false;
	}

	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
	if (!parser)
	{
		return false;
	}

	auto constructed = constructNetwork(builder, network, config, parser);
	if (!constructed)
	{
		return false;
	}

	mEngine = shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
	if (!mEngine)
	{
		return false;
	}

	assert(network->getNbInputs() == 1);
	mInputDims = network->getInput(0)->getDimensions();
	assert(mInputDims.nbDims == 4);

	assert(network->getNbOutputs() == 1);
	mOutputDims = network->getOutput(0)->getDimensions();
	assert(mOutputDims.nbDims == 2);
	
	return true;
}

bool SampleOnxx::processInput(const samplesCommon::BufferManager& buffers)
{
	const int inputH = mInputDims.d[2];
	const int inputW = mInputDims.d[3];

	srand(unsigned(time(nullptr)));
	vector<uint8_t> fileData(inputH * inputW);
	mNumber = rand() % 10;
	readPGMFile(locateFile(to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	for (int i = 0; i < inputH * inputW; i++)
	{
		hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
	}

	return true;
}

bool SampleOnxx::infer()
{
	samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context)
	{
		return false;
	}

	assert(mParams.inputTensorNames.size() == 1);
	if (!processInput(buffers))
	{
		return false;
	}

	//Memcpy from host input buffers to device input buffers
	buffers.copyInputToDevice();

	bool status = context->executeV2(buffers.getDeviceBindings().data());
	if (!status)
	{
		return false;
	}

	// Memcpy from device output buffers to host output buffers
	buffers.copyOutputToHost();

	if (!verifyOutput(buffers))
	{
		return false;
	}

	return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnxx::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser)
{
	auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()));
	if (!parsed)
	{
		return false;
	}
	builder->setMaxBatchSize(mParams.batchSize);
	config->setMaxWorkspaceSize(16_MiB);
	if (mParams.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (mParams.int8)
	{
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
	}
	samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

	return true;
}

bool SampleOnxx::verifyOutput(const samplesCommon::BufferManager& buffers)
{
	const int outputSize = mOutputDims.d[1];
	float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
	float val{ 0.0f };
	int idx{ 0 };

	// Calculate Softmax
	float sum{ 0.0f };
	for (int i = 0; i < outputSize; i++)
	{
		output[i] = exp(output[i]);
		sum += output[i];
	}

	for (int i = 0; i < outputSize; i++)
	{
		output[i] /= sum;
		val = max(val, output[i]);
		if (val == output[i])
		{
			idx = i;
		}
		gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
			<< "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
	}
	gLogInfo << std::endl;

	return idx == mNumber && val >= 0.9f;
}
