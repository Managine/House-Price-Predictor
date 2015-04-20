#include <thrust\functional.h>

//
//Calculates the mean for a column major ordered matrix
//
struct MeanFunctor : public thrust::unary_function<int, float>
{
	float * trainingData;
	unsigned int trainingDataCount;
	unsigned int featureCount;

	MeanFunctor(float * _trainingData, unsigned int _trainingDataCount, unsigned int _featureCount) : trainingData(_trainingData), trainingDataCount(_trainingDataCount), featureCount(_featureCount)
	{}

  __host__ __device__
  float operator()(int tid)
  {
	  float sum = 0;
	  for(int i = 0; i < trainingDataCount; i++)
		  sum += trainingData[featureCount * i + tid];
	  
	  return sum / trainingDataCount;
  }
};


//
//Calculates the standard deviation for a column major ordered matrix
//
struct STDFunctor : public thrust::binary_function<int, float, float>
{
	float * trainingData;
	unsigned int trainingDataCount;
	unsigned int featureCount;

	STDFunctor(float * _trainingData, unsigned int _trainingDataCount, unsigned int _featureCount ) : trainingData(_trainingData), trainingDataCount(_trainingDataCount), featureCount(_featureCount)
	{}

  __host__ __device__
  float operator()(int tid, float meanValue)
  {
	  float sum = 0;
	  for(int i = 0; i < trainingDataCount; i++)
		  sum += powf(trainingData[featureCount * i + tid] - meanValue, 2.0);
	  
	  return sqrtf(sum / (float)trainingDataCount);
  }
};

//
//Runs the first part of the training
//
struct TrainFunctor : public thrust::binary_function<int, float, float>
{
	float * trainingData;
	float * hypothesis;
	unsigned int featureCount;

	TrainFunctor(float * _trainingData, float * _hypothesis, int _featureCount) : trainingData(_trainingData), hypothesis(_hypothesis), featureCount(_featureCount)
	{}

  __host__ __device__
  float operator()(int tid, float labelData)
  {
	  float res=0;
	  for (int i=0;i<featureCount;i++)
		  res+=hypothesis[i]*trainingData[tid*featureCount+i];
	  res-=labelData;
	  return res;
  }
};

//
//Runs the first part of the training
//
struct TrainFunctor2 : public thrust::unary_function<int, float>
{
	float * trainingData;
	float * costData;
	unsigned int featureNumber;
	unsigned int featureCount;

	TrainFunctor2(float * _costData, float * _trainingData, unsigned int _featureCount) : costData(_costData), trainingData(_trainingData), featureCount(_featureCount), featureNumber(0)
	{}

	void SetFeatureNumber(unsigned int value)
	{
		featureNumber = value;
	}

  __host__ __device__
  float operator()(int tid)
  {
	  //if (trainingData[tid * featureCount + featureNumber]==0)
		 // return 0;
	  return costData[tid] * trainingData[tid * featureCount + featureNumber];
  }
};
//
//Applies feature normalization algorithm to the data. (data - mean) / standard deviation
//
struct FeatureNormalizationgFunctor : public thrust::binary_function<int, float, float>
{
	float * meanValue;
	float * stdValue;
	unsigned int featureCount;

	FeatureNormalizationgFunctor(float * _meanValue, float * _stdValue, unsigned int _featureCount) : meanValue(_meanValue), stdValue(_stdValue), featureCount(_featureCount)
	{}

  __host__ __device__
  float operator()(int tid, float trainingData)
  {
	  if (stdValue[tid%featureCount]==0)
		  return 1;
	  return (trainingData-meanValue[tid%featureCount])/stdValue[tid%featureCount];
  }
};

//
//Applies the hypothesis to the test data
//
struct PredictFunctor : public thrust::unary_function<int, float>
{
	float * testData;
	float * hypothesis;
	unsigned int featureCount;

	PredictFunctor(float * _testData, float * _hypothesis, unsigned int _featureCount) : testData(_testData), hypothesis(_hypothesis), featureCount(_featureCount)
	{}

  __host__ __device__
  float operator()(int tid)
  {
	  float price;
	  for (int i=0;i<featureCount;i++)
		  price+=hypothesis[i]*testData[tid*featureCount+i];
	  return price;
  }
};