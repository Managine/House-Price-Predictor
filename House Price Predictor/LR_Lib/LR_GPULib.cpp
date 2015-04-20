// 这是主 DLL 文件。

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <ctime>

#include "LR_GPULib.h"

std::ofstream f("time.txt");

struct bi_function
{
	virtual float operator()(int tid, float l)=0;
};

struct un_function
{
	virtual float operator()(int tid)=0;
};

struct MeanFunctor : public un_function
{
	float * trainingData;
	unsigned int trainingDataCount;
	unsigned int featureCount;

	MeanFunctor(float * _trainingData, unsigned int _trainingDataCount, unsigned int _featureCount) : trainingData(_trainingData), trainingDataCount(_trainingDataCount), featureCount(_featureCount)
	{}

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
struct STDFunctor : public bi_function
{
	float * trainingData;
	unsigned int trainingDataCount;
	unsigned int featureCount;

	STDFunctor(float * _trainingData, unsigned int _trainingDataCount, unsigned int _featureCount ) : trainingData(_trainingData), trainingDataCount(_trainingDataCount), featureCount(_featureCount)
	{}

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
struct TrainFunctor : public bi_function
{
	float * trainingData;
	float * hypothesis;
	unsigned int featureCount;

	TrainFunctor(float * _trainingData, float * _hypothesis, int _featureCount) : trainingData(_trainingData), hypothesis(_hypothesis), featureCount(_featureCount)
	{}

  float operator()(int tid, float labelData)
  {
	  float res=0;
	  for (int i=0;i<featureCount;i++)
		  res+=hypothesis[i]*trainingData[tid*featureCount+i];
	  res-=labelData;
	  return res;
  }

  void update(float *_hypothesis)
  {
	  for (int i=0;i<featureCount;i++)
		  hypothesis[i]=_hypothesis[i];
  }
};

//
//Runs the first part of the training
//
struct TrainFunctor2 : public un_function
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
struct FeatureNormalizationgFunctor : public bi_function
{
	float * meanValue;
	float * stdValue;
	unsigned int featureCount;

	FeatureNormalizationgFunctor(float * _meanValue, float * _stdValue, unsigned int _featureCount) : meanValue(_meanValue), stdValue(_stdValue), featureCount(_featureCount)
	{}

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
struct PredictFunctor : public un_function
{
	float * testData;
	float * hypothesis;
	unsigned int featureCount;

	PredictFunctor(float * _testData, float * _hypothesis, unsigned int _featureCount) : testData(_testData), hypothesis(_hypothesis), featureCount(_featureCount)
	{}

  float operator()(int tid)
  {
	  float price=0;
	  for (int i=0;i<featureCount;i++)
		  price+=hypothesis[i]*testData[tid*featureCount+i];
	  return price;
  }
};

struct floatArray
{
	float *data;
	int length;
	int begin;
	int end;

	floatArray(int l)
	{
		length=l;
		data=new float[length];
		begin=0;
		end=length-1;
	}
	floatArray(floatArray &r)
	{
		length=r.length;
		data=new float[length];
		for (int i=0;i<length;i++)
			data[i]=r.data[i];
		begin=0;
		end=length-1;
	}
	floatArray(int l, int d)
	{
		length=l;
		data=new float[length];
		for (int i=0;i<length;i++)
			data[i]=0;
		begin=0;
		end=length-1;
	}
	floatArray(float *l, float *r)
	{
		length=r-l;
		data=new float[length];
		for (int i=0;i<length;i++)
			data[i]=l[i];
		begin=0;
		end=length-1;
	}
	floatArray(float *p, int l)
	{
		length=l;
		data=p;
		begin=0;
		end=length-1;
	}
	void copy(floatArray r)
	{
		length=r.length;
		data=new float[length];
		for (int i=0;i<length;i++)
			data[i]=r.data[i];
		begin=0;
		end=length-1;
	}
	floatArray& operator=(const floatArray& r)
	{
		length=r.length;
		begin=r.begin;
		end=r.end;
		data=new float[length];
		for (int i=0;i<length;i++)
			data[i]=r.data[i];
		return *this;
	}
};

void transform(int begin, int end, floatArray l, floatArray *r, struct bi_function &b)
{
	for (int i=begin;i<end;i++)
		r->data[i]=b(i,l.data[i]);
}

void transform(int begin, int end, floatArray *r, struct un_function &u)
{
	for (int i=begin;i<end;i++)
		r->data[i]=u(i);
}

void copy(floatArray l, float *r)
{
	for (int i=0;i<l.length;i++)
		r[i]=l.data[i];
}

float transform_reduce(int begin, int end, struct un_function &tf2, int t)
{
	float sum=0;
	for (int i=begin;i<end;i++)
		sum+=tf2(i);
	return sum;
}

//
//This method does mean normalization
//
void NormalizeFeaturesByMeanAndStd(unsigned int trainingDataCount, float* d_trainingData, struct floatArray dv_mean, struct floatArray dv_std)
{
	//Calculate mean norm: (x - mean) / std
	unsigned int featureCount = dv_mean.length;
	float * dvp_Mean = dv_mean.data;
	float * dvp_Std = dv_std.data;
	FeatureNormalizationgFunctor featureNormalizationgFunctor(dvp_Mean, dvp_Std, featureCount); 
	floatArray dvp_trainingData(d_trainingData, trainingDataCount*featureCount);

	transform(0, trainingDataCount * featureCount, dvp_trainingData, &dvp_trainingData, featureNormalizationgFunctor);
}

//
//This method calculates mean, standard deviation and does mean normalization
//
void NormalizeFeatures(unsigned int featureCount, unsigned int trainingDataCount, floatArray &d_trainingData, float * meanResult, float * stdResult)
{
	//Calculate the mean. One thread per feature.

	floatArray dv_mean(featureCount,0);
	MeanFunctor meanFunctor(d_trainingData.data, trainingDataCount, featureCount); 

	transform(0, featureCount, &dv_mean, meanFunctor);

	//Calculate the standard deviation. One thread per feature.
	floatArray dv_std(featureCount,0);
	STDFunctor stdFunctor(d_trainingData.data, trainingDataCount, featureCount); 
	transform(0, featureCount, dv_mean, &dv_std, stdFunctor);

	//Calculate mean norm: (x - mean) / std
	NormalizeFeaturesByMeanAndStd(trainingDataCount, d_trainingData.data, dv_mean, dv_std);

	copy(dv_mean, meanResult);
	copy(dv_std, stdResult);
}

void AddBiasTerm(float * inputData, float * outputData, int dataCount, int featureCount)
{
	//transfer the trainindata by adding also the bias term
	for(int i = 0; i < dataCount; i++)
	{
		outputData[i * featureCount] = 1;
		for(int f = 1; f < featureCount; f++)
		{
			outputData[i * featureCount + f] = inputData[(i * (featureCount - 1)) + (f-1)];
		}
	}
}

#define IsValidNumber(x)  (x == x && x <= DBL_MAX && x >= -DBL_MAX)

//
//Learn the hypothesis for the given data
//
extern int Learn(float* trainingData, float * labelData, unsigned int featureCount, unsigned int trainingDataCount, unsigned int gdIterationCount, float learningRate, float regularizationParam, float * result, float * meanResult, float * stdResult)
{
	clock_t start, end;
	start=clock();
	featureCount++;
	//allcate host memory
	floatArray hv_hypothesis(featureCount, 0);
	floatArray hv_trainingData(trainingDataCount * featureCount);
	floatArray hv_labelData(labelData, labelData + trainingDataCount);
	//transfer the trainindata by adding also the bias term
	AddBiasTerm(trainingData, hv_trainingData.data, trainingDataCount, featureCount);
	
	//allocate device vector
	floatArray dv_hypothesis = hv_hypothesis;
	floatArray dv_trainingData = hv_trainingData;
	floatArray dv_labelData = hv_labelData;
	floatArray dv_costData(trainingDataCount, 0);

	//Normalize the features
	NormalizeFeatures(featureCount, trainingDataCount, dv_trainingData, meanResult, stdResult);

	TrainFunctor tf(dv_trainingData.data, dv_hypothesis.data, featureCount); 
	TrainFunctor2 tf2(dv_costData.data, dv_trainingData.data, featureCount);
	//run gdIterationCount of gradient descent iterations
	for(int i = 0; i < gdIterationCount; i++)
	{
		tf.update(hv_hypothesis.data);
		transform(0, trainingDataCount,  dv_labelData, &dv_costData, tf);

		//calculate gradient descent iterations
		for(int featureNumber = 0; featureNumber < featureCount; featureNumber++) 
		{
			tf2.SetFeatureNumber(featureNumber);
			float totalCost = transform_reduce(0, trainingDataCount,  tf2, 0);

			if (!IsValidNumber(totalCost))
			{
				i = gdIterationCount;
				break;
			}
			float regularizationTerm = 1 - (learningRate * (regularizationParam / trainingDataCount));
			hv_hypothesis.data[featureNumber] = (hv_hypothesis.data[featureNumber] * regularizationTerm) -  learningRate * (totalCost / trainingDataCount);
		}
		
		//Copy the theta back to the device vector
		dv_hypothesis = hv_hypothesis;
	}

	//copy the hypothesis into the result buffer

	copy(hv_hypothesis, result);

	end=clock();
	f<<"Learn time:"<<end-start<<std::endl;
	return 0;
}

//
//makes prediction for the given test data based on the hypothesis. Also applies feature normalization.
//
extern int Predict(float* testData, unsigned int featureCount, unsigned int testDataCount, float* hypothesis, float * mean, float * std, float * result)
{
	clock_t start,end;
	start=clock();
	featureCount++;
	floatArray hv_testData(testDataCount * featureCount);
	AddBiasTerm(testData, hv_testData.data, testDataCount, featureCount);
	
	//Allocate device memory
	floatArray dv_hypothesis(hypothesis, hypothesis + featureCount);
	floatArray dv_testData = hv_testData;
	floatArray dv_result(testDataCount);
	floatArray dv_mean(mean, mean + featureCount);
	floatArray dv_std(std, std + featureCount);

	//Normalize features
	float * pdv_hypothesis = dv_hypothesis.data;
	float * pdv_testData = dv_testData.data;
	NormalizeFeaturesByMeanAndStd(testDataCount, pdv_testData, dv_mean, dv_std);

	//Predict
	PredictFunctor predictFunctor(pdv_testData, pdv_hypothesis, featureCount);
	transform(0, testDataCount, &dv_result, predictFunctor);

	//copy the result from device memory into the result buffer
	copy(dv_result, result);

	end=clock();
	f<<"Predict time:"<<end-start<<std::endl;
	return 0;
}