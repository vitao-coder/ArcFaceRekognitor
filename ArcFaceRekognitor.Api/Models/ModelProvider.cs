﻿using Microsoft.ML.OnnxRuntime;
using SessionOptions = Microsoft.ML.OnnxRuntime.SessionOptions;

namespace ArcFaceRekognitor.Api.Models
{
    public class ModelProvider : IDisposable
    {
        InferenceSession _sessionSCRFD;
        InferenceSession _sessionBuffallo;

        public enum ModelType
        {
            ArcFace,
            ResNet,
            ScRfd,
            Buffalo,
        }

        public ModelProvider()
        {
            SessionOptions options = new();
            options.InterOpNumThreads = 2;
            options.IntraOpNumThreads = 2;
            options.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            options.LogVerbosityLevel = (int)Microsoft.ML.OnnxRuntime.LogLevel.Error;
            options.AppendExecutionProvider_CPU(0);                  

            var modelSCRFD = "./OriginalModel/scrfd_10g_bnkps_shape640x640OnnxV6.onnx";
            byte[] modelSCRFDBytes = File.ReadAllBytes(modelSCRFD);
            _sessionSCRFD = new InferenceSession(modelSCRFDBytes, options);

            var modelBuffallo = "./BuffaloModel/w600k_r50.onnx";
            byte[] modelBuffalloBytes = File.ReadAllBytes(modelBuffallo);
            _sessionBuffallo = new InferenceSession(modelBuffalloBytes, options);
        }

        public InferenceSession? GetSession(ModelType modelType)
        {
            switch (modelType)
            {
                case ModelType.ScRfd:
                    return _sessionSCRFD;
                case ModelType.Buffalo:
                    return _sessionBuffallo;
            }
            return null;
        }

        public void Dispose()
        {
            _sessionSCRFD.Dispose();
            _sessionBuffallo.Dispose();
        }
    }
}
