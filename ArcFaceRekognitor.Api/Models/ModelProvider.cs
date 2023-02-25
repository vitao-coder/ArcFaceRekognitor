using Microsoft.ML.OnnxRuntime;
using SessionOptions = Microsoft.ML.OnnxRuntime.SessionOptions;

namespace ArcFaceRekognitor.Api.Models
{
    public class ModelProvider : IDisposable
    {
        Dictionary<ModelType, InferenceSession> models = new Dictionary<ModelType, InferenceSession>();

        public enum ModelType
        {
            ArcFace,
            ResNet,
            ScRfd,
            Buffalo,
        }

        public ModelProvider(int ctx_id = 0)
        {
            SessionOptions options = new SessionOptions();
            options.LogVerbosityLevel = Microsoft.ML.OnnxRuntime.LogLevel.Error;
            options.AppendExecutionProvider_CPU(0);

            var modelSCRFD = "./OriginalModel/scrfd_10g_bnkps_shape640x640OnnxV6.onnx";
            byte[] modelSCRFDBytes = File.ReadAllBytes(modelSCRFD);
            var sessionSCRFD = new InferenceSession(modelSCRFDBytes, options);
            models.Add(ModelType.ScRfd, sessionSCRFD);

            var modelBuffallo = "./BuffaloModel/w600k_r50.onnx";
            byte[] modelBuffalloBytes = File.ReadAllBytes(modelBuffallo);
            var sessionBuffallo = new InferenceSession(modelBuffalloBytes, options);
            models.Add(ModelType.Buffalo, sessionBuffallo);

        }

        public InferenceSession? GetSession(ModelType modelType)
        {
            if (models.TryGetValue(modelType, out var session)) return session;
            return null;
        }

        public void Dispose()
        {
            foreach (var model in models)
            {
                model.Value.Dispose();
            }
        }
    }
}
