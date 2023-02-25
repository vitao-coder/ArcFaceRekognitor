using ArcFaceRekognitor.Api.FaceRecognition;
using Grpc.Core;

namespace ArcFaceRekognitor.Api.Services
{
    public class InferenceService : Inference.InferenceBase
    {
        private readonly ILogger<InferenceService> _logger;
        private readonly FaceRecognize _faceRecognize;      

        public InferenceService(ILogger<InferenceService> logger, FaceRecognize faceRecognize)
        {
            _faceRecognize = faceRecognize;
            _logger = logger;           
        }

        public override Task<DetectionReply> Detector(DetectionRequest request, ServerCallContext context)
        {   
            try
            {
                var score = _faceRecognize.CompareImage(request.ImageBytes1.ToArray(), request.ImageBytes2.ToArray());

                return Task.FromResult(new DetectionReply()
                {
                    Message = "Score:" + score.ToString()
                });
            }
            catch(Exception ex)
            {
                return Task.FromResult(new DetectionReply()
                {
                    Message = "Error:" + ex.Message.ToString()
                });
            }
        }
    }
}

