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

        public override async Task<DetectorReply> Detector(DetectorRequest request, ServerCallContext context)
        {   
            try
            {
                var detection = await _faceRecognize.DetectImage(request.ImageBytes.ToArray());

                return new DetectorReply()
                {
                    Score = detection.Score,
                    BoxLeft= detection.BoxLeft,
                    BoxRight = detection.BoxRight,
                    BoxTop = detection.BoxTop,
                    BoxBottom = detection.BoxBottom,
                    Landmark = { detection.Landmark }
                };
            }
            catch(Exception ex)
            {
                _logger.LogError(ex, "Error in Detector");
                return new DetectorReply()
                {
                    Error = ex.Message
                };                
            }
        }

        public override async Task<ComparatorReply> Comparator(ComparatorRequest request, ServerCallContext context)
        {
            try
            {
                var score = await _faceRecognize.CompareImage(request.ImageBytes1.ToArray(), request.ImageBytes2.ToArray());

                return new ComparatorReply()
                {
                    Score = score,                    
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in Comparator");
                return new ComparatorReply()
                {
                    Error = ex.Message
                };             
            }
        }
    }
}

