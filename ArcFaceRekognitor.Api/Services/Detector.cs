using ArcFaceRekognitor.Api.FaceRecognition;
using Grpc.Core;

namespace ArcFaceRekognitor.Api.Services
{
    public class DetectorService : Detect.DetectBase, IDisposable
    {
        private readonly ILogger<DetectorService> _logger;
        private readonly FaceRecognize _faceRecognize;

        public DetectorService(ILogger<DetectorService> logger, FaceRecognize faceRecognize)
        {
            _faceRecognize = faceRecognize;
            _logger = logger;
        }

        public override Task<DetectorReply> Detector(DetectorRequest request, ServerCallContext context)
        {
            try
            {
                var detectionResponse = _faceRecognize.DetectImage(request.ImageBytes.ToArray());
                detectionResponse.Wait();

                var detection = detectionResponse.Result;


                return Task.FromResult(new DetectorReply()
                {
                    Score = detection.Score,
                    BoxLeft = detection.BoxLeft,
                    BoxRight = detection.BoxRight,
                    BoxTop = detection.BoxTop,
                    BoxBottom = detection.BoxBottom,
                    Landmark = { detection.Landmark }
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in Detector");
                return Task.FromResult(new DetectorReply() { Error = ex.Message });
            }
            finally
            {
                GC.Collect();
            }
        }

        public void Dispose()
        {
            GC.Collect();
        }
    }
}
