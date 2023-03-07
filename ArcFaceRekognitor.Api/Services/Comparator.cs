using ArcFaceRekognitor.Api.FaceRecognition;
using Grpc.Core;


namespace ArcFaceRekognitor.Api.Services
{
    using Similarity;
    public class ComparatorService : Compare.CompareBase, IDisposable
    {
        private readonly ILogger<ComparatorService> _logger;
        private readonly FaceRecognize _faceRecognize;

        public ComparatorService(ILogger<ComparatorService> logger, FaceRecognize faceRecognize)
        {
            _faceRecognize = faceRecognize;
            _logger = logger;
        }

        public override Task<ComparatorReply> Comparator(ComparatorRequest request, ServerCallContext context)
        {
            try
            {
                var response = _faceRecognize.CompareImage(request.ImageBytes1.ToArray(), request.ImageBytes2.ToArray());
                response.Wait();

                var result = response.Result;

                return Task.FromResult(new ComparatorReply()
                {
                    Score = result.Score,
                    IsSame = result.Score <= _faceRecognize.reco_threshold,
                    CosineDistance = Similarity.CosineSimilarity(result.embedding1, result.embedding2),
                    EuclideanDistance = Similarity.EuclideanDistance(result.embedding1, result.embedding2),
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in Comparator");
                return Task.FromResult(new ComparatorReply()
                {
                    Error = ex.Message
                });
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

