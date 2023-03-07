using OpenCvSharp;
using ArcFaceRekognitor.Api.Models;
using System.Drawing;
using System.IO;

namespace ArcFaceRekognitor.Api.FaceRecognition
{
    public class FaceRecognize
    {
        private SCRFD detector;
        private ArcFace recognizer;
        public readonly float reco_threshold;
        public readonly float dete_threshold;
        private readonly ModelProvider _modelprovider;

        private Dictionary<string, float[]> faces_embedding;

        public FaceRecognize(ModelProvider modelprovider, int ctx_id = 0, float dete_threshold = 0.50f, float reco_threshold = 1.24f)
        {
            _modelprovider = modelprovider;
            var detectorSession = _modelprovider.GetSession(ModelProvider.ModelType.ScRfd);
            detector = new SCRFD(detectorSession!);

            var recognizerSession = _modelprovider.GetSession(ModelProvider.ModelType.Buffalo);

            recognizer = new ArcFace(recognizerSession!);

            faces_embedding = new Dictionary<string, float[]>();

            this.reco_threshold = reco_threshold;
            this.dete_threshold = dete_threshold;
        }

        public void LoadFaces(string face_db_path)
        {
            if (!Directory.Exists(face_db_path))
                Directory.CreateDirectory(face_db_path);
            foreach (string fileName in Directory.GetFiles(face_db_path))
            {
                Mat image = Cv2.ImRead(fileName);
                Register(image, System.IO.Path.GetFileNameWithoutExtension(fileName));
                image.Release();
            }
        }

        public struct ResultCompare
        {
            public double Score;
            public float[] embedding1;
            public float[] embedding2;
        }

        public async Task<ResultCompare> CompareImage(byte[] imageBytes1, byte[] imageBytes2)
        {
            try
            {
                Mat image1;
                Mat image2;

                var TaskMat1 = Task.Factory.StartNew(() =>
                {
                    using (var ms1 = new MemoryStream(imageBytes1))
                    {
                        return Mat.FromStream(ms1, ImreadModes.Unchanged);
                    }
                });
                var TaskMat2 = Task.Factory.StartNew(() =>
                {
                    using (var ms2 = new MemoryStream(imageBytes2))
                    {
                        return Mat.FromStream(ms2, ImreadModes.Unchanged);
                    }
                });
                image1 = await TaskMat1;
                image2 = await TaskMat2;

                var TaskDetectPb1 = Task.Factory.StartNew(() =>
                {
                    return detector.Detect(image1, dete_threshold);
                });

                var TaskDetectPb2 = Task.Factory.StartNew(() =>
                {
                    return detector.Detect(image2, dete_threshold);
                });


                var TaskExtractPb1 = Task.Factory.StartNew(async () =>
                {
                    var pbs1 = await TaskDetectPb1;
                    float[] embedding1 = recognizer.Extract(image1, pbs1[0].Landmark);
                    return embedding1;
                }).Unwrap();


                var TaskExtractPb2 = Task.Factory.StartNew(async () =>
                {
                    var pbs2 = await TaskDetectPb2;
                    float[] embedding2 = recognizer.Extract(image2, pbs2[0].Landmark);
                    return embedding2;
                }).Unwrap();

                float[] embedding1 = await TaskExtractPb1;
                float[] embedding2 = await TaskExtractPb2;
                image1.Release();
                image2.Release();
                TaskDetectPb1.Dispose();
                TaskDetectPb2.Dispose();
                TaskExtractPb1.Dispose();
                TaskExtractPb2.Dispose();

                var result = new ResultCompare
                {
                    Score = await CompareImage(embedding1, embedding2),
                    embedding1 = embedding1,
                    embedding2 = embedding2
                };

                return result;
            }
            catch (Exception)
            {
                throw;
            }
            finally
            {
                GC.Collect();
            }
        }

        public async Task<double> CompareImage(float[] embeddingImage1, float[] embeddingImage2)
        {
            var TaskCompare = Task.Run(() =>
            {
                var resultCompare = Compare(embeddingImage1, embeddingImage2);
                return resultCompare;
            });

            return await TaskCompare;
        }

        public int Register(Mat image, string user_id)
        {
            List<PredictionBox> pbs = detector.Detect(image, dete_threshold);
            if (pbs.Count == 0)
                return 1;
            if (pbs.Count > 1)
                return 2;

            float[] embedding = recognizer.Extract(image, pbs[0].Landmark);
            foreach (float[] face in faces_embedding.Values)
                if (Compare(embedding, face) < reco_threshold)
                    return 3;
            faces_embedding.Add(user_id, embedding);
            return 0;
        }

        public async Task<PredictionBox> DetectImage(byte[] imageBytes)
        {
            try
            {
                var taskDetect = Task.Factory.StartNew(async () =>
                {

                    var TaskMat1 = Task.Factory.StartNew(() =>
                    {
                        using (var ms1 = new MemoryStream(imageBytes))
                        {
                            return Mat.FromStream(ms1, ImreadModes.Unchanged);
                        }
                    });

                    Mat image = await TaskMat1;

                    var TaskDetect = Task.Factory.StartNew(() =>
                    {
                        return detector.Detect(image, dete_threshold);
                    });

                    var detection = await TaskDetect;
                    if (detection.Count > 1) throw new InvalidOperationException("More than one face detected");
                    if (detection.Count == 0) throw new InvalidOperationException("Not detected any face");

                    var TaskExtract = Task.Factory.StartNew(async () =>
                    {
                        float[] embedding1 = recognizer.Extract(image, detection[0].Landmark);
                        return embedding1;
                    });

                    var detected = detection[0];
                    var embedding = await TaskExtract.Unwrap();
                    detected.Landmark = embedding;
                    image.Release();
                    TaskMat1.Dispose();
                    TaskDetect.Dispose();
                    TaskExtract.Dispose();
                    return detected;
                });

                return await taskDetect.Unwrap();
            }
            catch (Exception)
            {
                throw;
            }
            finally
            {
                GC.Collect();
            }
        }               

        public Dictionary<string, PredictionBox> Recognize(System.Drawing.Bitmap bitmap)
        {
            Mat image = OpenCvSharp.Extensions.BitmapConverter.ToMat(bitmap);
            Dictionary<string, PredictionBox> results = new Dictionary<string, PredictionBox>();
            List<PredictionBox> pbs = detector.Detect(image, dete_threshold);
            foreach (PredictionBox pb in pbs)
            {
                float[] embedding = recognizer.Extract(image, pb.Landmark);
                foreach (string user_id in faces_embedding.Keys)
                {
                    float[] face = faces_embedding[user_id];
                    if (Compare(embedding, face) < reco_threshold)
                        results.Add(user_id, pb);
                }
            }
            image.Release();
            return results;
        }

        static public double Compare(float[] faceA, float[] faceB)
        {
            double result = 0;
            for (int i = 0; i < faceA.Length; i++)
                result = result + System.Math.Pow(faceA[i] - faceB[i], 2);
            return result;
        }
    }
}

