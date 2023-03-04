using OpenCvSharp;
using ArcFaceRekognitor.Api.Models;
using System.Drawing;

namespace ArcFaceRekognitor.Api.FaceRecognition
{
    public class FaceRecognize
    {
        private SCRFD detector;
        private ArcFace recognizer;
        private readonly float reco_threshold;
        private readonly float dete_threshold;
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
            }
        }

        private Bitmap? getBitmapFromBytes(byte[] image)
        {
            using (var ms = new MemoryStream(image))
            {
                return new Bitmap(ms);
            }
        }

        public async Task<double> CompareImage(byte[] imageBytes1, byte[]imageBytes2)
        {
            var TaskBitmap1 = Task.Run(() =>
            {
                var bitmap1 = getBitmapFromBytes(imageBytes1);
                return bitmap1;
            });

            var TaskBitmap2 = Task.Run(() =>
            {
                var bitmap2 = getBitmapFromBytes(imageBytes2);
                return bitmap2;
            });

            var bitmap1 = await TaskBitmap1;
            var bitmap2 = await TaskBitmap2;
            TaskBitmap1.Dispose();
            TaskBitmap2.Dispose();
            return await CompareImage(bitmap1!, bitmap2!);
        }

        public async Task<double> CompareImage(System.Drawing.Bitmap bitmap1, System.Drawing.Bitmap bitmap2)
        {
            Mat image1 = OpenCvSharp.Extensions.BitmapConverter.ToMat(bitmap1);
            Mat image2 = OpenCvSharp.Extensions.BitmapConverter.ToMat(bitmap2);
            bitmap1.Dispose();
            bitmap2.Dispose();


            var TaskDetectPb1 = Task.Run(() =>
            {
                List<PredictionBox> pbs1 = detector.Detect(image1, dete_threshold);
                return pbs1;
            });

            var TaskDetectPb2 = Task.Run(() =>
            {
                List<PredictionBox> pbs2 = detector.Detect(image2, dete_threshold);
                return pbs2;
            });


            var TaskExtractPb1 = Task.Run(async () =>
            {
                var pbs1 = await TaskDetectPb1;
                float[] embedding1 = recognizer.Extract(image1, pbs1[0].Landmark);
                return embedding1;
            });


            var TaskExtractPb2 = Task.Run(async () =>
            {
                var pbs2 = await TaskDetectPb2;
                float[] embedding2 = recognizer.Extract(image2, pbs2[0].Landmark);
                return embedding2;
            });


            float[] embedding1 = await TaskExtractPb1;
            float[] embedding2 = await TaskExtractPb2;
            image1.Dispose(); 
            image2.Dispose();
            TaskDetectPb1.Dispose();
            TaskDetectPb2.Dispose();
            TaskExtractPb1.Dispose();
            TaskExtractPb2.Dispose();

            return await CompareImage(embedding1, embedding2);
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
            var bitmap = getBitmapFromBytes(imageBytes);

            return await DetectImage(bitmap);
        }

        public async Task<PredictionBox> DetectImage(System.Drawing.Bitmap bitmap)
        {
            var taskDetect = Task.Run(() => {

                Mat image = OpenCvSharp.Extensions.BitmapConverter.ToMat(bitmap);
                bitmap.Dispose();

                var detection = detector.Detect(image, dete_threshold);
                if (detection.Count > 1) throw new InvalidOperationException("More than one face detected");
                if (detection.Count == 0) throw new InvalidOperationException("Not detected any face");

                var detected = detection[0];
                var embedding = recognizer.Extract(image, detected.Landmark);
                detected.Landmark = embedding;
                image.Dispose();                
                return detected;
            });

            return await taskDetect;
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
            return results;
        }

        static public double Compare(float[] faceA, float[] faceB)
        {
            double result = 0;
            for (int i = 0; i < faceA.Length; i++)
                result = result + Math.Pow(faceA[i] - faceB[i], 2);
            return result;
        }
    }
}

