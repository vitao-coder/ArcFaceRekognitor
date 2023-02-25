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

            return null;
        }
        public double CompareImage(byte[] imageBytes1, byte[]imageBytes2)
        {
            var bitmap1 = getBitmapFromBytes(imageBytes1);
            var bitmap2 = getBitmapFromBytes(imageBytes2);

            return CompareImage(bitmap1!, bitmap2!);
        }

        public double CompareImage(System.Drawing.Bitmap bitmap1, System.Drawing.Bitmap bitmap2)
        {
            Mat image1 = OpenCvSharp.Extensions.BitmapConverter.ToMat(bitmap1);
            Mat image2 = OpenCvSharp.Extensions.BitmapConverter.ToMat(bitmap2);

            List<PredictionBox> pbs1 = detector.Detect(image1, dete_threshold);

            List<PredictionBox> pbs2 = detector.Detect(image2, dete_threshold);

            float[] embedding1 = recognizer.Extract(image1, pbs1[0].Landmark);
            float[] embedding2 = recognizer.Extract(image2, pbs2[0].Landmark);

            return Compare(embedding1, embedding2);
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
