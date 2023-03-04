using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;


namespace ArcFaceRekognitor.Api.FaceRecognition
{
    public class ArcFace
    {
        private InferenceSession _onnxSession;
        private readonly string input_name;
        private readonly int[] input_dimensions;

        public ArcFace(InferenceSession session)
        {
            _onnxSession = session;

            IReadOnlyDictionary<string, NodeMetadata> _onnx_inputs = _onnxSession.InputMetadata;
            input_name = _onnx_inputs.Keys.ToList()[0];
            input_dimensions = _onnx_inputs.Values.ToList()[0].Dimensions;
        }

        public float[] Extract(Mat image, float[] landmarks)
        { 
            image = FaceAlign.Align(image, landmarks, 112, 112);
            Tensor<float> input_tensor = new DenseTensor<float>(new[] { 1, input_dimensions[1], input_dimensions[2], input_dimensions[3] });

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Vec3b color = image.Get<Vec3b>(y, x);
                    input_tensor[0, 0, y, x] = (color.Item2 - 127.5f) / 127.5f;
                    input_tensor[0, 1, y, x] = (color.Item1 - 127.5f) / 127.5f;
                    input_tensor[0, 2, y, x] = (color.Item0 - 127.5f) / 127.5f;                    
                }
            }

            var container = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(input_name, input_tensor)
            };
            

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(container);
            
            var resultsArray = results.ToArray();
        
            DisposableNamedOnnxValue nov = resultsArray[0];
            Tensor<float> tensor = nov.AsTensor<float>();
            float[] embedding = new float[tensor.Length];

            double l2 = 0;
            for (int i = 0; i < tensor.Length; i++)
            {
                embedding[i] = tensor[0, i];
                l2 = l2 + Math.Pow((double)tensor[0, i], 2);
            }
            l2 = Math.Sqrt(l2);

            for (int i = 0; i < embedding.Length; i++)
                embedding[i] = embedding[i] / (float)l2;
 
            results.Dispose();

            foreach (var item in resultsArray)
            {
                item.Dispose();
            }
            nov.Dispose();
            
            return embedding;
        }
            
    }
}
