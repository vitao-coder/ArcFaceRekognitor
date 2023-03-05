using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using Point = OpenCvSharp.Point;
using SessionOptions = Microsoft.ML.OnnxRuntime.SessionOptions;

namespace ArcFaceRekognitor.Api.FaceRecognition
{
    public partial class SCRFD
    {
        private readonly int[] feat_stride_fpn = new int[] { 8, 16, 32 };
        private readonly float nms_threshold = 0.4f;

        private Dictionary<string, List<Point>> anchor_centers;
        private InferenceSession _onnxSession;
        private readonly string input_name;
        private readonly int[] input_dimensions;

        public SCRFD(InferenceSession session)
        {
            anchor_centers = new Dictionary<string, List<Point>>();

            _onnxSession = session;

            IReadOnlyDictionary<string, NodeMetadata> _onnx_inputs = _onnxSession.InputMetadata;
            input_name = _onnx_inputs.Keys.ToList()[0];
            input_dimensions = _onnx_inputs.Values.ToList()[0].Dimensions;
        }

        public List<PredictionBox> Detect(Mat image, float dete_threshold = 0.5f)
        {
            int iWidth = image.Width;
            int iHeight = image.Height;

            int height = input_dimensions[2];
            int width = input_dimensions[3];

            float rate = width / (float)height;
            float iRate = iWidth / (float)iHeight;

            if (rate > iRate)
            {
                iWidth = (int)(height * iRate);
                iHeight = height;
            }
            else
            {
                iWidth = width;
                iHeight = (int)(width / iRate);
            }

            float resize_Rate = image.Size().Height / (float)iHeight;

            Tensor<float> input_tensor = new DenseTensor<float>(new[] { input_dimensions[0], input_dimensions[1], height, width });

            OpenCvSharp.Mat dst = new Mat();
            Cv2.Resize(image, dst, new OpenCvSharp.Size(iWidth, iHeight));

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    if (y < dst.Height && x < dst.Width)
                    {
                        Vec3b color = dst.Get<Vec3b>(y, x);
                        input_tensor[0, 0, y, x] = (color.Item2 - 127.5f) / 128f;
                        input_tensor[0, 1, y, x] = (color.Item1 - 127.5f) / 128f;
                        input_tensor[0, 2, y, x] = (color.Item0 - 127.5f) / 128f;
                    }
                    else
                    {
                        input_tensor[0, 0, y, x] = (0 - 127.5f) / 128f;
                        input_tensor[0, 1, y, x] = (0 - 127.5f) / 128f;
                        input_tensor[0, 2, y, x] = (0 - 127.5f) / 128f;
                    }

            var container = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(input_name, input_tensor)
            };


            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(container);

            var resultsArray = results.ToArray();           

            List<PredictionBox> preds = new List<PredictionBox>();

            for (int idx = 0; idx < feat_stride_fpn.Length; idx++)
            {
                int stride = feat_stride_fpn[idx];

                DisposableNamedOnnxValue scores_nov = resultsArray[idx];
                Tensor<float> scores_tensor = scores_nov.AsTensor<float>();
                Tensor<float> bboxs_tensor = resultsArray[idx + 3 * 1].AsTensor<float>();
                Tensor<float> kpss_tensor = resultsArray[idx + 3 * 2].AsTensor<float>();

                int sHeight = height / stride, sWidth = width / stride;
                
                for (int i = 0; i < scores_tensor.Dimensions[1]; i++)
                {                  
                    float score = scores_tensor[0, i, 0];
                    if (score <= dete_threshold)
                        continue;
                    string key_anchor_center = String.Format("{0}-{1}-{2}", sHeight, sWidth, stride);

                    List<Point> anchor_center = null;
                    try
                    {
                        anchor_center = anchor_centers[key_anchor_center];
                    }
                    catch (KeyNotFoundException)
                    {
                        anchor_center = new List<Point>();                        
                        for (int h = 0; h < sHeight; h++)
                            for (int w = 0; w < sWidth; w++)
                            {
                                anchor_center.Add(new Point(w * stride, h * stride));
                                anchor_center.Add(new Point(w * stride, h * stride));
                            }
                        anchor_centers.Add(key_anchor_center, anchor_center);
                    }

                    float[] box = new float[bboxs_tensor.Dimensions[2]];
                    for (int b = 0; b < bboxs_tensor.Dimensions[2]; b++)
                        box[b] = bboxs_tensor[0, i, b] * stride;
                    float[] kps = new float[kpss_tensor.Dimensions[2]];
                    for (int k = 0; k < kpss_tensor.Dimensions[2]; k++)
                        kps[k] = kpss_tensor[0, i, k] * stride;

                    box = Distance2Box(box, anchor_center[i], resize_Rate);
                    kps = Distance2Point(kps, anchor_center[i], resize_Rate);

                    preds.Add(new PredictionBox(score, box[0], box[1], box[2], box[3], kps));
                }
                scores_nov.Dispose();
            }

            dst.Release();
            results.Dispose();
            foreach (var item in resultsArray)
            {
                item.Dispose();
            }
            if (preds.Count == 0)
                return preds;

            preds = preds.OrderByDescending(a => a.Score).ToList();
            return NMS(preds, nms_threshold);
        }

        public static List<PredictionBox> NMS(List<PredictionBox> predictions, float nms_threshold)
        {
            List<PredictionBox> final_predications = new List<PredictionBox>();

            while (predictions.Count > 0)
            {
                PredictionBox pb = predictions[0];
                predictions.RemoveAt(0);
                final_predications.Add(pb);

                int idx = 0;
                while (idx < predictions.Count)
                {
                    if (ComputeIOU(pb, predictions[idx]) > nms_threshold)
                        predictions.RemoveAt(idx);
                    else
                        idx++;
                }
            }

            return final_predications;
        }

        private static float ComputeIOU(PredictionBox PB1, PredictionBox PB2)
        {
            float ay1 = PB1.BoxTop;
            float ax1 = PB1.BoxLeft;
            float ay2 = PB1.BoxBottom;
            float ax2 = PB1.BoxRight;
            float by1 = PB2.BoxTop;
            float bx1 = PB2.BoxLeft;
            float by2 = PB2.BoxBottom;
            float bx2 = PB2.BoxRight;

            float x_left = Math.Max(ax1, bx1);
            float y_top = Math.Max(ay1, by1);
            float x_right = Math.Min(ax2, bx2);
            float y_bottom = Math.Min(ay2, by2);

            if (x_right < x_left || y_bottom < y_top)
                return 0;

            float intersection_area = (x_right - x_left) * (y_bottom - y_top);
            float bb1_area = (ax2 - ax1) * (ay2 - ay1);
            float bb2_area = (bx2 - bx1) * (by2 - by1);
            float iou = intersection_area / (bb1_area + bb2_area - intersection_area);

            return iou;
        }

        private static float[] Distance2Box(float[] distance, Point anchor_center, float rate)
        {
            distance[0] = -distance[0];
            distance[1] = -distance[1];
            return Distance2Point(distance, anchor_center, rate);
        }

        private static float[] Distance2Point(float[] distance, Point anchor_center, float rate)
        {
            for (int i = 0; i < distance.Length; i = i + 2)
            {
                distance[i] = (anchor_center.X + distance[i]) * rate;
                distance[i + 1] = (anchor_center.Y + distance[i + 1]) * rate;
            }
            return distance;
        }
    }
}